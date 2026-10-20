"""
src/data.py
-----------
Data pipeline for H.BLSTM reproduction.

Stages
------
1. download   – fetch 15-min bars via cached parquet / tvkit (fallback: rolling yfinance)
2. clean      – null/duplicate removal, chronological sort, tz-aware timestamps
3. sessions   – tag each bar with session_id (local trading date) + bar index
4. features   – causal incremental EMA5 (no look-ahead leakage)
5. windows    – sliding lag-3 windows → X (N, lag, F), y (N,)
6. split      – strict chronological 70/30, scaler fit on train only

All functions accept explicit path arguments so they work identically
across local / Colab / Kaggle environments (see src/paths.py).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


# ──────────────────────────────────────────────────────────────────────────────
# 1. Download
# ──────────────────────────────────────────────────────────────────────────────

def _tvkit_fetch(exchange_symbol: str, n_bars: int) -> pd.DataFrame:
    """Fetch bars with tvkit (async WebSocket). Local machine only."""
    import asyncio
    import concurrent.futures

    from tvkit.api.chart.ohlcv import OHLCV  # type: ignore

    async def _run():
        async with OHLCV() as client:
            return await client.get_historical_ohlcv(
                exchange_symbol=exchange_symbol, interval="15", bars_count=n_bars
            )

    try:
        asyncio.get_running_loop()
        in_loop = True          # e.g. inside Jupyter
    except RuntimeError:
        in_loop = False

    if in_loop:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            bars = pool.submit(lambda: asyncio.run(_run())).result()
    else:
        bars = asyncio.run(_run())

    if not bars:
        raise ValueError("tvkit returned empty result.")
    rows = [{
        "dt": pd.Timestamp(b.timestamp, unit="s", tz="UTC"),
        "open": b.open, "high": b.high, "low": b.low,
        "close": b.close, "volume": b.volume,
    } for b in bars]
    df = pd.DataFrame(rows).set_index("dt").sort_index()
    df.index.name = None
    return df[~df.index.duplicated(keep="first")]


def download_tvkit(
    symbol: str,
    exchange: str,
    n_bars: int = 6000,
    save_path: Path | None = None,
    force: bool = False,
    tv_symbol: str | None = None,
) -> pd.DataFrame:
    """
    Load 15-min OHLCV bars: cached parquet → tvkit → rolling yfinance.

    Prefer producing the parquet with notebooks/05_download_tradingview.ipynb
    (local machine, authenticated) and uploading it as a Kaggle dataset;
    tvkit's WebSocket is blocked on Kaggle.

    Parameters
    ----------
    symbol    : index symbol, e.g. "SPX"
    exchange  : TradingView exchange/prefix, e.g. "SP"
    n_bars    : number of historical bars to download
    save_path : if given, save raw parquet here
    force     : re-download even if save_path already exists
    tv_symbol : full tvkit symbol "EXCHANGE:SYMBOL"; default f"{exchange}:{symbol}"

    Returns
    -------
    DataFrame with columns [open, high, low, close, volume],
    DatetimeIndex (UTC).
    """
    if save_path and save_path.exists() and not force:
        print(f"[download] Loading cached {symbol} from {save_path}")
        return pd.read_parquet(save_path)

    try:
        df = _tvkit_fetch(tv_symbol or f"{exchange}:{symbol}", n_bars)
        print(f"[download] tvkit OK: {symbol} — {len(df)} bars")
    except Exception as exc:
        warnings.warn(
            f"tvkit failed ({exc}). Falling back to rolling yfinance.", stacklevel=2
        )
        df = _download_yfinance_rolling(symbol, n_bars)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        print(f"[download] Saved → {save_path}")

    return df


def _download_yfinance_rolling(symbol: str, n_bars: int) -> pd.DataFrame:
    """
    yfinance only serves 60 days of 15-min data per pull.
    We roll backwards in 58-day chunks until we have enough bars.
    """
    import yfinance as yf  # type: ignore

    # Map index symbols to yfinance tickers
    _TICKER_MAP = {"SPX": "^GSPC", "NIFTY": "^NSEI", "FTSE": "^FTSE",
                   "DJI": "^DJI", "NDX": "^NDX", "NI225": "^N225",
                   "SSE": "000001.SS", "DAXEUR": "^GDAXI", "DXY": "DX-Y.NYB"}
    ticker = _TICKER_MAP.get(symbol, symbol)

    chunks: list[pd.DataFrame] = []
    end = pd.Timestamp.now(tz="UTC")          # fix: utcnow() deprecated in pandas 2.x
    MAX_ITERATIONS = 20                        # guard: yfinance only has ~60 days anyway
    for _ in range(MAX_ITERATIONS):
        start = end - pd.Timedelta(days=58)
        chunk = yf.download(
            ticker, start=start.date(), end=end.date(),
            interval="15m", progress=False, auto_adjust=True,
        )
        if chunk.empty:
            break
        chunk.index = pd.to_datetime(chunk.index, utc=True)
        # fix: yfinance ≥0.2.51 returns MultiIndex columns — flatten before .lower()
        if isinstance(chunk.columns, pd.MultiIndex):
            chunk.columns = [c[0].lower() for c in chunk.columns]
        else:
            chunk.columns = [str(c).lower() for c in chunk.columns]
        chunks.append(chunk[["open", "high", "low", "close", "volume"]])
        end = start
        total = sum(len(c) for c in chunks)
        print(f"[yfinance] {symbol}: {total} bars collected …")
        if total >= n_bars:
            break

    if not chunks:
        raise RuntimeError(f"Could not download any data for {symbol}.")

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    print(f"[yfinance] {symbol} — {len(df)} bars (rolling fallback)")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Clean
# ──────────────────────────────────────────────────────────────────────────────

def clean(
    df: pd.DataFrame,
    tz: str,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Clean raw OHLCV DataFrame.

    Steps:
    - Drop rows with null close
    - Drop duplicate timestamps (keep first)
    - Sort chronologically
    - Convert index to exchange-local timezone
    - Keep only columns [open, high, low, close, volume]

    Returns a quality report dict as a side-effect (printed to stdout).
    """
    n_raw = len(df)

    df = df.dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # Convert to exchange-local timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(tz)

    df = df[["open", "high", "low", "close", "volume"]].copy()

    # Quality report
    n_clean = len(df)
    date_range = f"{df.index[0].date()} → {df.index[-1].date()}"
    print(
        f"[clean] {n_raw} → {n_clean} bars | dropped {n_raw - n_clean} | "
        f"range: {date_range} | tz: {tz}"
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        print(f"[clean] Saved → {save_path}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Sessions
# ──────────────────────────────────────────────────────────────────────────────

def tag_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two columns:
      - session_id      : local date string ("2023-01-03"), uniquely identifies a trading day
      - bar_in_session  : 0-based bar index within that trading day

    Used by HBLSTMRunner to know when a session ends (trigger batch retrain).
    """
    df = df.copy()
    df["session_id"] = df.index.date.astype(str)
    df["bar_in_session"] = df.groupby("session_id").cumcount()
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. Features — Causal incremental EMA (no look-ahead)
# ──────────────────────────────────────────────────────────────────────────────

class IncrementalEMA:
    """
    Causal recursive EMA.

    ema_t = β * x_t + (1 - β) * ema_{t-1}
    β = 2 / (period + 1)

    Call update(x) for each new bar in order.
    This is identical to the EMA used in the paper (Section 3.1).

    The key property: only past + current values are used → zero look-ahead.
    Contrast with pandas.ewm() applied to the full series after splitting,
    which would leak future information.
    """

    def __init__(self, period: int):
        self.beta = 2.0 / (period + 1)
        self._ema: float | None = None

    def update(self, x: float) -> float:
        if self._ema is None:
            self._ema = x
        else:
            self._ema = self.beta * x + (1.0 - self.beta) * self._ema
        return self._ema

    def reset(self):
        self._ema = None


def add_ema5(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """
    Compute EMA5 causally over the entire series and add as a column.
    The EMA is initialised with the first close value (warm-up = 1 bar).
    """
    df = df.copy()
    ema = IncrementalEMA(period)
    df["ema5"] = [ema.update(c) for c in df["close"]]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5. Windows
# ──────────────────────────────────────────────────────────────────────────────

def make_windows(
    df: pd.DataFrame,
    features: Sequence[str],
    lag: int = 3,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build supervised learning windows.

    Returns
    -------
    X : (N, lag, len(features))  — input tensor
    y : (N,)                     — next-bar close price (raw, unscaled)
    timestamps : (N,)            — timestamp of the *target* bar
    """
    data = df[list(features)].values.astype(np.float32)
    close_col = list(features).index("close")

    X_list, y_list, ts_list = [], [], []
    for i in range(lag, len(data) - horizon + 1):
        X_list.append(data[i - lag : i])          # shape (lag, F)
        y_list.append(data[i + horizon - 1, close_col])  # next close
        ts_list.append(df.index[i + horizon - 1])

    X = np.stack(X_list)           # (N, lag, F)
    y = np.array(y_list)           # (N,)
    timestamps = pd.DatetimeIndex(ts_list)
    return X, y, timestamps


# ──────────────────────────────────────────────────────────────────────────────
# 6. Split + Scaling
# ──────────────────────────────────────────────────────────────────────────────

def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    train_ratio: float = 0.70,
) -> tuple:
    """
    Strict chronological split — no shuffling.

    Verifies that the last train index is strictly before the first test index
    (anti-leakage assertion).

    Returns
    -------
    X_train, y_train, ts_train, X_test, y_test, ts_test
    """
    n = len(X)
    n_train = int(n * train_ratio)

    X_train, y_train, ts_train = X[:n_train], y[:n_train], timestamps[:n_train]
    X_test,  y_test,  ts_test  = X[n_train:], y[n_train:], timestamps[n_train:]

    # Anti-leakage assertion
    assert ts_train[-1] < ts_test[0], (
        f"Leakage detected! Last train ts {ts_train[-1]} >= first test ts {ts_test[0]}"
    )
    print(
        f"[split] train: {len(X_train)} | test: {len(X_test)} | "
        f"train ends: {ts_train[-1].date()} | test starts: {ts_test[0].date()}"
    )
    return X_train, y_train, ts_train, X_test, y_test, ts_test


def fit_scaler(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler_path: Path | None = None,
) -> tuple[MinMaxScaler, MinMaxScaler]:
    """
    Fit two MinMaxScalers:
      - scaler_X : fitted on flattened X_train features
      - scaler_y : fitted on y_train (close price)

    Fit is performed on TRAIN ONLY — prevents future leakage into test set.
    Scalers are saved to disk if scaler_path is given.
    """
    n, lag, F = X_train.shape

    # Fit on 2-D view, transform back to 3-D
    scaler_X = MinMaxScaler()
    scaler_X.fit(X_train.reshape(-1, F))

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train.reshape(-1, 1))

    if scaler_path:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"X": scaler_X, "y": scaler_y}, scaler_path)
        print(f"[scaler] Saved → {scaler_path}")

    return scaler_X, scaler_y


def apply_scaler(
    X: np.ndarray,
    y: np.ndarray,
    scaler_X: MinMaxScaler,
    scaler_y: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform X and y using pre-fitted scalers."""
    n, lag, F = X.shape
    X_scaled = scaler_X.transform(X.reshape(-1, F)).reshape(n, lag, F)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()
    return X_scaled.astype(np.float32), y_scaled.astype(np.float32)


def inverse_scale_y(y_scaled: np.ndarray, scaler_y: MinMaxScaler) -> np.ndarray:
    """Inverse-transform predicted/actual values back to price scale."""
    return scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


# ──────────────────────────────────────────────────────────────────────────────
# 7. Full pipeline convenience function
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(
    symbol: str,
    exchange: str,
    tz: str,
    feature_set: str,               # "univariate" or "multivariate"
    paths,                           # src.paths.Paths instance
    n_bars: int = 6000,
    lag: int = 3,
    train_ratio: float = 0.70,
    force_download: bool = False,
    ema5_period: int = 5,
) -> dict:
    """
    End-to-end: download → clean → sessions → features → windows → split → scale.

    Returns a dict with keys:
      X_train, y_train, ts_train,
      X_test,  y_test,  ts_test,
      scaler_X, scaler_y,
      df_clean,          (full cleaned DataFrame with session tags + ema5)
      feature_names,
      symbol, feature_set
    """
    features = ["close"] if feature_set == "univariate" else ["close", "ema5"]

    # 1. Download
    df_raw = download_tvkit(
        symbol=symbol, exchange=exchange, n_bars=n_bars,
        save_path=paths.raw_parquet(symbol), force=force_download,
    )

    # 2. Clean
    df = clean(df_raw, tz=tz, save_path=paths.interim_parquet(symbol))

    # 3. Sessions
    df = tag_sessions(df)

    # 4. Features (always compute EMA5 so df is consistent, used or not)
    df = add_ema5(df, period=ema5_period)

    # 5. Windows
    X, y, timestamps = make_windows(df, features=features, lag=lag)

    # 6. Split
    X_train, y_train, ts_train, X_test, y_test, ts_test = chronological_split(
        X, y, timestamps, train_ratio=train_ratio
    )

    # 7. Scale (fit on train only)
    scaler_path = paths.processed / f"{symbol}_{feature_set}_scaler.pkl"
    scaler_X, scaler_y = fit_scaler(X_train, y_train, scaler_path=scaler_path)

    X_train_s, y_train_s = apply_scaler(X_train, y_train, scaler_X, scaler_y)
    X_test_s,  y_test_s  = apply_scaler(X_test,  y_test,  scaler_X, scaler_y)

    # Save processed arrays
    npz_path = paths.processed_npz(symbol, feature_set)
    np.savez(
        npz_path,
        X_train=X_train_s, y_train=y_train_s, ts_train=ts_train.astype(str),
        X_test=X_test_s,   y_test=y_test_s,   ts_test=ts_test.astype(str),
    )
    print(f"[pipeline] Saved processed arrays → {npz_path}")

    return dict(
        X_train=X_train_s, y_train=y_train_s, ts_train=ts_train,
        X_test=X_test_s,   y_test=y_test_s,   ts_test=ts_test,
        X_train_raw=X_train, y_train_raw=y_train,
        X_test_raw=X_test,   y_test_raw=y_test,
        scaler_X=scaler_X, scaler_y=scaler_y,
        df_clean=df,
        feature_names=features,
        symbol=symbol,
        feature_set=feature_set,
    )