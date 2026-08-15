"""
src/metrics.py
--------------
Evaluation metrics for H.BLSTM reproduction and thesis experiments.

Metrics implemented
-------------------
  MAE   — Mean Absolute Error (raw price units)
  RMSE  — Root Mean Squared Error (raw price units)
  MAPE  — Mean Absolute Percentage Error (matches base paper formula)
  MASE  — Mean Absolute Scaled Error  (primary RQ1 metric; < 1 = skill)
  DA    — Directional Accuracy (primary RQ1 metric; > 0.5 = predictive)

All metrics operate on raw (inverse-transformed) price values.
Pass scaled values → wrong numbers; the comparison table will not
match the paper's Table 2/3.

Usage
-----
from src.metrics import evaluate, compare_to_paper

results = evaluate(y_true, y_pred, symbol="SPX")
compare_to_paper(results)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Paper's Table 2/3 reference values (multivariate H.BLSTM)
_PAPER_RESULTS = {
    "SPX":    {"rmse": 8.574,   "mae": 5.617,   "mape": 0.001},
    "NIFTY":  {"rmse": 21.805,  "mae": 16.300,  "mape": 0.001},
    "FTSE":   {"rmse": 3.730,   "mae": 2.584,   "mape": 0.001},
    "DJI":    {"rmse": 66.282,  "mae": 45.743,  "mape": 0.001},
    "DXY":    {"rmse": 0.069,   "mae": 0.049,   "mape": 0.000},
    "NDX":    {"rmse": 37.851,  "mae": 26.268,  "mape": 0.002},
    "NI225":  {"rmse": 78.897,  "mae": 44.604,  "mape": 0.002},
    "SSE":    {"rmse": 7.206,   "mae": 5.315,   "mape": 0.002},
    "DAXEUR": {"rmse": 14.102,  "mae": 10.602,  "mape": 0.001},
}

# ──────────────────────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPE as defined in the base paper (Section 4.4):
      MAPE = (100% / n) * Σ |actual - predicted| / |actual|

    Returns a value in [0, 1] range (i.e. 0.001, not 0.1%).
    Clips denominator at 1e-8 to avoid division by zero.
    """
    return float(
        np.mean(np.abs(y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))
    )

def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_naive: np.ndarray | None = None,
) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    MASE = MAE(model) / MAE(naive)

    If y_naive is None, the naive persistence forecast is used:
      y_naive[t] = y_true[t-1]
    which means naive MAE = mean(|y_true[t] - y_true[t-1]|).

    Interpretation:
      MASE < 1  → model is better than naive persistence  (skill)
      MASE ≈ 1  → model is equivalent to naive persistence (no skill)
      MASE > 1  → model is worse than naive persistence
    """
    mae_model = mae(y_true, y_pred)

    if y_naive is not None:
        mae_naive = mae(y_true, y_naive)
    else:
        # Persistence forecast: MAE of naive y[t] = y[t-1]
        mae_naive = float(np.mean(np.abs(np.diff(y_true))))

    if mae_naive < 1e-10:
        return float("nan")

    return mae_model / mae_naive

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of bars where the predicted direction (up/down) matches the actual.

    DA = (1/n) * Σ 1[sign(ŷ_t - ŷ_{t-1}) == sign(y_t - y_{t-1})]

    DA = 0.5  → random (coin-flip)
    DA > 0.5  → some directional predictive power
    """
    actual_dir = np.sign(np.diff(y_true))
    pred_dir   = np.sign(np.diff(y_pred))
    return float(np.mean(actual_dir == pred_dir))

def naive_persistence(y_true: np.ndarray) -> np.ndarray:
    """y_naive[t] = y_true[t-1].  Returns array of same length (first element = NaN)."""
    naive = np.empty_like(y_true)
    naive[0] = np.nan
    naive[1:] = y_true[:-1]
    return naive

# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    symbol: str = "",
    feature_set: str = "",
    seed: int | None = None,
    extra: dict | None = None,
) -> dict:
    """
    Compute all metrics and return a flat result dict.

    Parameters
    ----------
    y_true      : (N,) actual close prices (raw scale)
    y_pred      : (N,) predicted close prices (raw scale)
    symbol      : index symbol string (for display)
    feature_set : "univariate" or "multivariate"
    seed        : random seed used for this run
    extra       : any additional fields to include in the dict

    Returns
    -------
    dict with keys: symbol, feature_set, seed, mae, rmse, mape, mase, da
    """
    result = {
        "symbol": symbol,
        "feature_set": feature_set,
        "seed": seed,
        "mae":  mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mase": mase(y_true, y_pred),
        "da":   directional_accuracy(y_true, y_pred),
        "n":    len(y_true),
    }
    if extra:
        result.update(extra)
    return result

# ──────────────────────────────────────────────────────────────────────────────

def compare_to_paper(
    results: list[dict] | dict,
    feature_set: str = "multivariate",
) -> pd.DataFrame:
    """
    Print a side-by-side comparison with the paper's Table 2/3 values.

    Parameters
    ----------
    results     : single result dict or list of dicts (multiple seeds)
                  When multiple seeds are passed, reports mean ± std.
    feature_set : used to select the reference column label

    Returns
    -------
    DataFrame for display in a notebook
    """
    if isinstance(results, dict):
        results = [results]

    # Aggregate across seeds
    df = pd.DataFrame(results)
    agg = df.groupby("symbol").agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mape_mean=("mape", "mean"), mape_std=("mape", "std"),
        mase_mean=("mase", "mean"),
        da_mean=("da", "mean"),
    )

    rows = []
    for sym, row in agg.iterrows():
        paper = _PAPER_RESULTS.get(sym, {})
        rows.append({
            "Symbol": sym,
            # Reproduced
            "RMSE (ours)": f"{row['rmse_mean']:.3f} ± {row['rmse_std']:.3f}"
                            if not np.isnan(row['rmse_std']) else f"{row['rmse_mean']:.3f}",
            "MAE (ours)":  f"{row['mae_mean']:.3f} ± {row['mae_std']:.3f}"
                            if not np.isnan(row['mae_std']) else f"{row['mae_mean']:.3f}",
            "MAPE (ours)": f"{row['mape_mean']:.4f}",
            # Paper reference
            "RMSE (paper)": paper.get("rmse", "—"),
            "MAE (paper)":  paper.get("mae", "—"),
            "MAPE (paper)": paper.get("mape", "—"),
            # RQ1 metrics
            "MASE": f"{row['mase_mean']:.3f}",
            "DA":   f"{row['da_mean']:.3f}",
        })

    cmp_df = pd.DataFrame(rows).set_index("Symbol")
    print(cmp_df.to_string())
    return cmp_df

# ──────────────────────────────────────────────────────────────────────────────

def summarise_runs(results_list: list[dict]) -> pd.DataFrame:
    """
    Aggregate a list of result dicts (from multiple symbols × seeds)
    into a summary DataFrame grouped by (symbol, feature_set).
    """
    df = pd.DataFrame(results_list)
    summary = (
        df.groupby(["symbol", "feature_set"])
        .agg(
            n_seeds=("seed", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mape_mean=("mape", "mean"),
            mase_mean=("mase", "mean"),
            da_mean=("da", "mean"),
        )
        .reset_index()
    )
    return summary