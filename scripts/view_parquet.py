"""
View parquet files in data/raw/ (output of notebooks/05_download_tradingview.ipynb).

Usage:
    python scripts/view_parquet.py              # summary of all files
    python scripts/view_parquet.py SPX          # head/tail of SPX_15m.parquet
    python scripts/view_parquet.py SPX --rows 10
    python scripts/view_parquet.py SPX --csv    # also export SPX_15m.csv
"""
import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def summary() -> None:
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        print(f"Tidak ada file .parquet di {RAW_DIR}")
        return
    for f in files:
        df = pd.read_parquet(f)
        print(f"{f.name:25s} {f.stat().st_size / 1024:8.1f} KB  {len(df):6d} bar  "
              f"{df.index[0]} -> {df.index[-1]}  tz={df.index.tz}")


def show(symbol: str, rows: int, to_csv: bool) -> None:
    path = RAW_DIR / f"{symbol}_15m.parquet"
    if not path.exists():
        print(f"File tidak ditemukan: {path}")
        return
    df = pd.read_parquet(path)
    print(f"File    : {path}")
    print(f"Shape   : {df.shape}")
    print(f"Columns : {list(df.columns)}")
    print(f"Index   : {df.index.min()} -> {df.index.max()}  (tz={df.index.tz})")
    print(f"Nulls   : {int(df.isnull().sum().sum())}")
    print(f"\nHead({rows}):\n{df.head(rows).to_string()}")
    print(f"\nTail({rows}):\n{df.tail(rows).to_string()}")
    print(f"\nDescribe:\n{df.describe().to_string()}")
    if to_csv:
        out = path.with_suffix(".csv")
        df.to_csv(out)
        print(f"\nCSV disimpan: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Lihat file parquet di data/raw/")
    p.add_argument("symbol", nargs="?", help="SPX, NIFTY, atau FTSE (kosong = ringkasan semua)")
    p.add_argument("--rows", type=int, default=5, help="jumlah baris head/tail")
    p.add_argument("--csv", action="store_true", help="ekspor juga ke CSV")
    args = p.parse_args()

    if args.symbol:
        show(args.symbol.upper(), args.rows, args.csv)
    else:
        summary()
