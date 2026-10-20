"""
src/paths.py
------------
Environment-aware path resolution.
Works identically on:
  - Local machine (auto-detects repo root via this file's location)
  - Google Colab  (mounts Drive; persists data across runtime resets)
  - Kaggle        (read inputs from /kaggle/input, write to /kaggle/working)

Usage
-----
from src.paths import Paths
p = Paths()
print(p.raw)        # Path to raw data dir
print(p.processed)  # Path to processed data dir
print(p.results)    # Path to results dir
"""

import os
import sys
from pathlib import Path


def _detect_env() -> str:
    if "google.colab" in sys.modules:
        return "colab"
    if os.path.exists("/kaggle/input"):
        return "kaggle"
    return "local"


class Paths:
    """Centralised paths, resolved once at import time."""

    def __init__(self, env: str | None = None):
        self.env = env or _detect_env()
        self.repo = self._find_repo()

        if self.env == "colab":
            # Mount Drive so data survives kernel resets
            self._mount_drive()
            base = Path("/content/drive/MyDrive/thesis_data")
        elif self.env == "kaggle":
            # Write outputs to /kaggle/working; read datasets from /kaggle/input
            base = Path("/kaggle/working/thesis_data")
            self.kaggle_input = Path("/kaggle/input")
        else:
            base = self.repo / "data"

        self.raw = base / "raw"
        self.interim = base / "interim"
        self.processed = base / "processed"

        if self.env == "colab":
            self.results = Path("/content/drive/MyDrive/thesis_results")
        elif self.env == "kaggle":
            self.results = Path("/kaggle/working/thesis_results")
        else:
            self.results = self.repo / "results"

        self.checkpoints = self.results / "checkpoints"
        self.preds = self.results / "preds"
        self.figures = self.results / "figures"
        self.configs = self.repo / "configs"

        # Create directories if they don't exist
        for d in [self.raw, self.interim, self.processed,
                  self.results, self.checkpoints, self.preds, self.figures]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _find_repo(self) -> Path:
        """Walk up from this file until we find configs/config.yaml."""
        candidate = Path(__file__).resolve().parent
        for _ in range(5):
            if (candidate / "configs" / "config.yaml").exists():
                return candidate
            candidate = candidate.parent
        # Fallback: current working directory
        return Path.cwd()

    def _mount_drive(self):
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive", force_remount=False)
        except Exception:
            pass  # Already mounted or not on Colab

    # ------------------------------------------------------------------
    def find_input_parquet(self, symbol: str) -> Path | None:
        """Kaggle: search /kaggle/input recursively for {symbol}_15m.parquet."""
        if self.env != "kaggle":
            return None
        root = getattr(self, "kaggle_input", Path("/kaggle/input"))
        if not root.exists():
            return None
        for p in root.rglob(f"{symbol}_15m.parquet"):
            return p
        return None

    def raw_parquet(self, symbol: str) -> Path:
        own = self.raw / f"{symbol}_15m.parquet"
        if own.exists():
            return own
        found = self.find_input_parquet(symbol)
        return found if found is not None else own

    def interim_parquet(self, symbol: str) -> Path:
        return self.interim / f"{symbol}_clean.parquet"

    def processed_npz(self, symbol: str, feature_set: str) -> Path:
        """feature_set: 'univariate' or 'multivariate'"""
        return self.processed / f"{symbol}_{feature_set}.npz"

    def checkpoint_path(self, symbol: str, seed: int) -> Path:
        return self.checkpoints / f"hblstm_{symbol}_seed{seed}.pt"

    def preds_path(self, symbol: str, feature_set: str, seed: int) -> Path:
        return self.preds / f"{symbol}_{feature_set}_seed{seed}.parquet"

    def __repr__(self) -> str:
        return (
            f"Paths(env={self.env!r}, repo={self.repo}, "
            f"raw={self.raw}, results={self.results})"
        )