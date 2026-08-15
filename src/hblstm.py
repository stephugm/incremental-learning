"""
src/hblstm.py
-------------
HBLSTMRunner — the hybrid incremental runner that reproduces the
exact learning mechanism from Kalra et al. (2024):

  Phase 0  — batch-train on historical (train) data, max 50 epochs
  Phase 1  — per-bar incremental update as each test bar arrives
  Phase 2  — full retrain on ALL accumulated data at end of each session

The runner is designed to be policy-aware from the start:
  update_policy = "fixed"   → end-of-session full retrain (base paper)
  update_policy = "none"    → no updates after Phase 0 (frozen model)
  update_policy = "drift"   → retrain only when drift detector fires (RQ3)

All predictions and cost metrics are logged per-bar and saved to disk,
because:
  - MASE and Diebold-Mariano tests (RQ1) need per-bar residuals
  - Pareto plot (RQ3) needs per-session retrain cost

Checkpoint/resume: the runner saves state after every N sessions so
long Colab/Kaggle runs can be interrupted and resumed safely.
"""

from __future__ import annotations

import time
import tracemalloc
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model import (
    BiLSTMModel,
    IncrementalUpdater,
    Trainer,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)
from src.data import inverse_scale_y
from sklearn.preprocessing import MinMaxScaler

# ──────────────────────────────────────────────────────────────────────────────

class HBLSTMRunner:
    """
    Hybrid BiLSTM runner.

    Parameters
    ----------
    model_kwargs    : passed to BiLSTMModel(...)
    trainer_kwargs  : passed to Trainer(...)
    incremental_lr  : learning rate for per-bar updates
    update_policy   : "fixed" | "none" | "drift"
    checkpoint_every: save checkpoint every N sessions (0 = disabled)
    device          : "auto" | "cpu" | "cuda"
    seed            : random seed for reproducibility
    verbose         : print progress
    """

    def __init__(
        self,
        model_kwargs: dict | None = None,
        trainer_kwargs: dict | None = None,
        incremental_lr: float = 1e-4,
        update_policy: str = "fixed",
        checkpoint_every: int = 5,
        device: str = "auto",
        seed: int = 42,
        verbose: bool = True,
    ):
        assert update_policy in ("fixed", "none", "drift"), \
            f"Unknown update_policy: {update_policy!r}"

        self.model_kwargs = model_kwargs or {}
        self.trainer_kwargs = trainer_kwargs or {}
        self.incremental_lr = incremental_lr
        self.update_policy = update_policy
        self.checkpoint_every = checkpoint_every
        self.device = get_device(device)
        self.seed = seed
        self.verbose = verbose

        self.model: BiLSTMModel | None = None
        self.trainer: Trainer | None = None
        self.updater: IncrementalUpdater | None = None

        # Accumulated data buffers (raw-scale, unscaled) for end-of-session retrain
        self._X_accum: list[np.ndarray] = []
        self._y_accum: list[float] = []

        # Per-bar prediction log (built during run, saved to disk)
        self._log: list[dict] = []

        # Cost metrics
        self._retrain_count = 0
        self._retrain_wall_clock: list[float] = []
        self._retrain_peak_mem_mb: list[float] = []

        # Resume state
        self._last_completed_session: int = -1

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit_initial(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict:
        """
        Phase 0: batch-train on the historical training set.

        X_train, y_train must be already scaled.
        """
        set_seed(self.seed)

        input_size = X_train.shape[2]
        self.model = BiLSTMModel(input_size=input_size, **self.model_kwargs).to(self.device)
        self.trainer = Trainer(device=self.device, verbose=self.verbose, **self.trainer_kwargs)
        self.updater = IncrementalUpdater(lr=self.incremental_lr, device=self.device)

        print(f"\n[runner] Phase 0 — initial batch training ({len(X_train)} samples) ...")
        history = self.trainer.fit(self.model, X_train, y_train)

        # Seed accumulation buffer with train data (unscaled not needed for retrain;
        # we accumulate scaled arrays — consistent with what scaler was fit on)
        self._X_accum = [X_train]
        self._y_accum_arr = [y_train]

        return history

    def run_test(
        self,
        X_test: np.ndarray,         # (N, lag, F) scaled
        y_test: np.ndarray,         # (N,) scaled
        y_test_raw: np.ndarray,     # (N,) original price scale (for metrics)
        timestamps: pd.DatetimeIndex,
        session_ids: np.ndarray,    # (N,) string, one per bar
        scaler_y: MinMaxScaler,
        checkpoint_path: Path | None = None,
        resume_session: int = -1,
    ) -> pd.DataFrame:
        """
        Phase 1+2: walk bar-by-bar through the test set.

        For each bar:
          1. Predict (scaled)
          2. Log prediction + latency
          3. Incremental update with true label
          4. If session boundary → end-of-session retrain (if policy="fixed")
          5. Optionally checkpoint

        Returns a DataFrame with one row per test bar:
          timestamp, y_true_raw, y_pred_raw, session_id,
          infer_ms, incremental_loss,
          retrain_triggered (bool)
        """
        assert self.model is not None, "Call fit_initial() first."

        self.model.eval()
        n = len(X_test)
        unique_sessions = pd.unique(session_ids)

        print(
            f"\n[runner] Phase 1+2 — walk-forward test "
            f"({n} bars, {len(unique_sessions)} sessions, policy={self.update_policy!r})"
        )

        self._log = []

        for i in range(n):
            # Skip already-processed bars if resuming
            if i <= resume_session:
                continue

            x_i = X_test[i : i + 1]                  # (1, lag, F)
            y_scaled_true = float(y_test[i])
            y_raw_true = float(y_test_raw[i])
            sid = session_ids[i]

            # ── 1. Predict ──────────────────────────────────────────────────
            t_infer_start = time.perf_counter()
            self.model.eval()
            with torch.no_grad():
                xb = torch.from_numpy(x_i).to(self.device)
                y_scaled_pred = self.model(xb).item()
            infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

            # Inverse-transform to price scale
            y_raw_pred = float(
                inverse_scale_y(np.array([y_scaled_pred], dtype=np.float32), scaler_y)[0]
            )

            # ── 2. Incremental update ────────────────────────────────────────
            inc_loss = 0.0
            if self.update_policy != "none":
                inc_loss = self.updater.step(self.model, x_i, y_scaled_true)

            # ── 3. Accumulate for end-of-session retrain ─────────────────────
            self._X_accum.append(x_i)
            self._y_accum_arr.append(np.array([y_scaled_true], dtype=np.float32))

            # ── 4. Check session boundary ────────────────────────────────────
            retrain_triggered = False
            is_last_bar_in_session = (
                i == n - 1 or session_ids[i + 1] != sid
            )

            if is_last_bar_in_session and self.update_policy == "fixed":
                retrain_triggered = True
                self._do_retrain()
                self._last_completed_session = i
                if (
                    self.checkpoint_every > 0
                    and self._retrain_count % self.checkpoint_every == 0
                    and checkpoint_path is not None
                ):
                    self._save_runner_checkpoint(checkpoint_path, i)

            # ── 5. Log ───────────────────────────────────────────────────────
            self._log.append({
                "timestamp": timestamps[i],
                "session_id": sid,
                "y_true_raw": y_raw_true,
                "y_pred_raw": y_raw_pred,
                "infer_ms": infer_ms,
                "incremental_loss": inc_loss,
                "retrain_triggered": retrain_triggered,
            })

            if self.verbose and i % 500 == 0:
                print(
                    f"  bar {i:5d}/{n} | session {sid} | "
                    f"pred={y_raw_pred:.2f} true={y_raw_true:.2f} | "
                    f"infer={infer_ms:.1f}ms"
                )

        df_preds = pd.DataFrame(self._log).set_index("timestamp")
        print(
            f"\n[runner] Done. Retrains: {self._retrain_count} | "
            f"Avg infer latency: {df_preds['infer_ms'].mean():.1f} ms"
        )
        return df_preds

    # ──────────────────────────────────────────────────────────────────────────
    # Cost report
    # ──────────────────────────────────────────────────────────────────────────

    def cost_report(self) -> dict:
        """
        Return a dict of RQ3 cost metrics.

        Keys:
          retrain_count         : total number of end-of-session retrains
          total_retrain_sec     : cumulative wall-clock retrain time (s)
          mean_retrain_sec      : mean per-retrain wall-clock (s)
          peak_mem_mb           : peak memory during any single retrain (MB)
        """
        return {
            "retrain_count": self._retrain_count,
            "total_retrain_sec": sum(self._retrain_wall_clock),
            "mean_retrain_sec": (
                np.mean(self._retrain_wall_clock) if self._retrain_wall_clock else 0.0
            ),
            "peak_mem_mb": max(self._retrain_peak_mem_mb, default=0.0),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _do_retrain(self):
        """Full retrain on all accumulated data (end-of-session step)."""
        X_all = np.concatenate(self._X_accum, axis=0)
        y_all = np.concatenate(self._y_accum_arr, axis=0)

        tracemalloc.start()
        t0 = time.perf_counter()

        self.trainer.retrain(self.model, X_all, y_all)

        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1e6
        self._retrain_count += 1
        self._retrain_wall_clock.append(elapsed)
        self._retrain_peak_mem_mb.append(peak_mb)

        # Rebuild incremental optimizer so its state is fresh after retrain
        self.updater = IncrementalUpdater(lr=self.incremental_lr, device=self.device)

        if self.verbose:
            print(
                f"  [retrain #{self._retrain_count}] "
                f"{elapsed:.1f}s | {len(X_all)} samples | peak {peak_mb:.1f} MB"
            )

    def _save_runner_checkpoint(self, path: Path, bar_idx: int):
        """Save model weights + runner state for resume."""
        meta = {
            "bar_idx": bar_idx,
            "retrain_count": self._retrain_count,
            "retrain_wall_clock": self._retrain_wall_clock,
            "retrain_peak_mem_mb": self._retrain_peak_mem_mb,
            "log_so_far": self._log,
        }
        save_checkpoint(self.model, path, metadata=meta)

    @classmethod
    def resume_from_checkpoint(
        cls,
        path: Path,
        model_kwargs: dict,
        input_size: int,
        **runner_kwargs,
    ) -> tuple["HBLSTMRunner", int, list[dict]]:
        """
        Load a runner checkpoint.

        Returns
        -------
        runner          : HBLSTMRunner with model weights restored
        resume_bar_idx  : last bar index that was completed
        log_so_far      : predictions logged before the checkpoint
        """
        runner = cls(model_kwargs=model_kwargs, **runner_kwargs)
        runner.model = BiLSTMModel(input_size=input_size, **model_kwargs)
        meta = load_checkpoint(runner.model, path)

        runner._retrain_count = meta.get("retrain_count", 0)
        runner._retrain_wall_clock = meta.get("retrain_wall_clock", [])
        runner._retrain_peak_mem_mb = meta.get("retrain_peak_mem_mb", [])
        runner._log = meta.get("log_so_far", [])

        return runner, meta.get("bar_idx", -1), runner._log