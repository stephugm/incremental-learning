"""
src/model.py
------------
BiLSTM architecture and batch trainer for H.BLSTM reproduction.

Architecture (matches paper Section 4.3):
  Input  (batch, lag, F)
  → 2-layer Bidirectional LSTM
  → hidden state of last time-step, both directions concatenated
  → Linear(hidden*2 → hidden)  + ReLU
  → Linear(hidden → 1)

Optimizer : Adam
Loss      : MSE
Activation: ReLU
Max epochs: 50  (from paper / config)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────────────────────────────────────
# Device helper
# ──────────────────────────────────────────────────────────────────────────────

def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


# ──────────────────────────────────────────────────────────────────────────────
# Seed helper
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class BiLSTMModel(nn.Module):
    """
    2-layer Bidirectional LSTM with a 2-layer dense head.

    Parameters
    ----------
    input_size  : number of features F (1 for univariate, 2 for multivariate)
    hidden_size : LSTM hidden units per direction (default 64)
    num_layers  : number of stacked LSTM layers (default 2, matches paper)
    dropout     : dropout between LSTM layers (default 0.0)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dense head: concat of forward + backward → hidden → 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, lag, input_size)
        returns : (batch,)  — scalar prediction per sample
        """
        out, _ = self.bilstm(x)          # (batch, lag, hidden*2)
        last = out[:, -1, :]             # take last time-step
        return self.head(last).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Batch trainer (Phase 0 — initial training + end-of-session retraining)
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Stateless batch trainer.  Pass model + data; returns trained model.

    Mirrors the base paper's setup:
    - Adam optimiser
    - MSE loss
    - max 50 epochs
    - No early stopping (matches paper; add a val_loader argument to enable it)
    """

    def __init__(
        self,
        lr: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 32,
        device: torch.device | None = None,
        verbose: bool = True,
    ):
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device or get_device()
        self.verbose = verbose

    def fit(
        self,
        model: BiLSTMModel,
        X: np.ndarray,
        y: np.ndarray,
        val_X: np.ndarray | None = None,
        val_y: np.ndarray | None = None,
    ) -> dict:
        """
        Train model in-place on (X, y).

        Parameters
        ----------
        X     : (N, lag, F)  float32, already scaled
        y     : (N,)         float32, already scaled
        val_X : optional validation set for loss tracking only (not for early stop)

        Returns
        -------
        history dict with keys 'train_loss' (and 'val_loss' if val set given)
        """
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(
            torch.from_numpy(X).to(self.device),
            torch.from_numpy(y).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        has_val = val_X is not None and val_y is not None
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        t0 = time.perf_counter()
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(dataset)
            history["train_loss"].append(epoch_loss)

            if has_val:
                model.eval()
                with torch.no_grad():
                    xv = torch.from_numpy(val_X).to(self.device)
                    yv = torch.from_numpy(val_y).to(self.device)
                    val_loss = criterion(model(xv), yv).item()
                history["val_loss"].append(val_loss)

            if self.verbose and (epoch % 10 == 0 or epoch == 1):
                msg = f"  epoch {epoch:3d}/{self.max_epochs}  train_loss={epoch_loss:.6f}"
                if has_val:
                    msg += f"  val_loss={history['val_loss'][-1]:.6f}"
                print(msg)

        elapsed = time.perf_counter() - t0
        print(f"[trainer] Training complete in {elapsed:.1f}s ({self.max_epochs} epochs)")
        return history

    def retrain(
        self,
        model: BiLSTMModel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """
        Full retrain from scratch on accumulated data (end-of-session step).
        Reinitialises the optimiser; keeps the architecture.
        Same as fit() but named separately for clarity and timing purposes.
        """
        return self.fit(model, X, y)


# ──────────────────────────────────────────────────────────────────────────────
# Incremental updater (Phase 1 — per-bar update within a session)
# ──────────────────────────────────────────────────────────────────────────────

class IncrementalUpdater:
    """
    Single-sample gradient step with a reduced learning rate.

    Called once per bar during the test walk-forward loop,
    after the true label becomes available.
    """

    def __init__(self, lr: float = 1e-4, device: torch.device | None = None):
        self.lr = lr
        self.device = device or get_device()
        self._optimizer: torch.optim.Adam | None = None
        self._criterion = nn.MSELoss()

    def _ensure_optimizer(self, model: BiLSTMModel):
        # Rebuild optimizer if model parameters changed (e.g. after full retrain)
        self._optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

    def step(
        self,
        model: BiLSTMModel,
        x: np.ndarray,     # (1, lag, F)  single sample
        y_true: float,
    ) -> float:
        """
        One incremental gradient step.

        Returns the scalar loss for this step.
        """
        self._ensure_optimizer(model)
        model.train()
        model.to(self.device)

        xb = torch.from_numpy(x).to(self.device)            # (1, lag, F)
        yb = torch.tensor([y_true], dtype=torch.float32).to(self.device)

        self._optimizer.zero_grad()
        pred = model(xb)
        loss = self._criterion(pred, yb)
        loss.backward()
        self._optimizer.step()

        return loss.item()


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: BiLSTMModel,
    path: Path,
    metadata: dict | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, path)
    print(f"[checkpoint] Saved → {path}")


def load_checkpoint(model: BiLSTMModel, path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    print(f"[checkpoint] Loaded ← {path}")
    return ckpt.get("metadata", {})