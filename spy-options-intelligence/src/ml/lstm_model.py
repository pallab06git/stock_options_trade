# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""PyTorch LSTM model for options trade signal classification.

Architecture
------------
* ``OptionsSequenceDataset``  — PyTorch Dataset that produces sliding-window
  sequences ``(seq_len, n_features)`` from a 2-D feature array.
* ``LSTMModel``               — Stacked LSTM + fully-connected classifier.
* ``LSTMTrainer``             — Training loop with early stopping, class
  weighting, and a ``predict_proba()`` adapter compatible with the
  ``ModelComparator`` / ``TradeSimulator`` interface.

predict_proba alignment
-----------------------
Because LSTM sequences require at least ``seq_len`` historical bars, the first
``seq_len - 1`` rows of the input cannot produce a valid prediction.
``LSTMTrainer.predict_proba()`` pads those rows with ``0.5`` (neutral
probability) so the output length always equals the input DataFrame length.
This keeps index alignment intact for downstream trade simulation.

Graceful PyTorch handling
--------------------------
If ``torch`` is not installed, importing this module raises a clear
``ImportError`` with installation instructions rather than a cryptic
``ModuleNotFoundError`` deep inside PyTorch internals.

Usage
-----
    from src.ml.lstm_model import LSTMModel, LSTMTrainer, OptionsSequenceDataset

    trainer = LSTMTrainer(
        input_size=len(feature_cols),
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        seq_len=20,
        lr=1e-3,
        epochs=50,
        batch_size=64,
        patience=10,
        pos_weight_factor=10.0,
    )
    trainer.fit(X_train, y_train)
    proba = trainer.predict_proba(X_test)   # shape (n_test,)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful torch import
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    """Raise a clear ImportError if PyTorch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for LSTM models but is not installed. "
            "Install it with:  pip install torch"
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class OptionsSequenceDataset:
    """Sliding-window sequence dataset for options bar data.

    Produces ``(X_seq, y_seq)`` pairs where each ``X_seq[i]`` is a 2-D array
    of shape ``(seq_len, n_features)`` containing the feature window ending at
    bar ``i + seq_len - 1``, and ``y_seq[i]`` is the label at bar
    ``i + seq_len - 1``.

    This is intentionally **not** a PyTorch ``Dataset`` subclass so the class
    can be imported even without PyTorch.  ``LSTMTrainer.fit()`` wraps the
    arrays in a torch-native dataset internally.

    Parameters
    ----------
    X:
        Feature array of shape ``(n_bars, n_features)``.
    y:
        Label array of shape ``(n_bars,)``.
    seq_len:
        Length of each sequence window (number of consecutive bars).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 20,
    ) -> None:
        if len(X) <= seq_len:
            raise ValueError(
                f"Dataset length ({len(X)}) must be greater than seq_len ({seq_len})."
            )
        self.seq_len = seq_len
        self.X_seq, self.y_seq = self._make_sequences(X, y, seq_len=seq_len)

    @staticmethod
    def _make_sequences(
        X: np.ndarray, y: np.ndarray, seq_len: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build sliding-window sequence arrays.

        Parameters
        ----------
        X:
            Shape ``(n, n_features)``.
        y:
            Shape ``(n,)``.
        seq_len:
            Window size.

        Returns
        -------
        X_seq:
            Shape ``(n - seq_len, seq_len, n_features)``.
        y_seq:
            Shape ``(n - seq_len,)`` — label at the **last** bar of each window.
        """
        n = len(X)
        if n <= seq_len:
            return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), np.empty(0, dtype=np.float32)

        X_seq = np.stack([X[i : i + seq_len] for i in range(n - seq_len)], axis=0)
        y_seq = y[seq_len:].astype(np.float32)
        return X_seq.astype(np.float32), y_seq

    def __len__(self) -> int:
        return len(self.y_seq)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X_seq[idx], self.y_seq[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LSTMModel:
    """Stacked LSTM binary classifier.

    This class **wraps** ``torch.nn.Module`` logic so it can be instantiated
    without PyTorch (the actual module is only created when ``build()`` is
    called or the class is used inside ``LSTMTrainer``).

    Parameters
    ----------
    input_size:
        Number of features per time step.
    hidden_size:
        Number of LSTM hidden units per layer (default 128).
    num_layers:
        Number of stacked LSTM layers (default 2).
    dropout:
        Dropout probability between LSTM layers (default 0.3).
        Ignored when ``num_layers == 1``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        _require_torch()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self._module: Optional[nn.Module] = None

    def build(self) -> nn.Module:
        """Instantiate and return the underlying ``nn.Module``."""
        _require_torch()
        self._module = _LSTMModule(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        return self._module

    @property
    def module(self) -> "nn.Module":  # type: ignore[name-defined]
        if self._module is None:
            self.build()
        return self._module  # type: ignore[return-value]


if _TORCH_AVAILABLE:

    class _LSTMModule(nn.Module):
        """Internal PyTorch module — use ``LSTMModel`` as the public API."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
            """Forward pass.

            Parameters
            ----------
            x:
                Shape ``(batch, seq_len, input_size)``.

            Returns
            -------
            Logit tensor of shape ``(batch,)``.
            """
            out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
            last = out[:, -1, :]           # take the last time step
            last = self.dropout(last)
            logit = self.fc(last).squeeze(-1)  # (batch,)
            return logit

else:

    class _LSTMModule:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            _require_torch()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class LSTMTrainer:
    """Training loop with early stopping and a ``predict_proba`` adapter.

    The trainer:
    1. Builds sequences from ``(X_train, y_train)`` using ``OptionsSequenceDataset``.
    2. Trains with ``BCEWithLogitsLoss`` weighted by ``pos_weight_factor`` to
       address class imbalance.
    3. Applies early stopping on a chronological 80/20 train/val split.
    4. Exposes ``predict_proba(X)`` that pads the first ``seq_len - 1`` rows
       with ``0.5`` to keep output length equal to input length.

    Parameters
    ----------
    input_size:
        Number of input features.
    hidden_size:
        LSTM hidden units (default 128).
    num_layers:
        Stacked LSTM layers (default 2).
    dropout:
        Dropout between LSTM layers (default 0.3).
    seq_len:
        Sequence window length (default 20).
    lr:
        Adam learning rate (default 1e-3).
    epochs:
        Maximum training epochs (default 50).
    batch_size:
        Mini-batch size (default 64).
    patience:
        Early-stopping patience in epochs (default 10).
    pos_weight_factor:
        ``BCEWithLogitsLoss`` ``pos_weight`` multiplier for the positive class.
        Helps with class imbalance (default 10.0).
    device:
        ``"cpu"``, ``"cuda"``, or ``"mps"`` (default auto-detect).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        seq_len: int = 20,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        pos_weight_factor: float = 10.0,
        device: Optional[str] = None,
    ) -> None:
        _require_torch()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.pos_weight_factor = float(pos_weight_factor)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self._module: Optional[nn.Module] = None
        self.train_losses_: list = []
        self.val_losses_: list = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMTrainer":
        """Train the LSTM model.

        Parameters
        ----------
        X:
            Feature array of shape ``(n_bars, n_features)``.  Values will be
            cast to ``float32``.
        y:
            Label array of shape ``(n_bars,)``.  Must contain binary 0/1 values.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        n = len(X)
        if n <= self.seq_len:
            raise ValueError(
                f"Training data length ({n}) must exceed seq_len ({self.seq_len})."
            )

        # Chronological 80/20 split for early stopping
        split = int(n * 0.8)
        X_tr, y_tr = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        # Build sequences
        ds_tr = OptionsSequenceDataset(X_tr, y_tr, seq_len=self.seq_len)
        if len(X_val) <= self.seq_len:
            # Val set too small — disable early stopping, use all data for training
            ds_tr = OptionsSequenceDataset(X, y, seq_len=self.seq_len)
            X_val_seq = None
            y_val_seq = None
        else:
            ds_val = OptionsSequenceDataset(X_val, y_val, seq_len=self.seq_len)
            X_val_seq = torch.tensor(ds_val.X_seq).to(self.device)
            y_val_seq = torch.tensor(ds_val.y_seq).to(self.device)

        # Build internal PyTorch Dataset wrapper
        train_loader = DataLoader(
            _TensorDataset(ds_tr.X_seq, ds_tr.y_seq),
            batch_size=self.batch_size,
            shuffle=False,  # preserve temporal order
        )

        # Instantiate model
        model_wrapper = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        module = model_wrapper.build().to(self.device)

        pos_weight = torch.tensor([self.pos_weight_factor], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(module.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state: Optional[dict] = None

        for epoch in range(self.epochs):
            # --- Training ---
            module.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = module(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y_batch)

            epoch_loss /= len(ds_tr)
            self.train_losses_.append(epoch_loss)

            # --- Validation / early stopping ---
            if X_val_seq is not None:
                module.eval()
                with torch.no_grad():
                    val_logits = module(X_val_seq)
                    val_loss = criterion(val_logits, y_val_seq).item()
                self.val_losses_.append(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = {k: v.cpu().clone() for k, v in module.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        logger.info(
                            "Early stopping at epoch %d (val_loss=%.4f)", epoch + 1, val_loss
                        )
                        break
            else:
                best_state = {k: v.cpu().clone() for k, v in module.state_dict().items()}

        # Restore best weights
        if best_state is not None:
            module.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self._module = module
        logger.info(
            "LSTM training complete — %d epochs, final train_loss=%.4f",
            epoch + 1,
            epoch_loss,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return per-bar probability for the positive class (shape ``(n,)``).

        The first ``seq_len - 1`` rows cannot form a complete sequence and are
        padded with ``0.5`` (neutral probability).  This preserves index
        alignment with the original DataFrame for downstream trade simulation.

        Parameters
        ----------
        X:
            Feature array of shape ``(n_bars, n_features)``.

        Returns
        -------
        ndarray of shape ``(n_bars,)`` with values in ``[0, 1]``.
        """
        if self._module is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)
        n = len(X)

        if n <= self.seq_len:
            # Every row needs padding
            return np.full(n, 0.5, dtype=np.float32)

        # Build sequences for bars [seq_len-1 .. n-1]
        X_seq, _ = OptionsSequenceDataset._make_sequences(
            X, np.zeros(n, dtype=np.float32), seq_len=self.seq_len
        )

        self._module.eval()
        all_proba: list = []

        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(X_seq), batch_size):
                batch = torch.tensor(X_seq[i : i + batch_size]).to(self.device)
                logits = self._module(batch)
                proba = torch.sigmoid(logits).cpu().numpy()
                all_proba.append(proba)

        seq_proba = np.concatenate(all_proba)  # length: n - seq_len

        # Pad the first seq_len rows with neutral probability
        padding = np.full(self.seq_len, 0.5, dtype=np.float32)
        return np.concatenate([padding, seq_proba])

    def predict_proba_2d(self, X: np.ndarray) -> np.ndarray:
        """Return 2-column probability array compatible with sklearn's interface.

        Column 0 = P(negative class), column 1 = P(positive class).
        This mirrors the ``model.predict_proba(X)[:, 1]`` convention used by
        ``ModelComparator`` and ``TradeSimulator``.

        Parameters
        ----------
        X:
            Feature array of shape ``(n_bars, n_features)``.

        Returns
        -------
        ndarray of shape ``(n_bars, 2)``.
        """
        proba_pos = self.predict_proba(X)
        proba_neg = 1.0 - proba_pos
        return np.column_stack([proba_neg, proba_pos])


# ---------------------------------------------------------------------------
# Internal PyTorch dataset wrapper
# ---------------------------------------------------------------------------


if _TORCH_AVAILABLE:

    class _TensorDataset:
        """Minimal PyTorch Dataset wrapping numpy arrays."""

        def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
            return self.X[idx], self.y[idx]

else:

    class _TensorDataset:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            _require_torch()
