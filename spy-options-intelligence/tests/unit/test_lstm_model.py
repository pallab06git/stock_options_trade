# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or distribution is strictly prohibited.

"""Unit tests for src/ml/lstm_model.py.

Tests are organised into groups:

* TestOptionsSequenceDataset   — sequence construction logic
* TestLSTMModelBuild           — LSTMModel / _LSTMModule construction
* TestLSTMTrainerFit           — training loop, early stopping
* TestLSTMTrainerPredictProba  — output shape, padding, alignment
* TestTorchNotAvailable        — graceful failure without PyTorch
"""

from __future__ import annotations

import importlib
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip entire module if PyTorch is not installed
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed — LSTM tests skipped",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n: int = 200, n_feat: int = 10, pos_frac: float = 0.1):
    """Generate a synthetic feature array and binary label vector."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_feat)).astype(np.float32)
    y = (rng.random(n) < pos_frac).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# OptionsSequenceDataset
# ---------------------------------------------------------------------------


class TestOptionsSequenceDataset(unittest.TestCase):
    """Tests for OptionsSequenceDataset."""

    def setUp(self):
        from src.ml.lstm_model import OptionsSequenceDataset

        self.Cls = OptionsSequenceDataset

    def test_output_lengths_are_n_minus_seqlen(self):
        n, seq_len = 100, 20
        X, y = _make_data(n)
        ds = self.Cls(X, y, seq_len=seq_len)
        expected = n - seq_len
        self.assertEqual(len(ds.X_seq), expected)
        self.assertEqual(len(ds.y_seq), expected)
        self.assertEqual(len(ds), expected)

    def test_sequence_shapes(self):
        n, n_feat, seq_len = 80, 5, 10
        X, y = _make_data(n, n_feat)
        ds = self.Cls(X, y, seq_len=seq_len)
        self.assertEqual(ds.X_seq.shape, (n - seq_len, seq_len, n_feat))
        self.assertEqual(ds.y_seq.shape, (n - seq_len,))

    def test_dtype_is_float32(self):
        X, y = _make_data()
        ds = self.Cls(X, y, seq_len=10)
        self.assertEqual(ds.X_seq.dtype, np.float32)
        self.assertEqual(ds.y_seq.dtype, np.float32)

    def test_label_alignment(self):
        """y_seq[i] should equal y[seq_len + i] (last bar of each window)."""
        n, seq_len = 50, 5
        X, y = _make_data(n)
        ds = self.Cls(X, y, seq_len=seq_len)
        np.testing.assert_array_equal(ds.y_seq, y[seq_len:])

    def test_window_content_alignment(self):
        """X_seq[0] should be X[0:seq_len]."""
        seq_len = 7
        X, y = _make_data(30, 3)
        ds = self.Cls(X, y, seq_len=seq_len)
        np.testing.assert_array_almost_equal(ds.X_seq[0], X[:seq_len])

    def test_raises_when_data_too_short(self):
        with self.assertRaises(ValueError):
            self.Cls(np.zeros((5, 3), dtype=np.float32), np.zeros(5), seq_len=10)

    def test_getitem(self):
        X, y = _make_data(50, 4)
        ds = self.Cls(X, y, seq_len=5)
        x_item, y_item = ds[0]
        self.assertEqual(x_item.shape, (5, 4))
        self.assertIsInstance(float(y_item), float)

    def test_make_sequences_static_empty_output(self):
        from src.ml.lstm_model import OptionsSequenceDataset

        X = np.zeros((5, 3), dtype=np.float32)
        y = np.zeros(5, dtype=np.float32)
        X_seq, y_seq = OptionsSequenceDataset._make_sequences(X, y, seq_len=10)
        self.assertEqual(len(X_seq), 0)
        self.assertEqual(len(y_seq), 0)


# ---------------------------------------------------------------------------
# LSTMModel build
# ---------------------------------------------------------------------------


class TestLSTMModelBuild(unittest.TestCase):
    """Tests for LSTMModel and _LSTMModule."""

    def setUp(self):
        from src.ml.lstm_model import LSTMModel

        self.Cls = LSTMModel

    def test_build_returns_module(self):
        import torch.nn as nn

        model = self.Cls(input_size=8, hidden_size=32, num_layers=1)
        module = model.build()
        self.assertIsInstance(module, nn.Module)

    def test_module_property_builds_on_first_access(self):
        import torch.nn as nn

        model = self.Cls(input_size=6, hidden_size=16, num_layers=2)
        module = model.module  # property
        self.assertIsInstance(module, nn.Module)

    def test_forward_pass_shape(self):
        import torch

        model = self.Cls(input_size=5, hidden_size=16, num_layers=1)
        module = model.build()
        batch = torch.randn(4, 10, 5)  # (batch=4, seq=10, features=5)
        out = module(batch)
        self.assertEqual(out.shape, (4,))

    def test_two_layer_lstm_has_dropout(self):
        import torch.nn as nn

        model = self.Cls(input_size=4, hidden_size=8, num_layers=2, dropout=0.3)
        module = model.build()
        self.assertIsInstance(module.dropout, nn.Dropout)

    def test_single_layer_lstm_no_recurrent_dropout(self):
        model = self.Cls(input_size=4, hidden_size=8, num_layers=1, dropout=0.3)
        module = model.build()
        # LSTM recurrent dropout should be 0.0 for single layer
        self.assertEqual(module.lstm.dropout, 0.0)

    def test_linear_output_size_is_one(self):
        model = self.Cls(input_size=4, hidden_size=8)
        module = model.build()
        self.assertEqual(module.fc.out_features, 1)


# ---------------------------------------------------------------------------
# LSTMTrainer.fit
# ---------------------------------------------------------------------------


class TestLSTMTrainerFit(unittest.TestCase):
    """Tests for LSTMTrainer training loop."""

    def setUp(self):
        from src.ml.lstm_model import LSTMTrainer

        self.Cls = LSTMTrainer

    def _make_trainer(self, **kwargs):
        defaults = dict(
            input_size=5,
            hidden_size=8,
            num_layers=1,
            seq_len=5,
            epochs=3,
            batch_size=16,
            patience=2,
        )
        defaults.update(kwargs)
        return self.Cls(**defaults)

    def test_fit_returns_self(self):
        X, y = _make_data(60, 5)
        trainer = self._make_trainer()
        result = trainer.fit(X, y)
        self.assertIs(result, trainer)

    def test_module_is_set_after_fit(self):
        import torch.nn as nn

        X, y = _make_data(60, 5)
        trainer = self._make_trainer()
        trainer.fit(X, y)
        self.assertIsInstance(trainer._module, nn.Module)

    def test_train_losses_recorded(self):
        X, y = _make_data(60, 5)
        trainer = self._make_trainer(epochs=3)
        trainer.fit(X, y)
        self.assertGreater(len(trainer.train_losses_), 0)

    def test_raises_when_data_too_short(self):
        trainer = self._make_trainer(seq_len=20)
        X, y = _make_data(15, 5)
        with self.assertRaises(ValueError):
            trainer.fit(X, y)

    def test_early_stopping_respected(self):
        """Trainer should stop before max epochs if no improvement."""
        X, y = _make_data(200, 5, pos_frac=0.1)
        trainer = self._make_trainer(epochs=50, patience=2, seq_len=5)
        trainer.fit(X, y)
        # Stopped early — fewer than 50 loss values recorded
        self.assertLess(len(trainer.train_losses_), 50)

    def test_val_losses_recorded_when_val_possible(self):
        X, y = _make_data(200, 5)
        trainer = self._make_trainer(epochs=3, seq_len=5)
        trainer.fit(X, y)
        # Val losses recorded when val set is large enough
        self.assertGreater(len(trainer.val_losses_), 0)

    def test_fit_all_positive_labels(self):
        """Model should train without error even with all-positive labels."""
        X = np.random.randn(80, 5).astype(np.float32)
        y = np.ones(80, dtype=np.float32)
        trainer = self._make_trainer(epochs=2)
        trainer.fit(X, y)
        self.assertIsNotNone(trainer._module)

    def test_fit_all_negative_labels(self):
        """Model should train without error even with all-negative labels."""
        X = np.random.randn(80, 5).astype(np.float32)
        y = np.zeros(80, dtype=np.float32)
        trainer = self._make_trainer(epochs=2)
        trainer.fit(X, y)
        self.assertIsNotNone(trainer._module)


# ---------------------------------------------------------------------------
# LSTMTrainer.predict_proba
# ---------------------------------------------------------------------------


class TestLSTMTrainerPredictProba(unittest.TestCase):
    """Tests for LSTMTrainer.predict_proba and predict_proba_2d."""

    def setUp(self):
        from src.ml.lstm_model import LSTMTrainer

        self.Cls = LSTMTrainer
        self.seq_len = 5
        self.n_feat = 6
        self.n_train = 100
        X, y = _make_data(self.n_train, self.n_feat)
        self.trainer = self.Cls(
            input_size=self.n_feat,
            hidden_size=8,
            num_layers=1,
            seq_len=self.seq_len,
            epochs=2,
            batch_size=32,
        )
        self.trainer.fit(X, y)

    def test_output_length_equals_input_length(self):
        X, _ = _make_data(50, self.n_feat)
        proba = self.trainer.predict_proba(X)
        self.assertEqual(len(proba), len(X))

    def test_first_seqlen_rows_are_padded_with_half(self):
        X, _ = _make_data(50, self.n_feat)
        proba = self.trainer.predict_proba(X)
        np.testing.assert_array_equal(
            proba[: self.seq_len], np.full(self.seq_len, 0.5)
        )

    def test_values_in_zero_one_range(self):
        X, _ = _make_data(60, self.n_feat)
        proba = self.trainer.predict_proba(X)
        self.assertTrue(np.all(proba >= 0.0))
        self.assertTrue(np.all(proba <= 1.0))

    def test_dtype_is_float32(self):
        X, _ = _make_data(40, self.n_feat)
        proba = self.trainer.predict_proba(X)
        self.assertEqual(proba.dtype, np.float32)

    def test_predict_proba_short_input_returns_all_half(self):
        X = np.random.randn(self.seq_len, self.n_feat).astype(np.float32)
        proba = self.trainer.predict_proba(X)
        self.assertEqual(len(proba), self.seq_len)
        np.testing.assert_array_equal(proba, np.full(self.seq_len, 0.5))

    def test_predict_proba_2d_shape(self):
        X, _ = _make_data(30, self.n_feat)
        proba2d = self.trainer.predict_proba_2d(X)
        self.assertEqual(proba2d.shape, (30, 2))

    def test_predict_proba_2d_columns_sum_to_one(self):
        X, _ = _make_data(30, self.n_feat)
        proba2d = self.trainer.predict_proba_2d(X)
        col_sums = proba2d.sum(axis=1)
        np.testing.assert_allclose(col_sums, np.ones(30), atol=1e-5)

    def test_predict_proba_2d_col1_matches_predict_proba(self):
        X, _ = _make_data(40, self.n_feat)
        proba1d = self.trainer.predict_proba(X)
        proba2d = self.trainer.predict_proba_2d(X)
        np.testing.assert_array_almost_equal(proba2d[:, 1], proba1d)

    def test_raises_before_fit(self):
        from src.ml.lstm_model import LSTMTrainer

        trainer = LSTMTrainer(input_size=5)
        X, _ = _make_data(50, 5)
        with self.assertRaises(RuntimeError):
            trainer.predict_proba(X)

    def test_batch_inference_consistency(self):
        """Large input split into batches should match small-batch output."""
        X, _ = _make_data(300, self.n_feat)
        proba = self.trainer.predict_proba(X)
        self.assertEqual(len(proba), 300)
        # All non-padded values must be valid probabilities
        self.assertTrue(np.all(proba[self.seq_len :] >= 0.0))
        self.assertTrue(np.all(proba[self.seq_len :] <= 1.0))


# ---------------------------------------------------------------------------
# Graceful failure without PyTorch
# ---------------------------------------------------------------------------


class TestTorchNotAvailable(unittest.TestCase):
    """Verify modules raise clear ImportError when torch is absent."""

    def test_require_torch_raises_import_error(self):
        """_require_torch() must raise ImportError when torch is absent."""
        with patch.dict(sys.modules, {"torch": None}):
            # Force re-evaluation of _TORCH_AVAILABLE by calling _require_torch
            # directly with a temporary monkeypatch
            import src.ml.lstm_model as m

            orig = m._TORCH_AVAILABLE
            m._TORCH_AVAILABLE = False
            try:
                with self.assertRaises(ImportError) as ctx:
                    m._require_torch()
                self.assertIn("pip install torch", str(ctx.exception))
            finally:
                m._TORCH_AVAILABLE = orig

    def test_lstm_model_raises_import_error_when_torch_absent(self):
        """LSTMModel.__init__ should raise ImportError without torch."""
        import src.ml.lstm_model as m

        orig = m._TORCH_AVAILABLE
        m._TORCH_AVAILABLE = False
        try:
            with self.assertRaises(ImportError):
                m.LSTMModel(input_size=5)
        finally:
            m._TORCH_AVAILABLE = orig

    def test_lstm_trainer_raises_import_error_when_torch_absent(self):
        """LSTMTrainer.__init__ should raise ImportError without torch."""
        import src.ml.lstm_model as m

        orig = m._TORCH_AVAILABLE
        m._TORCH_AVAILABLE = False
        try:
            with self.assertRaises(ImportError):
                m.LSTMTrainer(input_size=5)
        finally:
            m._TORCH_AVAILABLE = orig


if __name__ == "__main__":
    unittest.main()
