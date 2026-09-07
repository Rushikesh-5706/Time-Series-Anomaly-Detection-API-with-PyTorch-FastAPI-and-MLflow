"""
Unit tests for the LSTMAutoencoder model.

Verifies that the forward pass preserves input shape across different
batch sizes, sequence lengths, and layer configurations.
"""

import torch
import pytest

from src.models.anomaly_model import LSTMAutoencoder


class TestLSTMAutoencoder:
    def test_output_shape_matches_input(self) -> None:
        model = LSTMAutoencoder(input_dim=1, hidden_dim=16, num_layers=1)
        x = torch.randn(4, 24, 1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape, (
            f"Expected output shape {x.shape}, got {out.shape}"
        )

    def test_output_shape_with_two_layers(self) -> None:
        model = LSTMAutoencoder(input_dim=1, hidden_dim=32, num_layers=2)
        x = torch.randn(8, 12, 1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape

    def test_single_sample_batch(self) -> None:
        model = LSTMAutoencoder(input_dim=1, hidden_dim=8, num_layers=1)
        x = torch.randn(1, 24, 1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 24, 1)

    def test_model_in_train_mode_returns_gradients(self) -> None:
        model = LSTMAutoencoder(input_dim=1, hidden_dim=16, num_layers=1)
        x = torch.randn(2, 10, 1)
        out = model(x)
        loss = out.mean()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_eval_mode_no_grad_does_not_raise(self) -> None:
        model = LSTMAutoencoder(input_dim=1, hidden_dim=16, num_layers=1)
        model.eval()
        x = torch.randn(3, 24, 1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape
