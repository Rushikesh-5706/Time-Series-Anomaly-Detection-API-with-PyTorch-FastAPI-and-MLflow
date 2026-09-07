"""
Unit tests for data preprocessing utilities.

Covers create_sequences shape correctness and StandardScaler fit-on-train-only
behaviour.
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.preprocess import create_sequences, fit_and_transform


class TestCreateSequences:
    def test_output_shape_matches_expected(self) -> None:
        window_size = 5
        data = np.arange(20, dtype=np.float64)
        seqs = create_sequences(data, window_size)
        expected_samples = len(data) - window_size + 1
        assert seqs.shape == (expected_samples, window_size, 1), (
            f"Expected shape ({expected_samples}, {window_size}, 1), got {seqs.shape}"
        )

    def test_first_window_content_is_correct(self) -> None:
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        seqs = create_sequences(data, window_size=3)
        np.testing.assert_array_equal(seqs[0].flatten(), [10.0, 20.0, 30.0])

    def test_last_window_content_is_correct(self) -> None:
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        seqs = create_sequences(data, window_size=3)
        np.testing.assert_array_equal(seqs[-1].flatten(), [30.0, 40.0, 50.0])

    def test_single_sample_edge_case(self) -> None:
        data = np.ones(7, dtype=np.float64)
        seqs = create_sequences(data, window_size=7)
        assert seqs.shape == (1, 7, 1)

    def test_feature_dimension_is_one(self) -> None:
        data = np.random.randn(50)
        seqs = create_sequences(data, window_size=10)
        assert seqs.shape[2] == 1


class TestFitAndTransform:
    def test_scaler_fitted_only_on_train(self) -> None:
        rng = np.random.default_rng(42)
        train = rng.normal(loc=5.0, scale=2.0, size=800)
        val = rng.normal(loc=50.0, scale=0.1, size=200)

        train_scaled, val_scaled, scaler = fit_and_transform(train, val)

        # Scaler mean should be close to the training mean, not contaminated by val
        assert abs(scaler.mean_[0] - train.mean()) < 0.5, (
            "Scaler mean deviates too much from training mean — possible leakage."
        )

    def test_train_scaled_is_approximately_standardized(self) -> None:
        rng = np.random.default_rng(0)
        train = rng.normal(loc=10.0, scale=3.0, size=1000)
        val = rng.normal(loc=10.0, scale=3.0, size=200)

        train_scaled, _, _ = fit_and_transform(train, val)

        assert abs(train_scaled.mean()) < 0.05, "Scaled train mean should be ~0"
        assert abs(train_scaled.std() - 1.0) < 0.05, "Scaled train std should be ~1"

    def test_transform_applied_to_val(self) -> None:
        train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        val = np.array([10.0, 20.0])

        _, val_scaled, scaler = fit_and_transform(train, val)

        expected = scaler.transform(val.reshape(-1, 1)).flatten()
        np.testing.assert_allclose(val_scaled, expected)
