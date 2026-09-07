"""
Data preprocessing utilities for the NYC taxi anomaly detection pipeline.

Handles dataset download, chronological train/validation split, StandardScaler
fitting (train split only to prevent data leakage), and sliding-window sequence
generation for the LSTM autoencoder.
"""

import logging
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DATASET_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"


def download_dataset(target_path: Path) -> None:
    """Download the NAB NYC taxi CSV if not already present."""
    if target_path.exists():
        logger.info("Dataset already present at %s, skipping download.", target_path)
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from %s", DATASET_URL)
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    logger.info("Dataset saved to %s (%d bytes)", target_path, target_path.stat().st_size)


def load_values(csv_path: Path) -> np.ndarray:
    """Load the 'value' column from the CSV as a 1-D float64 array."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df["value"].values.astype(np.float64)


def chronological_split(
    data: np.ndarray, train_fraction: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data chronologically into train and validation segments.

    Never shuffles — temporal order must be preserved so that the validation
    window represents genuinely unseen future observations.
    """
    split_idx = int(len(data) * train_fraction)
    return data[:split_idx], data[split_idx:]


def fit_and_transform(
    train: np.ndarray, val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler exclusively on the training split, then transform both.

    Fitting on the full dataset would let future statistics bleed into training,
    invalidating the train/val separation as a proxy for deployment conditions.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val.reshape(-1, 1)).flatten()
    return train_scaled, val_scaled, scaler


def create_sequences(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Generate overlapping sliding windows from a 1-D time series.

    Args:
        data:        1-D array of length N.
        window_size: Number of timesteps per sequence.

    Returns:
        Array of shape (N - window_size + 1, window_size, 1).
        Sample i covers data[i : i + window_size].
    """
    n_samples = len(data) - window_size + 1
    sequences = np.empty((n_samples, window_size, 1), dtype=data.dtype)
    for i in range(n_samples):
        sequences[i] = data[i : i + window_size].reshape(window_size, 1)
    return sequences


def build_dataset(
    data_dir: Path,
    scaler_save_path: Path,
    window_size: int = 24,
    train_fraction: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline: download → load → split → scale → sequence.

    Returns:
        train_sequences: shape (n_train, window_size, 1)
        val_sequences:   shape (n_val,   window_size, 1)
    """
    csv_path = data_dir / "nyc_taxi.csv"
    download_dataset(csv_path)

    values = load_values(csv_path)
    logger.info("Loaded %d data points from %s", len(values), csv_path)

    train_raw, val_raw = chronological_split(values, train_fraction)
    logger.info("Train split: %d points, Val split: %d points", len(train_raw), len(val_raw))

    train_scaled, val_scaled, scaler = fit_and_transform(train_raw, val_raw)

    scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    logger.info("Scaler saved to %s", scaler_save_path)

    train_seqs = create_sequences(train_scaled, window_size)
    val_seqs = create_sequences(val_scaled, window_size)

    logger.info(
        "Sequence shapes — train: %s, val: %s",
        train_seqs.shape,
        val_seqs.shape,
    )
    return train_seqs, val_seqs
