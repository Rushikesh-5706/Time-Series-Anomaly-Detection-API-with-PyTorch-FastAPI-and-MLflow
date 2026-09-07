"""
Training pipeline for the LSTM Autoencoder anomaly detector.

Reads all hyperparameters from config.yaml — nothing is hardcoded here.
Trains with MLflow tracking, persists the best checkpoint by validation loss,
computes the anomaly threshold on the validation set, and copies final
artifacts to artifacts/ for stable, run-id-agnostic loading by the API.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocess import build_dataset
from src.models.anomaly_model import LSTMAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sequences_to_tensor(sequences: np.ndarray) -> torch.Tensor:
    return torch.tensor(sequences, dtype=torch.float32)


def compute_reconstruction_errors(
    model: LSTMAutoencoder,
    sequences: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return per-sequence mean squared reconstruction error."""
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size].to(device)
            recon = model(batch)
            mse = ((batch - recon) ** 2).mean(dim=(1, 2))
            errors.extend(mse.cpu().numpy())
    return np.array(errors)


def train() -> None:
    cfg = load_config(CONFIG_PATH)

    window_size: int = cfg["window_size"]
    batch_size: int = cfg["batch_size"]
    epochs: int = cfg["epochs"]
    learning_rate: float = cfg["learning_rate"]
    hidden_dim: int = cfg["hidden_dim"]
    num_layers: int = cfg["num_layers"]
    threshold_percentile: float = cfg["threshold_percentile"]

    logger.info("Config loaded: %s", cfg)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    train_seqs, val_seqs = build_dataset(
        data_dir=DATA_DIR,
        scaler_save_path=SCALER_PATH,
        window_size=window_size,
    )

    train_tensor = sequences_to_tensor(train_seqs)
    val_tensor = sequences_to_tensor(val_seqs)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    input_dim = 1
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- MLflow ---
    mlflow.set_experiment("nyc_taxi_anomaly_detection")

    with mlflow.start_run() as run:
        mlflow.log_params(cfg)
        logger.info("MLflow run id: %s", run.info.run_id)

        best_val_loss = float("inf")
        best_model_path: Path | None = None

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "best_model.pth"

            for epoch in range(1, epochs + 1):
                # --- Training ---
                model.train()
                train_loss_sum = 0.0
                for (batch,) in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    recon = model(batch)
                    loss = criterion(recon, batch)
                    loss.backward()
                    optimizer.step()
                    train_loss_sum += loss.item() * len(batch)
                train_loss = train_loss_sum / len(train_tensor)

                # --- Validation ---
                model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for (batch,) in val_loader:
                        batch = batch.to(device)
                        recon = model(batch)
                        loss = criterion(recon, batch)
                        val_loss_sum += loss.item() * len(batch)
                val_loss = val_loss_sum / len(val_tensor)

                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
                )

                logger.info(
                    "Epoch %03d/%03d — train_loss: %.6f  val_loss: %.6f",
                    epoch,
                    epochs,
                    train_loss,
                    val_loss,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.info("  New best checkpoint saved (val_loss=%.6f)", best_val_loss)

            # Reload best weights before computing threshold
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info("Best model reloaded (val_loss=%.6f)", best_val_loss)

            # --- Anomaly threshold ---
            errors = compute_reconstruction_errors(model, val_tensor, batch_size, device)
            threshold = float(np.percentile(errors, threshold_percentile))
            logger.info(
                "Anomaly threshold (p%.0f of val errors): %.6f",
                threshold_percentile,
                threshold,
            )
            mlflow.log_metric("anomaly_threshold", threshold)

            # --- Persist artifacts ---
            threshold_path = ARTIFACT_DIR / "anomaly_threshold.npy"
            np.save(threshold_path, np.array(threshold))

            model_path = ARTIFACT_DIR / "model.pth"
            shutil.copy(checkpoint_path, model_path)

            # Log to MLflow
            mlflow.log_artifact(str(model_path), artifact_path="model")
            mlflow.log_artifact(str(SCALER_PATH), artifact_path="scaler")
            mlflow.log_artifact(str(threshold_path), artifact_path="threshold")
            mlflow.pytorch.log_model(model, artifact_path="pytorch_model")

            logger.info("All artifacts written to %s and logged to MLflow.", ARTIFACT_DIR)


if __name__ == "__main__":
    train()
