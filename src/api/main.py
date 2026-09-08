"""
FastAPI inference service for the LSTM Autoencoder anomaly detector.

Artifacts (model, scaler, threshold) are loaded once at startup via the
lifespan context manager and held in app.state — zero file I/O per request.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List

import joblib
import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.models.anomaly_model import LSTMAutoencoder

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", str(PROJECT_ROOT / "artifacts")))
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    data_point: List[float]


class PredictResponse(BaseModel):
    input_data: List[float]
    anomaly_score: float
    is_anomaly: int
    threshold: float


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    window_size: int
    hidden_dim: int
    num_layers: int
    threshold: float


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


# ---------------------------------------------------------------------------
# Lifespan — load artifacts once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("Loading inference artifacts from %s", ARTIFACT_DIR)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    model = LSTMAutoencoder(
        input_dim=1,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    )
    model.load_state_dict(
        torch.load(ARTIFACT_DIR / "model.pth", map_location="cpu")
    )
    model.eval()

    scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")
    threshold = float(np.load(ARTIFACT_DIR / "anomaly_threshold.npy"))

    app.state.model = model
    app.state.scaler = scaler
    app.state.threshold = threshold
    app.state.config = cfg

    logger.info(
        "Startup complete — window_size=%d, threshold=%.6f",
        cfg["window_size"],
        threshold,
    )
    yield
    logger.info("Shutting down inference service.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Time-Series Anomaly Detection API",
    description="LSTM Autoencoder anomaly detector trained on the NAB NYC taxi dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, status_code=200)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse, status_code=200)
@limiter.limit("60/minute")
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    cfg = request.app.state.config
    window_size: int = cfg["window_size"]

    if len(body.data_point) != window_size:
        logger.warning(
            "Rejected /predict: expected %d values, got %d",
            window_size,
            len(body.data_point),
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"data_point must contain exactly {window_size} floats "
                f"(received {len(body.data_point)})."
            ),
        )

    scaler = request.app.state.scaler
    model: LSTMAutoencoder = request.app.state.model
    threshold: float = request.app.state.threshold

    raw = np.array(body.data_point, dtype=np.float32).reshape(-1, 1)
    scaled = scaler.transform(raw).flatten()

    x = torch.tensor(scaled, dtype=torch.float32).reshape(1, window_size, 1)

    with torch.no_grad():
        recon = model(x)

    anomaly_score = float(((x - recon) ** 2).mean().item())
    is_anomaly = 1 if anomaly_score > threshold else 0

    return PredictResponse(
        input_data=body.data_point,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        threshold=threshold,
    )


@app.get("/model-info", response_model=ModelInfoResponse, status_code=200)
async def model_info(request: Request) -> ModelInfoResponse:
    cfg = request.app.state.config
    return ModelInfoResponse(
        window_size=cfg["window_size"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        threshold=request.app.state.threshold,
    )
