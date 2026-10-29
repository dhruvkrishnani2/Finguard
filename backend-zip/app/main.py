
from contextlib import asynccontextmanager
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model_loader import load_models
from app.schemas import TransactionRequest, PredictionResponse
from app.risk_engine import assess_risk


# --------------------------------------------------
# Application and model lifecycle
# --------------------------------------------------

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models

    try:
        models = load_models()
        app.state.models = models
        print("FinGuard models initialized successfully.")
    except Exception as exc:
        print(f"FinGuard model initialization failed: {exc}")
        raise

    yield

    models.clear()
    app.state.models = {}


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="FinGuard API",
    description=(
        "Hybrid financial fraud detection using "
        "XGBoost and a Variational Autoencoder (VAE)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# CORS configuration
# --------------------------------------------------

# Set FRONTEND_ORIGINS in Render to your deployed frontend URL.
# Multiple origins can be separated by commas.
origins_env = os.getenv("FRONTEND_ORIGINS", "")

allowed_origins = [
    origin.strip().rstrip("/")
    for origin in origins_env.split(",")
    if origin.strip()
]

# Allow local development when using the default configuration.
if not allowed_origins:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_models():
    """Return loaded models or report that startup failed."""
    loaded_models = getattr(app.state, "models", None)

    if not loaded_models:
        raise HTTPException(
            status_code=503,
            detail="FinGuard models are not available.",
        )

    return loaded_models


def get_scaled_features(features, scaler):
    """Validate and scale the 30 transaction features."""
    values = np.asarray(features, dtype=np.float32)

    if values.shape != (30,):
        raise HTTPException(
            status_code=422,
            detail="Exactly 30 transaction features are required.",
        )

    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=422,
            detail="All feature values must be finite numbers.",
        )

    try:
        scaled = scaler.transform(values.reshape(1, -1))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feature scaling failed: {str(exc)}",
        ) from exc

    return np.asarray(scaled, dtype=np.float32)


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "FinGuard API is running",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    loaded_models = getattr(app.state, "models", {})

    return {
        "status": "healthy" if loaded_models else "initializing",
        "models_loaded": bool(loaded_models),
        "xgboost_loaded": (
            "finguard_xgboost_model" in loaded_models
        ),
        "scaler_loaded": (
            "finguard_scaler" in loaded_models
        ),
        "vae_encoder_loaded": (
            "finguard_vae_encoder" in loaded_models
        ),
        "vae_decoder_loaded": (
            "finguard_vae_decoder" in loaded_models
        ),
        "vae_threshold_loaded": (
            "finguard_vae_threshold" in loaded_models
        ),
    }


# --------------------------------------------------
# Fraud prediction endpoint
# --------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(request: TransactionRequest):
    loaded_models = get_models()

    xgb_model = loaded_models["finguard_xgboost_model"]
    scaler = loaded_models["finguard_scaler"]
    encoder = loaded_models["finguard_vae_encoder"]
    decoder = loaded_models["finguard_vae_decoder"]
    vae_threshold = float(
        loaded_models["finguard_vae_threshold"]
    )
    risk_config = loaded_models["risk_config"]

    # 1. Scale features using the saved training scaler.
    scaled_features = get_scaled_features(
        request.features,
        scaler,
    )

    # 2. Get XGBoost fraud probability.
    try:
        probabilities = xgb_model.predict_proba(scaled_features)
        fraud_probability = float(probabilities[0][1])

        xgb_threshold = float(
            risk_config.get("xgboost_threshold", 0.45)
        )

        xgboost_prediction = int(
            fraud_probability >= xgb_threshold
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"XGBoost prediction failed: {str(exc)}",
        ) from exc

    # 3. Encode the transaction with the VAE.
    try:
        encoded = encoder.predict(
            scaled_features,
            verbose=0,
        )

        # The reconstructed encoder returns:
        # [z_mean, z_log_var, z]
        if isinstance(encoded, (tuple, list)):
            if len(encoded) >= 3:
                latent_vector = encoded[2]
            elif len(encoded) == 1:
                latent_vector = encoded[0]
            else:
                raise ValueError(
                    "Unexpected VAE encoder output structure."
                )
        else:
            latent_vector = encoded

        # 4. Reconstruct the scaled transaction.
        reconstructed = decoder.predict(
            latent_vector,
            verbose=0,
        )

        # 5. Calculate mean squared reconstruction error.
        reconstruction_error = float(
            np.mean(
                np.square(
                    scaled_features - reconstructed
                )
            )
        )

        if not np.isfinite(reconstruction_error):
            raise ValueError(
                "Reconstruction error is not finite."
            )

        vae_anomaly_detected = bool(
            reconstruction_error >= vae_threshold
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"VAE prediction failed: {str(exc)}",
        ) from exc

    # 6. Combine model results through the risk engine.
    try:
        risk_result = assess_risk(
            fraud_probability=fraud_probability,
            vae_anomaly_detected=vae_anomaly_detected,
            risk_config=risk_config,
        )

        # Support either a dictionary or a risk-level string.
        if isinstance(risk_result, dict):
            risk_level = risk_result.get(
                "risk_level",
                risk_result.get("risk", "Unknown"),
            )
            risk_reasons = risk_result.get(
                "risk_reasons",
                risk_result.get("reasons", []),
            )
        else:
            risk_level = str(risk_result)
            risk_reasons = []

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Risk assessment failed: {str(exc)}",
        ) from exc

    # 7. Return the response expected by the frontend.
    return PredictionResponse(
        fraud_probability=fraud_probability,
        xgboost_prediction=xgboost_prediction,
        reconstruction_error=reconstruction_error,
        vae_anomaly_detected=vae_anomaly_detected,
        risk_level=risk_level,
        risk_reasons=risk_reasons,
    )


# --------------------------------------------------
# Run locally
# --------------------------------------------------
# Render's start command should be:
# uvicorn app.main:app --host 0.0.0.0 --port $PORT
