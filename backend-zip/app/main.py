from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model_loader import load_models
from app.schemas import TransactionRequest, PredictionResponse, ExplanationRequest
from app.risk_engine import assess_risk


models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    models = load_models()
    app.state.models = models
    yield
    models.clear()


app = FastAPI(
    title="FinGuard API",
    description="Hybrid financial fraud detection using XGBoost and VAE",
    version="1.0.0",
    lifespan=lifespan,
)

# Set FRONTEND_ORIGINS as a comma-separated list in production.
# Example: https://your-app.amplifyapp.com
import os

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://localhost:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/")
def root():
    return {
        "message": "FinGuard API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    loaded = bool(models)

    return {
        "status": "healthy" if loaded else "not_ready",
        "models_loaded": loaded,
        "database_required": False,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(request: TransactionRequest):
    if not models:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded yet.",
        )

    try:
        # Original request must contain 30 features in the trained order.
        raw_features = np.asarray(
            request.features,
            dtype=np.float32,
        ).reshape(1, -1)

        scaler = models["finguard_scaler"]
        xgb_model = models["finguard_xgboost_model"]
        encoder = models["finguard_vae_encoder"]
        decoder = models["finguard_vae_decoder"]
        risk_config = models["risk_config"]

        # Use the same preprocessing fitted during training.
        scaled_features = scaler.transform(raw_features)

        # XGBoost fraud probability and threshold-based prediction.
        probabilities = xgb_model.predict_proba(scaled_features)
        fraud_probability = float(probabilities[0][1])

        xgb_threshold = float(
            risk_config.get("xgboost_threshold", 0.45)
        )
        xgb_prediction = int(fraud_probability >= xgb_threshold)

        # Encode the input and reconstruct it using the VAE.
        encoded = encoder.predict(scaled_features, verbose=0)

        if isinstance(encoded, (tuple, list)):
            # Expected encoder outputs: z_mean, z_log_var, sampled_z.
            latent_vector = encoded[0]
        else:
            latent_vector = encoded

        reconstructed = decoder.predict(latent_vector, verbose=0)

        reconstruction_error = float(
            np.mean(
                np.square(
                    scaled_features.astype(np.float64)
                    - reconstructed.astype(np.float64)
                ),
                axis=1,
            )[0]
        )

        vae_threshold = float(
            models["finguard_vae_threshold"]
        )
        vae_anomaly_detected = bool(
            reconstruction_error > vae_threshold
        )

        risk = assess_risk(
            fraud_probability=fraud_probability,
            vae_anomaly_detected=vae_anomaly_detected,
            risk_config=risk_config,
        )

        return PredictionResponse(
            fraud_probability=round(fraud_probability, 6),
            xgboost_prediction=xgb_prediction,
            reconstruction_error=round(reconstruction_error, 6),
            vae_anomaly_score=round(reconstruction_error, 6),
            vae_anomaly_detected=vae_anomaly_detected,
            risk_score=risk["risk_score"],
            risk_level=risk["risk_level"],
            risk_reasons=risk["risk_reasons"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        # Detailed errors should be logged privately in production.
        print(f"Prediction failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend logs.",
        ) from exc


@app.post("/explain")
def explain_prediction(request: ExplanationRequest):
    """Return a simple explanation without storing transaction data."""
    explanations = {
        "Critical": "This transaction has strong fraud indicators. "
                    "Review it carefully before approving it.",
        "High": "This transaction has significant suspicious signals. "
                "Additional verification is recommended.",
        "Medium": "This transaction has some risk indicators. "
                  "Review it if other warning signs are present.",
        "Low": "The models found relatively few fraud indicators. "
               "A low risk score does not guarantee that a transaction is safe.",
    }

    return {
        "risk_level": request.risk_level,
        "explanation": explanations.get(
            request.risk_level,
            "Review the model results before making a decision.",
        ),
        "fraud_probability": request.fraud_probability,
        "vae_anomaly_detected": request.vae_anomaly_detected,
        "risk_reasons": request.risk_reasons,
        "disclaimer": "This is a model-generated risk assessment, "
                      "not a guarantee of fraud or legitimacy.",
    }