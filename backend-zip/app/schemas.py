from pydantic import BaseModel, Field, field_validator
import math


class TransactionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="30 features in the order Time, V1-V28, Amount",
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, values: list[float]) -> list[float]:
        if len(values) != 30:
            raise ValueError("Exactly 30 features are required.")

        if not all(math.isfinite(value) for value in values):
            raise ValueError("All features must be finite numbers.")

        return values


class PredictionResponse(BaseModel):
    fraud_probability: float
    xgboost_prediction: int
    reconstruction_error: float
    vae_anomaly_score: float
    vae_anomaly_detected: bool
    risk_score: float
    risk_level: str
    risk_reasons: list[str]


class ExplanationRequest(BaseModel):
    fraud_probability: float
    xgboost_prediction: int
    reconstruction_error: float
    vae_anomaly_score: float
    vae_anomaly_detected: bool
    risk_score: float
    risk_level: str
    risk_reasons: list[str]