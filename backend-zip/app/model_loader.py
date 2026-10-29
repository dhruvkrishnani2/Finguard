import json
import os
from pathlib import Path

import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
from tensorflow.keras import layers


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


@layers.register_keras_serializable(package="FinGuard")
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs

        import tensorflow as tf

        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_threshold(path: Path, default_key: str = "threshold") -> float:
    config = read_json(path)

    if isinstance(config, (int, float)):
        return float(config)

    for key in (default_key, "threshold"):
        if key in config:
            return float(config[key])

    raise ValueError(f"No threshold value found in {path.name}")


def load_models() -> dict:
    # XGBoost model
    xgb_path = MODEL_DIR / "finguard_xgboost_model.json"
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_path))

    # StandardScaler
    scaler_path = MODEL_DIR / "finguard_scaler.pkl"
    scaler = joblib.load(scaler_path)

    # VAE encoder and decoder
    encoder_path = MODEL_DIR / "finguard_vae_encoder.keras"
    decoder_path = MODEL_DIR / "finguard_vae_decoder.keras"

    if not encoder_path.exists():
        raise FileNotFoundError(f"VAE encoder not found: {encoder_path}")

    if not decoder_path.exists():
        raise FileNotFoundError(f"VAE decoder not found: {decoder_path}")

    custom_objects = {"Sampling": Sampling}

    encoder = load_model(
        encoder_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=True,
    )
    decoder = load_model(
        decoder_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=True,
    )

    # Keep XGBoost and VAE thresholds separate.
    xgb_threshold = read_threshold(
        MODEL_DIR / "finguard_threshold.json"
    )
    vae_threshold = read_threshold(
        MODEL_DIR / "finguard_vae_threshold.json"
    )

    risk_config = read_json(MODEL_DIR / "finguard_risk_config.json")

    risk_config["xgboost_threshold"] = xgb_threshold
    risk_config["vae_threshold"] = vae_threshold

    print("FinGuard models loaded successfully.")

    return {
        "finguard_xgboost_model": xgb_model,
        "finguard_scaler": scaler,
        "finguard_vae_encoder": encoder,
        "finguard_vae_decoder": decoder,
        "finguard_vae_threshold": vae_threshold,
        "risk_config": risk_config,
    }