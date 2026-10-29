
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb

from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# VAE Sampling Layer
# --------------------------------------------------

@register_keras_serializable(package="FinGuard")
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs

        epsilon = tf.random.normal(
            shape=tf.shape(z_mean)
        )

        return (
            z_mean
            + tf.exp(0.5 * z_log_var) * epsilon
        )


# --------------------------------------------------
# Configuration Helpers
# --------------------------------------------------

def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_threshold(path: Path) -> float:
    config = read_json(path)

    if isinstance(config, (int, float)):
        return float(config)

    if isinstance(config, dict):
        for key in (
            "threshold",
            "xgboost_threshold",
            "vae_threshold",
        ):
            if key in config:
                return float(config[key])

    raise ValueError(
        f"No threshold value found in {path.name}"
    )


# --------------------------------------------------
# Load All FinGuard Models
# --------------------------------------------------

def load_models() -> dict:
    print("Loading FinGuard models...")

    # 1. Load XGBoost
    xgb_path = MODEL_DIR / "finguard_xgboost_model.json"

    if not xgb_path.is_file():
        raise FileNotFoundError(
            f"XGBoost model not found: {xgb_path}"
        )

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_path))

    print("XGBoost model loaded.")

    # 2. Load the fitted scaler
    scaler_path = MODEL_DIR / "finguard_scaler.pkl"

    if not scaler_path.is_file():
        raise FileNotFoundError(
            f"Scaler file not found: {scaler_path}"
        )

    scaler = joblib.load(scaler_path)

    print("Scaler loaded.")

    # 3. Load VAE encoder and decoder
    encoder_path = MODEL_DIR / "finguard_vae_encoder.keras"
    decoder_path = MODEL_DIR / "finguard_vae_decoder.keras"

    if not encoder_path.is_file():
        raise FileNotFoundError(
            f"VAE encoder not found: {encoder_path}"
        )

    if not decoder_path.is_file():
        raise FileNotFoundError(
            f"VAE decoder not found: {decoder_path}"
        )

    custom_objects = {
        "Sampling": Sampling,
        "FinGuard>Sampling": Sampling,
    }

    encoder = load_model(
        str(encoder_path),
        custom_objects=custom_objects,
        compile=False,
    )

    decoder = load_model(
        str(decoder_path),
        custom_objects=custom_objects,
        compile=False,
    )

    print("VAE encoder and decoder loaded.")

    # 4. Load separate thresholds
    xgb_threshold_path = MODEL_DIR / "finguard_threshold.json"
    vae_threshold_path = MODEL_DIR / "finguard_vae_threshold.json"

    xgb_threshold = read_threshold(xgb_threshold_path)
    vae_threshold = read_threshold(vae_threshold_path)

    # 5. Load risk configuration
    risk_config = read_json(
        MODEL_DIR / "finguard_risk_config.json"
    )

    risk_config["xgboost_threshold"] = xgb_threshold
    risk_config["vae_threshold"] = vae_threshold

    print(f"XGBoost threshold: {xgb_threshold}")
    print(f"VAE threshold: {vae_threshold}")
    print("All FinGuard models loaded successfully.")

    return {
        "finguard_xgboost_model": xgb_model,
        "finguard_scaler": scaler,
        "finguard_vae_encoder": encoder,
        "finguard_vae_decoder": decoder,
        "finguard_vae_threshold": vae_threshold,
        "risk_config": risk_config,
    }
