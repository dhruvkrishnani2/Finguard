
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb

from tensorflow.keras import Model, layers


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# VAE Sampling Layer
# --------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="FinGuard")
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# --------------------------------------------------
# Rebuild Encoder
# Architecture: 30 -> 64 -> 32 -> latent_dim 8
# --------------------------------------------------

def build_encoder(input_dim=30, latent_dim=8):
    inputs = layers.Input(shape=(input_dim,), name="encoder_input")

    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="sampling")([z_mean, z_log_var])

    encoder = Model(
        inputs,
        [z_mean, z_log_var, z],
        name="encoder",
    )

    return encoder


# --------------------------------------------------
# Rebuild Decoder
# Architecture: latent_dim 8 -> 32 -> 64 -> 30
# --------------------------------------------------

def build_decoder(latent_dim=8, output_dim=30):
    latent_inputs = layers.Input(
        shape=(latent_dim,),
        name="decoder_input",
    )

    x = layers.Dense(32, activation="relu")(latent_inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)

    decoder = Model(
        latent_inputs,
        outputs,
        name="decoder",
    )

    return decoder


# --------------------------------------------------
# Configuration Helpers
# --------------------------------------------------

def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

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

    raise ValueError(f"No threshold found in {path.name}")


# --------------------------------------------------
# Load All Models
# --------------------------------------------------

def load_models():
    print("Loading FinGuard models...")

    # 1. XGBoost model
    xgb_path = MODEL_DIR / "finguard_xgboost_model.json"

    if not xgb_path.is_file():
        raise FileNotFoundError(f"XGBoost model missing: {xgb_path}")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_path))

    # 2. StandardScaler
    scaler_path = MODEL_DIR / "finguard_scaler.pkl"

    if not scaler_path.is_file():
        raise FileNotFoundError(f"Scaler missing: {scaler_path}")

    scaler = joblib.load(scaler_path)

    # 3. VAE weights
    encoder_weights = MODEL_DIR / "finguard_vae_encoder.weights.h5"
    decoder_weights = MODEL_DIR / "finguard_vae_decoder.weights.h5"

    if not encoder_weights.is_file():
        raise FileNotFoundError(f"Encoder weights missing: {encoder_weights}")

    if not decoder_weights.is_file():
        raise FileNotFoundError(f"Decoder weights missing: {decoder_weights}")

    encoder = build_encoder(input_dim=30, latent_dim=8)
    decoder = build_decoder(latent_dim=8, output_dim=30)

    # Build layers before loading saved weights.
    encoder(
        tf.zeros((1, 30), dtype=tf.float32),
        training=False,
    )
    decoder(
        tf.zeros((1, 8), dtype=tf.float32),
        training=False,
    )

    encoder.load_weights(str(encoder_weights))
    decoder.load_weights(str(decoder_weights))

    print("XGBoost, scaler, and VAE weights loaded.")

    # 4. Separate model thresholds
    xgb_threshold = read_threshold(
        MODEL_DIR / "finguard_threshold.json"
    )
    vae_threshold = read_threshold(
        MODEL_DIR / "finguard_vae_threshold.json"
    )

    # 5. Risk configuration
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
