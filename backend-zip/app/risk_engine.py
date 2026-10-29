def assess_risk(
    fraud_probability: float,
    vae_anomaly_detected: bool,
    risk_config: dict,
) -> dict:
    reasons = []

    critical_threshold = float(
        risk_config.get("critical_threshold", 0.90)
    )
    high_threshold = float(
        risk_config.get("high_threshold", 0.50)
    )
    medium_threshold = float(
        risk_config.get("medium_threshold", 0.20)
    )

    if fraud_probability >= critical_threshold:
        risk_level = "Critical"
        reasons.append("XGBoost detected a very high fraud probability.")

    elif fraud_probability >= high_threshold and vae_anomaly_detected:
        risk_level = "Critical"
        reasons.append("Both models indicate suspicious activity.")

    elif fraud_probability >= high_threshold or vae_anomaly_detected:
        risk_level = "High"
        if fraud_probability >= high_threshold:
            reasons.append("XGBoost detected elevated fraud probability.")
        if vae_anomaly_detected:
            reasons.append("VAE detected an unusual transaction pattern.")

    elif fraud_probability >= medium_threshold:
        risk_level = "Medium"
        reasons.append("XGBoost detected moderate fraud probability.")

    else:
        risk_level = "Low"
        reasons.append("No strong fraud indicators were detected.")

    # Weighted risk score, expressed from 0 to 100.
    vae_score = 100.0 if vae_anomaly_detected else 0.0
    risk_score = (
        0.8 * fraud_probability * 100.0
        + 0.2 * vae_score
    )
    risk_score = max(0.0, min(100.0, risk_score))

    return {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 2),
        "risk_reasons": reasons,
    }