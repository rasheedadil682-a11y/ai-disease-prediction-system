from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "backend" / "artifacts"

ROUTING_KEYS = {"model", "type", "target", "endpoint"}

HEART_ALIASES = {
    "age": "Age",
    "sex": "Sex",
    "chest_pain_type": "ChestPainType",
    "chestpaintype": "ChestPainType",
    "resting_bp": "RestingBP",
    "restingbp": "RestingBP",
    "cholesterol": "Cholesterol",
    "fasting_bs": "FastingBS",
    "fastingbs": "FastingBS",
    "resting_ecg": "RestingECG",
    "restingecg": "RestingECG",
    "maxhr": "MaxHR",
    "max_hr": "MaxHR",
    "exercise_angina": "ExerciseAngina",
    "exerciseangina": "ExerciseAngina",
    "oldpeak": "Oldpeak",
    "st_slope": "ST_Slope",
    "stslope": "ST_Slope",
}

DIABETES_ALIASES = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "bp": "BloodPressure",
    "blood_pressure": "BloodPressure",
    "bloodpressure": "BloodPressure",
    "skin_thickness": "SkinThickness",
    "skinthickness": "SkinThickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "dpf": "DiabetesPedigreeFunction",
    "diabetes_pedigree_function": "DiabetesPedigreeFunction",
    "diabetespedigreefunction": "DiabetesPedigreeFunction",
    "age": "Age",
}

MODEL_STACKS: dict[str, dict] = {}

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def _load_artifact(name: str):
    path = ARTIFACTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return joblib.load(path)


def _load_stacks() -> None:
    MODEL_STACKS["heart"] = {
        "model": _load_artifact("heart_model.pkl"),
        "scaler": _load_artifact("heart_scaler.pkl"),
        "features": _load_artifact("heart_features.pkl"),
        "categorical_map": _load_artifact("heart_category_maps.pkl"),
        "impute_map": _load_artifact("heart_impute_map.pkl"),
        "aliases": HEART_ALIASES,
    }
    MODEL_STACKS["diabetes"] = {
        "model": _load_artifact("diabetes_model.pkl"),
        "scaler": _load_artifact("diabetes_scaler.pkl"),
        "features": _load_artifact("diabetes_features.pkl"),
        "categorical_map": {},
        "impute_map": _load_artifact("diabetes_impute_map.pkl"),
        "aliases": DIABETES_ALIASES,
    }
    logger.info("Loaded model stacks from %s", ARTIFACTS_DIR)


def _error(msg: str, code: int = 400):
    return jsonify({"ok": False, "error": msg}), code


def _canonicalize(payload: dict, aliases: dict[str, str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in payload.items():
        if key in ROUTING_KEYS:
            continue
        if key in aliases.values():
            out[key] = value
            continue
        mapped = aliases.get(key.strip().lower().replace(" ", "_"))
        if mapped:
            out[mapped] = value
    return out


def _normalize_string(value: object) -> str:
    return str(value).strip()


def _normalize_category(column: str, value: object) -> str:
    raw = _normalize_string(value).upper()
    if column == "Sex":
        return "M" if raw in {"1", "M", "MALE"} else "F" if raw in {"0", "F", "FEMALE"} else raw
    if column == "ExerciseAngina":
        return "Y" if raw in {"1", "Y", "YES", "TRUE"} else "N" if raw in {"0", "N", "NO", "FALSE"} else raw
    if column == "ST_Slope":
        title = raw.title()
        return title
    if column == "RestingECG":
        return raw.capitalize() if raw in {"NORMAL"} else raw
    return _normalize_string(value)


def _to_feature_vector(model_name: str, payload: dict) -> np.ndarray:
    stack = MODEL_STACKS[model_name]
    features: list[str] = stack["features"]
    categorical_map: dict[str, dict[str, int]] = stack["categorical_map"]
    impute_map: dict[str, float] = stack["impute_map"]
    aliases: dict[str, str] = stack["aliases"]

    canonical = _canonicalize(payload, aliases)
    missing = [f for f in features if f not in canonical or canonical[f] in (None, "")]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    values: list[float] = []
    for feature in features:
        raw = canonical[feature]
        if feature in categorical_map:
            normalized = _normalize_category(feature, raw)
            mapping = categorical_map[feature]
            if normalized not in mapping:
                allowed = ", ".join(sorted(mapping.keys()))
                raise ValueError(f"Invalid value for {feature}: {raw!r}. Allowed values: {allowed}")
            values.append(float(mapping[normalized]))
            continue

        try:
            num = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field '{feature}' must be numeric.") from exc

        if feature in impute_map and num == 0.0:
            num = float(impute_map[feature])
        values.append(num)

    vector = np.array([values], dtype=float)
    logger.info("processed features (%s): %s", model_name, values)
    return vector


def _predict(model_name: str, payload: dict):
    print(f"[REQUEST] model={model_name} payload={payload}")
    x = _to_feature_vector(model_name, payload)
    stack = MODEL_STACKS[model_name]
    scaled = stack["scaler"].transform(x)
    model = stack["model"]
    pred = int(model.predict(scaled)[0])
    result = "High Risk" if pred == 1 else "Low Risk"
    probability = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(scaled)[0]
            if len(proba) > 1:
                probability = float(proba[1])
            else:
                probability = float(proba[0])
        except Exception as exc:
            logger.warning("predict_proba failed for %s: %s", model_name, exc)
    print(
        f"[PREDICT] model={model_name} scaled={scaled.tolist()} pred={pred} "
        f"label={result} probability={probability}"
    )
    response = {"ok": True, "model": model_name, "result": result}
    if probability is not None:
        response["probability"] = max(0.0, min(1.0, probability))
    return jsonify(response)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "message": "Backend is healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error("Request body must be a JSON object.")

        selector = payload.get("model") or payload.get("type") or payload.get("target") or payload.get("endpoint")
        model_name = str(selector).strip().lower() if selector is not None else ""
        if model_name in {"heart", "heart_disease", "cardiac"}:
            return _predict("heart", payload)
        if model_name in {"diabetes", "diabetic"}:
            return _predict("diabetes", payload)
        return _error("Invalid model. Use 'heart' or 'diabetes'.")
    except ValueError as exc:
        logger.warning("validation error: %s", exc)
        return _error(str(exc))
    except Exception as exc:
        logger.exception("unexpected prediction error")
        return _error(f"Internal server error: {exc}", 500)


@app.errorhandler(404)
def _not_found(_e):
    return _error("Not found.", 404)


@app.errorhandler(405)
def _method_not_allowed(_e):
    return _error("Method not allowed.", 405)


if __name__ == "__main__":
    _load_stacks()
    import os

port = int(os.environ.get("PORT", 5001))
app.run(host="0.0.0.0", port=port)
