"""
Flask API for disease prediction using pre-trained joblib models.

Matches the training notebook: label-encoded heart categoricals, StandardScaler
on all features, then estimator.predict on scaled rows.
"""

from __future__ import annotations

import csv
import logging
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "disease-prediction-master" / "models"
DATA_DIR = ROOT / "disease-prediction-master" / "data"

HEART_CSV = DATA_DIR / "heart.csv"
DIABETES_CSV = DATA_DIR / "diabetes.csv"

HEART_CAT_COLS = ("Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope")

# snake_case / common aliases -> notebook / CSV column names
HEART_ALIASES: dict[str, str] = {
    "age": "Age",
    "sex": "Sex",
    "chestpaintype": "ChestPainType",
    "chest_pain_type": "ChestPainType",
    "restingbp": "RestingBP",
    "resting_bp": "RestingBP",
    "resting_blood_pressure": "RestingBP",
    "bp": "RestingBP",
    "cholesterol": "Cholesterol",
    "fastingbs": "FastingBS",
    "fasting_bs": "FastingBS",
    "restingecg": "RestingECG",
    "resting_ecg": "RestingECG",
    "maxhr": "MaxHR",
    "max_hr": "MaxHR",
    "max_heart_rate": "MaxHR",
    "exerciseangina": "ExerciseAngina",
    "exercise_angina": "ExerciseAngina",
    "oldpeak": "Oldpeak",
    "stslope": "ST_Slope",
    "st_slope": "ST_Slope",
}

DIABETES_ALIASES: dict[str, str] = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "bp": "BloodPressure",
    "blood_pressure": "BloodPressure",
    "bloodpressure": "BloodPressure",
    "skinthickness": "SkinThickness",
    "skin_thickness": "SkinThickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "diabetespedigreefunction": "DiabetesPedigreeFunction",
    "diabetes_pedigree_function": "DiabetesPedigreeFunction",
    "dpf": "DiabetesPedigreeFunction",
    "age": "Age",
}

DIABETES_ZERO_AS_NA = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")

ROUTING_KEYS = frozenset({"model", "type", "target", "endpoint"})

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
    expose_headers=["Content-Type"],
)

heart_model = None
diabetes_model = None
heart_scaler = None
diabetes_scaler = None
heart_feature_names: list[str] | None = None
diabetes_feature_names: list[str] | None = None
heart_cat_encoders: dict[str, LabelEncoder] | None = None
diabetes_medians: np.ndarray | None = None


def _prediction_label(raw) -> str:
    if hasattr(raw, "item"):
        raw = raw.item()
    try:
        v = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Unexpected prediction type: {type(raw)!r}") from e
    if v == 1:
        return "High Risk"
    if v == 0:
        return "Low Risk"
    raise ValueError(f"Unexpected class label: {v!r} (expected 0 or 1)")


def _positive_class_probability(model, x_sc) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(x_sc)[0]
    except Exception:
        logger.exception("Probability estimation failed")
        return None

    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        index = classes.index(1)
    elif len(proba) == 2:
        index = 1
    else:
        index = int(np.argmax(proba))
    return float(proba[index])


def _strip_routing_keys(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in ROUTING_KEYS}


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return [], []
        fields = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return fields, rows


def _fit_heart_encoders() -> dict[str, LabelEncoder]:
    if not HEART_CSV.is_file():
        raise FileNotFoundError(f"Heart dataset not found: {HEART_CSV}")
    _, rows = _read_csv_rows(HEART_CSV)
    encoders: dict[str, LabelEncoder] = {}
    for col in HEART_CAT_COLS:
        values = [str(r[col]) for r in rows if col in r]
        le = LabelEncoder()
        le.fit(values)
        encoders[col] = le
    return encoders


def _diabetes_feature_medians() -> tuple[list[str], np.ndarray]:
    if not DIABETES_CSV.is_file():
        raise FileNotFoundError(f"Diabetes dataset not found: {DIABETES_CSV}")
    fields, rows = _read_csv_rows(DIABETES_CSV)
    feat_names = [c for c in fields if c != "Outcome"]
    matrix = np.zeros((len(rows), len(feat_names)), dtype=float)
    for i, row in enumerate(rows):
        for j, name in enumerate(feat_names):
            matrix[i, j] = float(row[name])
    for j, name in enumerate(feat_names):
        if name in DIABETES_ZERO_AS_NA:
            col = matrix[:, j]
            col[col == 0] = np.nan
            matrix[:, j] = col
    medians = np.nanmedian(matrix, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    return feat_names, medians


def _canonicalize_payload(raw: dict, aliases: dict[str, str]) -> dict:
    """Merge JSON keys onto canonical CSV column names."""
    out: dict = {}
    allowed = set(aliases.values())
    for key, val in raw.items():
        if key in ROUTING_KEYS:
            continue
        if key in allowed:
            out[key] = val
            continue
        lk = key.strip().lower().replace(" ", "_")
        if lk in aliases:
            out[aliases[lk]] = val
            continue
    return out


def _normalize_sex(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    s = str(value).strip().upper()
    if s in ("M", "MALE", "1", "TRUE"):
        return "M"
    if s in ("F", "FEMALE", "0", "FALSE"):
        return "F"
    return str(value).strip()


def _normalize_exercise_angina(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    s = str(value).strip().upper()
    if s in ("Y", "YES", "1", "TRUE"):
        return "Y"
    if s in ("N", "NO", "0", "FALSE"):
        return "N"
    return str(value).strip()


def _encode_heart_categorical(col: str, value, enc: LabelEncoder) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(int(value))
    if col == "Sex":
        value = _normalize_sex(value)
    elif col == "ExerciseAngina":
        value = _normalize_exercise_angina(value)
    else:
        value = str(value).strip()
    try:
        return float(enc.transform([str(value)])[0])
    except ValueError as e:
        opts = ", ".join(map(str, enc.classes_))
        raise ValueError(f"Invalid value for {col}: {value!r}. Use one of: {opts}") from e


def _load_models() -> None:
    global heart_model, diabetes_model, heart_scaler, diabetes_scaler
    global heart_feature_names, diabetes_feature_names, heart_cat_encoders, diabetes_medians

    heart_model_path = MODELS_DIR / "heart_disease_best_model.pkl"
    diabetes_model_path = MODELS_DIR / "diabetes_best_model.pkl"
    heart_scaler_path = MODELS_DIR / "heart_scaler.pkl"
    diabetes_scaler_path = MODELS_DIR / "diabetes_scaler.pkl"
    heart_features_path = MODELS_DIR / "heart_features.pkl"
    diabetes_features_path = MODELS_DIR / "diabetes_features.pkl"

    for p, label in [
        (heart_model_path, "Heart model"),
        (diabetes_model_path, "Diabetes model"),
        (heart_scaler_path, "Heart scaler"),
        (diabetes_scaler_path, "Diabetes scaler"),
        (heart_features_path, "Heart feature list"),
        (diabetes_features_path, "Diabetes feature list"),
    ]:
        if not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")

    heart_model = joblib.load(heart_model_path)
    diabetes_model = joblib.load(diabetes_model_path)
    heart_scaler = joblib.load(heart_scaler_path)
    diabetes_scaler = joblib.load(diabetes_scaler_path)
    heart_feature_names = joblib.load(heart_features_path)
    diabetes_feature_names = joblib.load(diabetes_features_path)
    heart_cat_encoders = _fit_heart_encoders()
    _, diabetes_medians = _diabetes_feature_medians()

    logger.info("Loaded models, scalers, and preprocessing from %s", MODELS_DIR)


def _missing_fields(canonical: dict, required: list[str]) -> list[str]:
    missing: list[str] = []
    for f in required:
        if f not in canonical:
            missing.append(f)
            continue
        v = canonical[f]
        if v is None:
            missing.append(f)
            continue
        if isinstance(v, str) and v.strip() == "":
            missing.append(f)
    return missing


def _predict_heart_body(payload: dict | None):
    if (
        heart_model is None
        or heart_scaler is None
        or not heart_feature_names
        or not heart_cat_encoders
    ):
        return jsonify({"error": "Heart model stack is not loaded.", "ok": False}), 503

    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object.", "ok": False}), 400

    logger.info(
        "Heart request payload keys: %s",
        sorted(k for k in payload if k not in ROUTING_KEYS),
    )

    canon = _canonicalize_payload(payload, HEART_ALIASES)
    missing = _missing_fields(canon, heart_feature_names)
    if missing:
        return (
            jsonify(
                {
                    "error": f"Missing required field(s): {', '.join(missing)}",
                    "ok": False,
                    "expected": heart_feature_names,
                }
            ),
            400,
        )

    row: list[float] = []
    try:
        for name in heart_feature_names:
            raw_val = canon[name]
            if name in HEART_CAT_COLS:
                row.append(_encode_heart_categorical(name, raw_val, heart_cat_encoders[name]))
            else:
                if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
                    raise ValueError(f"Field '{name}' must be a number")
                row.append(float(raw_val))
    except ValueError as exc:
        return jsonify({"error": str(exc), "ok": False}), 400

    logger.info(
        "Heart processed feature vector (%s): %s",
        heart_feature_names,
        [round(float(v), 5) for v in row],
    )

    x = np.array([row], dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*valid feature names.*",
                category=UserWarning,
            )
            x_sc = heart_scaler.transform(x)
        logger.info("Heart scaled vector (first 6): %s", [round(float(v), 5) for v in x_sc[0, :6]])
        pred = heart_model.predict(x_sc)[0]
        probability = _positive_class_probability(heart_model, x_sc)
    except Exception as exc:
        logger.exception("Heart prediction failed")
        return jsonify({"error": "Prediction failed.", "detail": str(exc), "ok": False}), 500

    try:
        result = _prediction_label(pred)
    except ValueError as exc:
        return jsonify({"error": str(exc), "ok": False}), 500

    logger.info("Heart prediction OK: class=%s label=%s", pred, result)
    response = {"ok": True, "model": "heart", "result": result}
    if probability is not None:
        response["probability"] = round(probability, 4)
    return jsonify(response)


def _predict_diabetes_body(payload: dict | None):
    if (
        diabetes_model is None
        or diabetes_scaler is None
        or not diabetes_feature_names
        or diabetes_medians is None
    ):
        return jsonify({"error": "Diabetes model stack is not loaded.", "ok": False}), 503

    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object.", "ok": False}), 400

    logger.info(
        "Diabetes request payload keys: %s",
        sorted(k for k in payload if k not in ROUTING_KEYS),
    )

    canon = _canonicalize_payload(payload, DIABETES_ALIASES)
    missing = _missing_fields(canon, diabetes_feature_names)
    if missing:
        return (
            jsonify(
                {
                    "error": f"Missing required field(s): {', '.join(missing)}",
                    "ok": False,
                    "expected": diabetes_feature_names,
                }
            ),
            400,
        )

    vec: list[float] = []
    for i, name in enumerate(diabetes_feature_names):
        v = canon[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return jsonify({"error": f"Field '{name}' must be a number.", "ok": False}), 400
        fv = float(v)
        if name in DIABETES_ZERO_AS_NA and fv == 0.0:
            fv = float(diabetes_medians[i])
        vec.append(fv)

    logger.info(
        "Diabetes processed feature vector (%s): %s",
        diabetes_feature_names,
        [round(float(v), 5) for v in vec],
    )

    x = np.array([vec], dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*valid feature names.*",
                category=UserWarning,
            )
            x_sc = diabetes_scaler.transform(x)
        logger.info("Diabetes scaled vector (first 4): %s", [round(float(v), 5) for v in x_sc[0, :4]])
        pred = diabetes_model.predict(x_sc)[0]
        probability = _positive_class_probability(diabetes_model, x_sc)
    except Exception as exc:
        logger.exception("Diabetes prediction failed")
        return jsonify({"error": "Prediction failed.", "detail": str(exc), "ok": False}), 500

    try:
        result = _prediction_label(pred)
    except ValueError as exc:
        return jsonify({"error": str(exc), "ok": False}), 500

    logger.info("Diabetes prediction OK: class=%s label=%s", pred, result)
    response = {"ok": True, "model": "diabetes", "result": result}
    if probability is not None:
        response["probability"] = round(probability, 4)
    return jsonify(response)


@app.route("/predict", methods=["POST"])
def predict():
    raw = request.get_json(silent=True)
    if not isinstance(raw, dict):
        logger.warning("POST /predict: body is not a JSON object")
        return jsonify({"error": "Request body must be a JSON object.", "ok": False}), 400

    selector = raw.get("model") or raw.get("type") or raw.get("target") or raw.get("endpoint")
    logger.info(
        "POST /predict selector=%r feature_key_count=%s",
        selector,
        len([k for k in raw if k not in ROUTING_KEYS]),
    )
    payload = _strip_routing_keys(raw)

    if selector in ("heart", "heart_disease", "cardiac"):
        return _predict_heart_body(payload)
    if selector in ("diabetes", "diabetic"):
        return _predict_diabetes_body(payload)

    return (
        jsonify(
            {
                "error": "Missing or invalid model selector. "
                "Use model / type / target / endpoint with value 'heart' or 'diabetes'.",
                "ok": False,
            }
        ),
        400,
    )


@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    payload = request.get_json(silent=True)
    logger.info("POST /predict/heart (legacy)")
    return _predict_heart_body(payload)


@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    payload = request.get_json(silent=True)
    logger.info("POST /predict/diabetes (legacy)")
    return _predict_diabetes_body(payload)


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found.", "ok": False}), 404


@app.errorhandler(405)
def method_not_allowed(_e):
    return jsonify({"error": "Method not allowed. Use POST.", "ok": False}), 405


@app.errorhandler(500)
def server_error(_e):
    return jsonify({"error": "Internal server error.", "ok": False}), 500


try:
    _load_models()
except Exception as exc:
    logger.error("Could not load inference stack: %s", exc)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)