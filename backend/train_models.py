from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "disease-prediction-master" / "data"
ARTIFACTS_DIR = ROOT / "backend" / "artifacts"

HEART_FILE = DATA_DIR / "heart.csv"
DIABETES_FILE = DATA_DIR / "diabetes.csv"


def train_heart() -> None:
    features = [
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope",
    ]
    target = "HeartDisease"
    categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

    with HEART_FILE.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    category_maps: dict[str, dict[str, int]] = {}
    for col in categorical:
        classes = sorted({str(row[col]).strip() for row in rows})
        mapping = {label: idx for idx, label in enumerate(classes)}
        category_maps[col] = mapping

    x_matrix: list[list[float]] = []
    y_values: list[int] = []
    for row in rows:
        features_row: list[float] = []
        for feature in features:
            value = str(row[feature]).strip()
            if feature in category_maps:
                features_row.append(float(category_maps[feature][value]))
            else:
                features_row.append(float(value))
        x_matrix.append(features_row)
        y_values.append(int(row[target]))

    X = np.array(x_matrix, dtype=float)
    y = np.array(y_values, dtype=int)
    impute_map = {feature: float(np.median(X[:, idx])) for idx, feature in enumerate(features)}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_scaled, y)

    joblib.dump(model, ARTIFACTS_DIR / "heart_model.pkl")
    joblib.dump(scaler, ARTIFACTS_DIR / "heart_scaler.pkl")
    joblib.dump(features, ARTIFACTS_DIR / "heart_features.pkl")
    joblib.dump(category_maps, ARTIFACTS_DIR / "heart_category_maps.pkl")
    joblib.dump(impute_map, ARTIFACTS_DIR / "heart_impute_map.pkl")


def train_diabetes() -> None:
    features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    target = "Outcome"
    zero_as_missing = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}

    with DIABETES_FILE.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    x_matrix = np.array(
        [[float(str(row[feature]).strip()) for feature in features] for row in rows],
        dtype=float,
    )
    y = np.array([int(row[target]) for row in rows], dtype=int)

    impute_map: dict[str, float] = {}
    for idx, feature in enumerate(features):
        col_values = x_matrix[:, idx]
        if feature in zero_as_missing:
            non_zero = col_values[col_values != 0]
            median = float(np.median(non_zero)) if len(non_zero) else 0.0
            col_values[col_values == 0] = median
            x_matrix[:, idx] = col_values
            impute_map[feature] = median
        else:
            impute_map[feature] = float(np.median(col_values))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x_matrix)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_scaled, y)

    joblib.dump(model, ARTIFACTS_DIR / "diabetes_model.pkl")
    joblib.dump(scaler, ARTIFACTS_DIR / "diabetes_scaler.pkl")
    joblib.dump(features, ARTIFACTS_DIR / "diabetes_features.pkl")
    joblib.dump(impute_map, ARTIFACTS_DIR / "diabetes_impute_map.pkl")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    train_heart()
    train_diabetes()
    print(f"Artifacts generated in: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
