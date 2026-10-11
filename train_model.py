"""
Titanic Survival Prediction - Random Forest training script.

Run:  python train_model.py
Creates: model.pkl, encoders.pkl, features.pkl  (same folder as this script)
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE = Path(__file__).parent
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET = "survived"


def load_clean_data(csv_path=BASE / "titanic.csv"):
    """Load the dataset and fill missing values (same logic as the notebook)."""
    df = pd.read_csv(csv_path)
    data = df[FEATURES + [TARGET]].copy()
    data["age"] = data["age"].fillna(data["age"].median())
    data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])
    data["fare"] = data["fare"].fillna(data["fare"].median())
    return data


def encode(data):
    """Label-encode sex and embarked; return encoded frame + the encoders."""
    data = data.copy()
    encoders = {}
    for col in ["sex", "embarked"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le
    return data, encoders


def split(data):
    X, y = data[FEATURES], data[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train():
    data, encoders = encode(load_clean_data())
    X_train, X_test, y_train, y_test = split(data)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

    joblib.dump(model, BASE / "model.pkl")
    joblib.dump(encoders, BASE / "encoders.pkl")
    joblib.dump(FEATURES, BASE / "features.pkl")
    print("Saved: model.pkl, encoders.pkl, features.pkl")
    return model, encoders


if __name__ == "__main__":
    train()
