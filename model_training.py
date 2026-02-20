"""Model training pipeline for aspect-based sentiment analysis."""

import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import preprocess_series

MODEL_PATH = Path("sentiment_model.pkl")


def _default_training_data() -> pd.DataFrame:
    """Small fallback dataset so the app can run end-to-end out of the box."""
    samples = [
        ("Faculty members are very supportive and explain concepts clearly", "Positive"),
        ("Teachers are often late and classes feel unorganized", "Negative"),
        ("Infrastructure is clean and labs are well equipped", "Positive"),
        ("Library timing is okay but nothing exceptional", "Neutral"),
        ("Curriculum is outdated and needs industry updates", "Negative"),
        ("Course content is balanced and relevant", "Positive"),
        ("Placements are average this year", "Neutral"),
        ("Placement cell brought many good companies", "Positive"),
        ("Management takes too long to resolve student issues", "Negative"),
        ("Administrative support is decent", "Neutral"),
        ("Campus wifi is unreliable in hostels", "Negative"),
        ("Internship opportunities are excellent", "Positive"),
    ]
    return pd.DataFrame(samples, columns=["feedback", "sentiment"])


def load_training_data(csv_path: str = "") -> pd.DataFrame:
    """Load training data from CSV or use fallback sample data.

    Expected columns in CSV: feedback, sentiment
    """
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        df = _default_training_data()

    required_cols = {"feedback", "sentiment"}
    missing = required_cols.difference(df.columns.str.lower())
    if missing:
        # Attempt to normalize case-insensitive column names.
        rename_map = {}
        for col in df.columns:
            lower_col = col.lower()
            if lower_col in required_cols:
                rename_map[col] = lower_col
        df = df.rename(columns=rename_map)

    if not required_cols.issubset(set(df.columns)):
        raise ValueError("Training CSV must include 'feedback' and 'sentiment' columns.")

    return df[["feedback", "sentiment"]].dropna()


def train_model(csv_path: str = "") -> Tuple[Pipeline, dict]:
    """Train TF-IDF + Logistic Regression model and persist it to disk."""
    df = load_training_data(csv_path)
    df["clean_feedback"] = preprocess_series(df["feedback"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_feedback"],
        df["sentiment"],
        test_size=0.25,
        random_state=42,
        stratify=df["sentiment"],
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(pipeline, model_file)

    return pipeline, metrics


def load_model() -> Pipeline:
    """Load a pickled sentiment model, training one if it does not exist."""
    if not MODEL_PATH.exists():
        model, _ = train_model()
        return model

    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)
