from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "final_year_spam_model.joblib"

app = Flask(__name__)


bundle = joblib.load(ARTIFACT_PATH)
tfidf = bundle["tfidf"]
svd = bundle["svd"]
scaler = bundle["scaler"]
model = bundle["model"]
threshold = float(bundle["threshold"])
feature_columns = bundle["feature_columns"]


def uppercase_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(char.isupper() for char in letters) / len(letters)


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(char.isdigit() for char in text) / len(text)


def lexical_diversity(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def statistical_features(text: str) -> pd.DataFrame:
    message_length = len(text)
    word_count = len(text.split())
    avg_word_length = message_length / max(word_count, 1)
    exclamation_count = text.count("!")
    question_count = text.count("?")
    currency_count = text.count("$") + text.count("£")

    return pd.DataFrame(
        [[
            message_length,
            word_count,
            avg_word_length,
            exclamation_count,
            question_count,
            currency_count,
            digit_ratio(text),
            uppercase_ratio(text),
            lexical_diversity(text),
        ]],
        columns=feature_columns,
    )


def predict_message(text: str) -> dict[str, float | str]:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Please enter a message before predicting.")

    tfidf_features = tfidf.transform([cleaned_text])
    reduced_features = svd.transform(tfidf_features)
    stats = scaler.transform(statistical_features(cleaned_text))
    final_features = np.hstack([reduced_features, stats])

    probability = float(model.predict_proba(final_features)[0, 1])
    label = "spam" if probability >= threshold else "ham"

    return {
        "label": label,
        "spam_probability": round(probability * 100, 2),
        "ham_probability": round((1 - probability) * 100, 2),
        "threshold_used": round(threshold, 4),
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("message", "")

    try:
        result = predict_message(text)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
