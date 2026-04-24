from __future__ import annotations

import copy
import json
import random
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


SEED = 42
DATA_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "assets" / "final_year_output"
ARTIFACT_DIR = ROOT / "artifacts"
DATA_PATH = DATA_DIR / "sms.tsv"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        DATA_PATH.write_text(response.text, encoding="utf-8")

    data = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    data["label_num"] = data["label"].map({"ham": 0, "spam": 1})
    return data


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


def add_text_features(data: pd.DataFrame) -> pd.DataFrame:
    enriched = data.copy()
    enriched["message_length"] = enriched["message"].str.len()
    enriched["word_count"] = enriched["message"].str.split().str.len()
    enriched["avg_word_length"] = (
        enriched["message_length"] / enriched["word_count"].replace(0, 1)
    )
    enriched["exclamation_count"] = enriched["message"].str.count("!")
    enriched["question_count"] = enriched["message"].str.count(r"\?")
    enriched["currency_count"] = enriched["message"].str.count(r"[$£]")
    enriched["digit_ratio"] = enriched["message"].apply(digit_ratio)
    enriched["uppercase_ratio"] = enriched["message"].apply(uppercase_ratio)
    enriched["lexical_diversity"] = enriched["message"].apply(lexical_diversity)
    return enriched


def plot_class_distribution(data: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    ax = sns.countplot(
        data=data,
        x="label",
        hue="label",
        palette=["#4C78A8", "#E45756"],
        legend=False,
    )
    ax.set_title("SMS Spam Dataset Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution_final.png", dpi=220)
    plt.close()


def plot_message_length(data: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=data,
        x="message_length",
        hue="label",
        bins=40,
        kde=True,
        element="step",
        palette=["#4C78A8", "#E45756"],
    )
    plt.title("Message Length Distribution by Class")
    plt.xlabel("Message Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "message_length_distribution_final.png", dpi=220)
    plt.close()


def plot_feature_heatmap(data: pd.DataFrame) -> None:
    numeric_cols = [
        "label_num",
        "message_length",
        "word_count",
        "avg_word_length",
        "exclamation_count",
        "question_count",
        "currency_count",
        "digit_ratio",
        "uppercase_ratio",
        "lexical_diversity",
    ]
    corr = data[numeric_cols].corr()
    plt.figure(figsize=(9, 6))
    sns.heatmap(corr, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_correlation_final.png", dpi=220)
    plt.close()


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5, 0.0

    precision_slice = precision[:-1]
    recall_slice = recall[:-1]
    f1_scores = (2 * precision_slice * recall_slice) / (
        precision_slice + recall_slice + 1e-12
    )

    eligible = np.where(precision_slice >= 0.95)[0]
    if len(eligible) > 0:
        best_index = eligible[np.argmax(recall_slice[eligible])]
    else:
        best_index = int(np.argmax(f1_scores))

    return float(thresholds[best_index]), float(f1_scores[best_index])


def train_epoch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 35,
    patience: int = 6,
) -> tuple[MLPClassifier, list[dict[str, float]], int]:
    model = MLPClassifier(
        hidden_layer_sizes=(256, 64),
        activation="relu",
        solver="adam",
        batch_size=64,
        learning_rate_init=0.001,
        alpha=1e-4,
        max_iter=1,
        warm_start=True,
        random_state=SEED,
    )

    history: list[dict[str, float]] = []
    best_model: MLPClassifier | None = None
    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.fit(X_train, y_train)

        train_probs = model.predict_proba(X_train)[:, 1]
        val_probs = model.predict_proba(X_val)[:, 1]
        train_pred = (train_probs >= 0.5).astype(int)
        val_pred = (val_probs >= 0.5).astype(int)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": log_loss(y_train, train_probs, labels=[0, 1]),
            "val_loss": log_loss(y_val, val_probs, labels=[0, 1]),
            "train_f1": f1_score(y_train, train_pred),
            "val_f1": f1_score(y_val, val_pred),
            "val_auc": roc_auc_score(y_val, val_probs),
        }
        history.append(epoch_metrics)

        if epoch_metrics["val_f1"] > best_val_f1:
            best_val_f1 = epoch_metrics["val_f1"]
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    if best_model is None:
        best_model = copy.deepcopy(model)

    return best_model, history, best_epoch


def plot_training_curves(history: list[dict[str, float]]) -> None:
    history_df = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_title("Epoch-wise Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Log Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_f1"], label="Train F1", linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_f1"], label="Validation F1", linewidth=2)
    axes[1].set_title("Epoch-wise F1 Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=220)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    plt.figure(figsize=(6.5, 5))
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
    display.plot(cmap="Blues", colorbar=False)
    plt.title("Final Neural Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_confusion_matrix.png", dpi=220)
    plt.close()


def plot_roc_pr_curves(y_true: np.ndarray, probabilities: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(fpr, tpr, linewidth=2, color="#1f77b4")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].text(0.55, 0.12, f"AUC = {auc(fpr, tpr):.4f}")

    axes[1].plot(recall, precision, linewidth=2, color="#d62728")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_pr_curves.png", dpi=220)
    plt.close()


def plot_threshold_analysis(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    precision = precision[:-1]
    recall = recall[:-1]
    if len(thresholds) == 0:
        return
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision, label="Precision", linewidth=2)
    plt.plot(thresholds, recall, label="Recall", linewidth=2)
    plt.plot(thresholds, f1_scores, label="F1 Score", linewidth=2)
    plt.axvline(threshold, color="black", linestyle="--", label=f"Chosen Threshold = {threshold:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_analysis.png", dpi=220)
    plt.close()


def plot_model_comparison(metrics: dict[str, dict[str, float]]) -> None:
    comparison = pd.DataFrame(
        [
            {"Model": "Baseline TF-IDF + LR", **metrics["baseline"]},
            {"Model": "Final TF-IDF + Stats + MLP", **metrics["final_neural"]},
        ]
    )
    melted = comparison.melt(id_vars="Model", value_vars=["accuracy", "f1", "roc_auc"])

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=melted, x="Model", y="value", hue="variable", palette="viridis")
    ax.set_title("Model Comparison Across Key Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0.85, 1.0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=220)
    plt.close()


def save_misclassifications(messages: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> None:
    mistakes = pd.DataFrame(
        {
            "message": messages.values,
            "actual": y_true,
            "predicted": y_pred,
            "probability_spam": probabilities,
        }
    )
    mistakes = mistakes[mistakes["actual"] != mistakes["predicted"]].sort_values(
        "probability_spam", ascending=False
    )
    mistakes.to_csv(OUTPUT_DIR / "misclassified_examples.csv", index=False)


def main() -> None:
    set_seed()
    ensure_dirs()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.figsize"] = (10, 5)

    data = add_text_features(load_dataset())

    plot_class_distribution(data)
    plot_message_length(data)
    plot_feature_heatmap(data)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        data["message"],
        data["label_num"],
        test_size=0.15,
        random_state=SEED,
        stratify=data["label_num"],
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1765,
        random_state=SEED,
        stratify=y_train_full,
    )

    train_features = data.loc[X_train.index]
    val_features = data.loc[X_val.index]
    test_features = data.loc[X_test.index]

    tfidf = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    baseline_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    baseline_model.fit(X_train_tfidf, y_train)
    baseline_probs = baseline_model.predict_proba(X_test_tfidf)[:, 1]
    baseline_pred = (baseline_probs >= 0.5).astype(int)

    stat_cols = [
        "message_length",
        "word_count",
        "avg_word_length",
        "exclamation_count",
        "question_count",
        "currency_count",
        "digit_ratio",
        "uppercase_ratio",
        "lexical_diversity",
    ]
    scaler = StandardScaler()
    X_train_stats = scaler.fit_transform(train_features[stat_cols])
    X_val_stats = scaler.transform(val_features[stat_cols])
    X_test_stats = scaler.transform(test_features[stat_cols])

    svd = TruncatedSVD(n_components=220, random_state=SEED)
    X_train_reduced = svd.fit_transform(X_train_tfidf)
    X_val_reduced = svd.transform(X_val_tfidf)
    X_test_reduced = svd.transform(X_test_tfidf)

    X_train_final = np.hstack([X_train_reduced, X_train_stats])
    X_val_final = np.hstack([X_val_reduced, X_val_stats])
    X_test_final = np.hstack([X_test_reduced, X_test_stats])

    final_model, history, best_epoch = train_epoch_model(
        X_train_final, y_train.to_numpy(), X_val_final, y_val.to_numpy(), max_epochs=40, patience=7
    )
    plot_training_curves(history)

    val_probs = final_model.predict_proba(X_val_final)[:, 1]
    threshold, best_val_f1 = choose_threshold(y_val.to_numpy(), val_probs)

    final_probs = final_model.predict_proba(X_test_final)[:, 1]
    final_pred = (final_probs >= threshold).astype(int)

    plot_confusion(y_test.to_numpy(), final_pred)
    plot_roc_pr_curves(y_test.to_numpy(), final_probs)
    plot_threshold_analysis(y_val.to_numpy(), val_probs, threshold)
    save_misclassifications(X_test, y_test.to_numpy(), final_pred, final_probs)

    metrics = {
        "baseline": {
            "accuracy": accuracy_score(y_test, baseline_pred),
            "f1": f1_score(y_test, baseline_pred),
            "roc_auc": roc_auc_score(y_test, baseline_probs),
        },
        "final_neural": {
            "accuracy": accuracy_score(y_test, final_pred),
            "f1": f1_score(y_test, final_pred),
            "roc_auc": roc_auc_score(y_test, final_probs),
            "precision": precision_score(y_test, final_pred),
            "recall": recall_score(y_test, final_pred),
        },
        "training": {
            "best_epoch": best_epoch,
            "chosen_threshold": threshold,
            "best_validation_f1": best_val_f1,
        },
    }
    plot_model_comparison(metrics)

    report_bundle = {
        "metrics": metrics,
        "classification_report": classification_report(y_test, final_pred, output_dict=True),
        "dataset": {
            "total_samples": int(len(data)),
            "train_samples": int(len(X_train)),
            "validation_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
        },
    }
    (ARTIFACT_DIR / "final_year_metrics.json").write_text(
        json.dumps(report_bundle, indent=2), encoding="utf-8"
    )

    joblib.dump(
        {
            "tfidf": tfidf,
            "svd": svd,
            "scaler": scaler,
            "model": final_model,
            "threshold": threshold,
            "feature_columns": stat_cols,
        },
        ARTIFACT_DIR / "final_year_spam_model.joblib",
    )

    print("Final-year pipeline completed.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
