"""
Train a Random Forest on collected MediaPipe landmarks.

Usage (from Code/):
  .venv/bin/python train_classifier.py

Writes:
  models/sign_rf.joblib
  reports/metrics.json
  reports/confusion_matrix.png
  reports/classification_report.txt
"""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(APP_DIR, "dataset", "landmarks.csv")
MODEL_PATH = os.path.join(APP_DIR, "models", "sign_rf.joblib")
REPORT_DIR = os.path.join(APP_DIR, "reports")


def main():
    if not os.path.exists(CSV_PATH):
        print("No dataset found at dataset/landmarks.csv")
        print("Open the web app, use Record Dataset, then run this script again.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    if "label" not in df.columns or len(df) < 20:
        print(f"Need at least 20 labeled rows. Found {len(df)}.")
        sys.exit(1)

    counts = df["label"].value_counts()
    rare = counts[counts < 3]
    if len(rare):
        print("Dropping labels with fewer than 3 samples:", list(rare.index))
        df = df[~df["label"].isin(rare.index)]

    if df["label"].nunique() < 2:
        print("Need at least two different signs to train.")
        sys.exit(1)

    x = df.drop(columns=["label"])
    y = df["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=180,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(labels) + 2), max(5, 0.45 * len(labels) + 2)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Sign classifier confusion matrix  (acc={acc:.1%})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=140)
    plt.close(fig)

    metrics = {
        "accuracy": round(float(acc), 4),
        "n_samples": int(len(df)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_classes": int(y.nunique()),
        "labels": labels,
        "per_class_counts": {k: int(v) for k, v in counts.items()},
        "model": "RandomForestClassifier",
    }
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"Accuracy: {acc:.1%}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved matrix: {cm_path}")
    print("Restart the Flask server so it loads the new model.")


if __name__ == "__main__":
    main()
