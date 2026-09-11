"""
Compare several classifiers on the same landmark dataset.

This exists for the project report: it shows Random Forest was chosen by
measurement rather than by guesswork.

Usage (from Code/):
  .venv/bin/python compare_models.py

Writes:
  reports/model_comparison.json
  reports/model_comparison.md
  reports/model_comparison.png
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(APP_DIR, "dataset", "landmarks.csv")
REPORT_DIR = os.path.join(APP_DIR, "reports")

RANDOM_STATE = 42


def build_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=180, max_depth=18, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(C=10, gamma="scale", random_state=RANDOM_STATE)
        ),
        "K-Nearest Neighbours": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=18, random_state=RANDOM_STATE
        ),
    }


def main():
    if not os.path.exists(CSV_PATH):
        print("No dataset found at dataset/landmarks.csv")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    counts = df["label"].value_counts()
    rare = counts[counts < 3]
    if len(rare):
        df = df[~df["label"].isin(rare.index)]

    x = df.drop(columns=["label"]).to_numpy(dtype="float32")
    y = df["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    results = []
    for name, model in build_models().items():
        print(f"Training {name} ...")
        start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start

        pred = model.predict(x_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        cv = cross_val_score(model, x, y, cv=5, n_jobs=-1)

        results.append({
            "model": name,
            "test_accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1), 4),
            "cv5_mean": round(float(cv.mean()), 4),
            "cv5_std": round(float(cv.std()), 4),
            "fit_seconds": round(fit_seconds, 2),
        })

    results.sort(key=lambda r: r["test_accuracy"], reverse=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    summary = {
        "n_samples": int(len(df)),
        "n_classes": int(y.nunique()),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "results": results,
    }
    with open(os.path.join(REPORT_DIR, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Model comparison",
        "",
        f"Dataset: **{len(df)} samples**, **{y.nunique()} classes** "
        f"({len(x_train)} train / {len(x_test)} test, stratified split, seed {RANDOM_STATE}).",
        "",
        "| Model | Test accuracy | Macro F1 | 5-fold CV | Fit time |",
        "|-------|---------------|----------|-----------|----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['test_accuracy']:.1%} | {r['macro_f1']:.3f} | "
            f"{r['cv5_mean']:.1%} ± {r['cv5_std']:.1%} | {r['fit_seconds']}s |"
        )
    best = results[0]
    lines += [
        "",
        f"**Chosen model:** {best['model']} "
        f"({best['test_accuracy']:.1%} test accuracy, {best['cv5_mean']:.1%} cross-validated).",
        "",
        "Notes:",
        "",
        "- All models use the same 63-D wrist-relative landmark features.",
        "- Scaled models are wrapped in a `StandardScaler` pipeline.",
        "- Cross-validation is 5-fold over the full dataset.",
    ]
    md_path = os.path.join(REPORT_DIR, "model_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    names = [r["model"] for r in results]
    accs = [r["test_accuracy"] * 100 for r in results]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(names[::-1], accs[::-1], color="#0b5f66")
    bars[-1].set_color("#c97816")
    ax.set_xlabel("Test accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title(f"Classifier comparison on {len(df)} landmark samples")
    for bar, value in zip(bars, accs[::-1]):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    png_path = os.path.join(REPORT_DIR, "model_comparison.png")
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    print("\n".join(lines[4:]))
    print(f"\nSaved: {md_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
