"""
Compare six classifiers on exactly the same data, features and split.

This is the evidence for choosing the SVM: every model gets the same
70 / 15 / 15 split as train_classifier.py, the same augmented training data and
the same version-2 features. Models are ranked by validation macro F1; the test
set is reported for reference only.

Usage (from Code/):
  .venv/bin/python compare_models.py
  .venv/bin/python compare_models.py --features 1 --no-augment   original setup

Writes:
  reports/model_comparison.json
  reports/model_comparison.md
  reports/model_comparison.png
"""

import argparse
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import augment
from features import FEATURE_VERSION, batch_features
from train_classifier import REPORT_DIR, SEED, load_letters, split_indices


def build_models():
    return {
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced", random_state=SEED)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)),
        "MLP neural network": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, max_iter=400, random_state=SEED)),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=1, class_weight="balanced",
                                                random_state=SEED, n_jobs=-1),
        "K-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=int, default=FEATURE_VERSION, choices=[1, 2])
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    points, labels, is_test, _ = load_letters()
    points, labels = points[~is_test], labels[~is_test]
    tr, va, te = split_indices(labels)
    train_p, train_y = (points[tr], labels[tr]) if args.no_augment else augment.augment(points[tr], labels[tr])
    f_train = batch_features(train_p, args.features)
    f_val = batch_features(points[va], args.features)
    f_test = batch_features(points[te], args.features)

    results = []
    for name, model in build_models().items():
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(f_train, train_y)
        fit = time.perf_counter() - start
        val_pred, test_pred = model.predict(f_val), model.predict(f_test)
        start = time.perf_counter()
        for i in range(100):
            model.predict(f_test[i % len(f_test)][None])
        ms = (time.perf_counter() - start) * 10
        results.append({
            "model": name,
            "val_accuracy": round(float(accuracy_score(labels[va], val_pred)), 4),
            "val_macro_f1": round(float(f1_score(labels[va], val_pred, average="macro")), 4),
            "test_accuracy": round(float(accuracy_score(labels[te], test_pred)), 4),
            "test_macro_f1": round(float(f1_score(labels[te], test_pred, average="macro")), 4),
            "fit_seconds": round(fit, 2),
            "ms_per_prediction": round(ms, 3),
        })
        r = results[-1]
        print(f"{name:<22} val {r['val_accuracy']:.1%}  test {r['test_accuracy']:.1%}  "
              f"macro F1 {r['test_macro_f1']:.3f}  fit {fit:.1f}s  {ms:.2f} ms")

    results.sort(key=lambda r: (r["val_macro_f1"], r["val_accuracy"]), reverse=True)
    best = results[0]
    os.makedirs(REPORT_DIR, exist_ok=True)
    setup = {"feature_version": args.features, "n_features": int(f_train.shape[1]), "augmented": not args.no_augment,
             "train": int(len(tr)), "train_after_augmentation": int(len(train_y)),
             "validation": int(len(va)), "test": int(len(te))}
    with open(os.path.join(REPORT_DIR, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump({"setup": setup, "results": results}, f, indent=2)

    lines = [
        "# Model comparison", "",
        f"Same data and split for every model: {len(tr)} train / {len(va)} validation / {len(te)} test "
        f"(stratified, seed {SEED}). Features: version {args.features} ({setup['n_features']} numbers). "
        + ("Training data augmented to " + str(len(train_y)) + " samples." if not args.no_augment else "No augmentation."),
        "",
        "| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 | Fit time | Time per prediction |",
        "|-------|---------------------|---------------------|---------------|---------------|----------|---------------------|",
    ]
    for r in results:
        lines.append(f"| {r['model']} | {r['val_accuracy']:.1%} | {r['val_macro_f1']:.3f} | {r['test_accuracy']:.1%} | "
                     f"{r['test_macro_f1']:.3f} | {r['fit_seconds']}s | {r['ms_per_prediction']} ms |")
    lines += ["", f"**Chosen by validation macro F1:** {best['model']}.", "",
              "The live app uses the SVM after tuning C and gamma in `train_classifier.py`; see `reports/metrics.json`."]
    with open(os.path.join(REPORT_DIR, "model_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    names = [r["model"] for r in results][::-1]
    vals = [r["test_accuracy"] * 100 for r in results][::-1]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(names, vals, color=["#c96a12" if n == best["model"] else "#008c99" for n in names])
    for i, v in enumerate(vals):
        ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 105); ax.set_xlabel("Test accuracy (%)")
    ax.set_title(f"Six classifiers, same data and features (v{args.features})", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "model_comparison.png"), dpi=150); plt.close(fig)
    print(f"Best by validation: {best['model']}. Wrote reports/model_comparison.md")


if __name__ == "__main__":
    main()
