"""
Train the letter classifier (RBF SVM) on raw hand landmarks.

Steps
  1. Load every letter sample (ASL Now + team recordings) as raw 21-point hands.
     Samples recorded with "Test set" ticked are held out as the new-signer test set.
  2. Split the rest 70 / 15 / 15 into train / validation / test, stratified by letter.
  3. Augment the training part only (mirror, rotate, jitter; see augment.py).
  4. Compute version-2 features (see features.py).
  5. Grid search C and gamma, scoring each setting on the validation set.
  6. Train the best setting on the augmented training data with Platt scaling
     (a sigmoid fitted to the SVM's decision values by internal 5-fold CV), so
     every prediction comes with a probability.
  7. Pick the confidence threshold on the validation set.
  8. Report on the test set once, on harder versions of it (mirrored, tilted,
     smaller hand), and on the new-signer test set if there is one.

Usage (from Code/):
  .venv/bin/python train_classifier.py              full run with grid search
  .venv/bin/python train_classifier.py --quick      reuse the last best C and gamma
  .venv/bin/python train_classifier.py --features 1 --no-augment   the original setup

Writes
  models/sign_svm.joblib         model bundle used by the server
  reports/metrics.json           all numbers below
  reports/grid_search.json       validation score for every C / gamma
  reports/classification_report.txt
  reports/confusion_matrix.png
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import augment
import data
from features import FEATURE_VERSION, batch_features

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "models", "sign_svm.joblib")
REPORT_DIR = os.path.join(APP_DIR, "reports")
GRID_C = [3, 10, 30, 100]
GRID_GAMMA = ["scale", 0.01, 0.03]
SEED = 42


def split_indices(labels, seed=SEED):
    """70 / 15 / 15 stratified split of the given labels -> (train, val, test) index arrays."""
    idx = np.arange(len(labels))
    dev, test = train_test_split(idx, test_size=0.15, stratify=labels, random_state=seed)
    train, val = train_test_split(dev, test_size=0.15 / 0.85, stratify=labels[dev], random_state=seed)
    return train, val, test


def load_letters(min_per_label=10):
    """Points and labels for training, plus the new-signer test set."""
    points, meta = data.load_samples("letter")
    if not len(points):
        print("No letter samples found. Run build_dataset.py or record samples in the app.")
        sys.exit(1)
    labels = np.array([m["label"] for m in meta])
    is_test = np.array([m["split"] == "test" for m in meta])
    signers = np.array([m["signer"] for m in meta])

    counts = Counter(labels[~is_test])
    rare = sorted(k for k, v in counts.items() if v < min_per_label)
    if rare:
        print(f"Skipping labels with fewer than {min_per_label} training samples: {rare}")
    keep = ~np.isin(labels, rare)
    return points[keep], labels[keep], is_test[keep], signers[keep]


def svm(C, gamma, probability=False):
    extra = {"probability": True} if probability else {}
    return make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, class_weight="balanced",
                                               random_state=SEED, **extra))


def grid_search(f_train, y_train, f_val, y_val):
    results = []
    for C in GRID_C:
        for gamma in GRID_GAMMA:
            start = time.perf_counter()
            model = svm(C, gamma).fit(f_train, y_train)
            pred = model.predict(f_val)
            results.append({
                "C": C, "gamma": gamma,
                "val_accuracy": round(float(accuracy_score(y_val, pred)), 4),
                "val_macro_f1": round(float(f1_score(y_val, pred, average="macro")), 4),
                "fit_seconds": round(time.perf_counter() - start, 2),
            })
            print(f"  C={C:<4} gamma={str(gamma):<6} val acc {results[-1]['val_accuracy']:.3f}  "
                  f"macro F1 {results[-1]['val_macro_f1']:.3f}")
    best = max(results, key=lambda r: (r["val_macro_f1"], r["val_accuracy"], -r["C"]))
    return best, results


def choose_threshold(proba, classes, y_true, keep_correct=0.98):
    """Highest confidence cut-off that still keeps `keep_correct` of the correct predictions."""
    conf = proba.max(1)
    correct = classes[proba.argmax(1)] == y_true
    best = {"threshold": 0.3}
    for t in np.round(np.arange(0.30, 0.91, 0.05), 2):
        kept = conf >= t
        kept_correct = (kept & correct).sum() / max(correct.sum(), 1)
        rejected_errors = (~kept & ~correct).sum() / max((~correct).sum(), 1)
        if kept_correct >= keep_correct:
            best = {"threshold": float(t), "kept_correct": round(float(kept_correct), 4),
                    "rejected_errors": round(float(rejected_errors), 4)}
    return best


def evaluate(model, classes, points, labels, version):
    proba = model.predict_proba(batch_features(points, version))
    pred = classes[proba.argmax(1)]
    return pred, proba, {
        "n": int(len(labels)),
        "accuracy": round(float(accuracy_score(labels, pred)), 4),
        "macro_f1": round(float(f1_score(labels, pred, average="macro")), 4),
    }


def robustness(model, classes, points, labels, version):
    """Accuracy on harder copies of the test set: what a left hand, a tilt or a far hand does."""
    wrist = points[:, :1]
    variants = {
        "original": points,
        "mirrored (other hand)": augment.mirror(points),
        "tilted +15 degrees": augment.rotate(points, np.full(len(points), 15.0)),
        "tilted -15 degrees": augment.rotate(points, np.full(len(points), -15.0)),
        "hand 40% smaller (farther away)": (points - wrist) * 0.6 + wrist,
    }
    return {name: evaluate(model, classes, p, labels, version)[2]["accuracy"] for name, p in variants.items()}


def save_confusion(labels, pred, classes, acc, path):
    cm = confusion_matrix(labels, pred, labels=classes)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(6, 0.42 * n + 2), max(5, 0.42 * n + 2)))
    ax.imshow(cm, cmap="Blues")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if i == j and cm[i, j] > cm.max() / 2 else ("#c96a12" if i != j else "#1f2a30"))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=8); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Letter classifier on the test set (accuracy {acc:.1%}); errors in orange")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    pairs = [(int(cm[i, j]), classes[i], classes[j]) for i in range(n) for j in range(n) if i != j and cm[i, j]]
    return [{"actual": a, "predicted": p, "count": c} for c, a, p in sorted(pairs, reverse=True)[:8]]


def latency_ms(model, sample, version, runs=200):
    feats = batch_features(sample[None], version)
    model.predict_proba(feats)
    start = time.perf_counter()
    for _ in range(runs):
        model.predict_proba(batch_features(sample[None], version))
    return round((time.perf_counter() - start) / runs * 1000, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=int, default=FEATURE_VERSION, choices=[1, 2])
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--quick", action="store_true", help="skip the grid search; reuse the last best C and gamma")
    args = ap.parse_args()
    version = args.features
    os.makedirs(REPORT_DIR, exist_ok=True)

    points, labels, is_test, signers = load_letters()
    dev_p, dev_y = points[~is_test], labels[~is_test]
    new_p, new_y = points[is_test], labels[is_test]
    tr, va, te = split_indices(dev_y)
    print(f"Samples: {len(dev_y)} for train/val/test ({len(tr)} / {len(va)} / {len(te)}), "
          f"{len(new_y)} in the new-signer test set, {len(set(dev_y))} letters")

    if args.no_augment:
        train_p, train_y = dev_p[tr], dev_y[tr]
    else:
        train_p, train_y = augment.augment(dev_p[tr], dev_y[tr])
        print(f"Augmented training set: {len(dev_y[tr])} -> {len(train_y)} samples")
    f_train = batch_features(train_p, version)
    f_val = batch_features(dev_p[va], version)

    grid_path = os.path.join(REPORT_DIR, "grid_search.json")
    if args.quick and os.path.exists(grid_path):
        with open(grid_path, encoding="utf-8") as f:
            best = json.load(f)["best"]
        grid = None
        print(f"Quick mode: using C={best['C']}, gamma={best['gamma']}")
    else:
        print("Grid search (scored on the validation set):")
        best, grid = grid_search(f_train, train_y, f_val, dev_y[va])
        print(f"Best: C={best['C']}, gamma={best['gamma']}")
        with open(grid_path, "w", encoding="utf-8") as f:
            json.dump({"best": best, "results": grid, "feature_version": version,
                       "augmented": not args.no_augment}, f, indent=2)

    start = time.perf_counter()
    with warnings.catch_warnings():
        # scikit-learn 1.9 marks SVC(probability=True) as deprecated (removed in 1.11);
        # requirements pin scikit-learn < 1.10, so it is safe here.
        warnings.simplefilter("ignore", FutureWarning)
        model = svm(best["C"], best["gamma"], probability=True).fit(f_train, train_y)
    fit_seconds = round(time.perf_counter() - start, 2)
    classes = np.array(model.classes_)

    val_proba = model.predict_proba(f_val)
    threshold = choose_threshold(val_proba, classes, dev_y[va])
    val_metrics = evaluate(model, classes, dev_p[va], dev_y[va], version)[2]
    test_pred, _, test_metrics = evaluate(model, classes, dev_p[te], dev_y[te], version)
    robust = robustness(model, classes, dev_p[te], dev_y[te], version)
    report = classification_report(dev_y[te], test_pred, digits=3, zero_division=0)
    cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
    confusions = save_confusion(dev_y[te], test_pred, list(classes), test_metrics["accuracy"], cm_path)

    new_signer = None
    if len(new_y):
        new_pred, _, new_signer = evaluate(model, classes, new_p, new_y, version)
        per = {}
        for s in sorted(set(signers[is_test])):
            m = signers[is_test] == s
            per[s] = {"n": int(m.sum()), "accuracy": round(float(accuracy_score(new_y[m], new_pred[m])), 4)}
        new_signer["per_signer"] = per

    metrics = {
        # keys read by the app's Retrain button
        "accuracy": test_metrics["accuracy"],
        "n_samples": int(len(labels)),
        "n_classes": int(len(classes)),
        "model": f"StandardScaler + SVC(rbf, C={best['C']}, gamma={best['gamma']}, class_weight=balanced), "
                 "Platt-scaled probabilities",
        "feature_version": version,
        "n_features": int(f_train.shape[1]),
        "augmented": not args.no_augment,
        "split": {"train": int(len(tr)), "train_after_augmentation": int(len(train_y)),
                  "validation": int(len(va)), "test": int(len(te)), "new_signer_test": int(len(new_y))},
        "best_params": {"C": best["C"], "gamma": best["gamma"]},
        "validation": val_metrics,
        "test": test_metrics,
        "robustness_test_accuracy": robust,
        "new_signer_test": new_signer,
        "confidence_threshold": threshold,
        "top_confusions": confusions,
        "latency_ms_per_frame": latency_ms(model, dev_p[te][0], version),
        "fit_seconds": fit_seconds,
        "per_class_counts": {k: int(v) for k, v in sorted(Counter(labels).items())},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": model,
        "classes": list(classes),
        "feature_version": version,
        "threshold": threshold["threshold"],
        "metrics": {k: metrics[k] for k in ("accuracy", "test", "validation", "best_params", "trained_at")},
        "sklearn_version": sklearn.__version__,
    }, MODEL_PATH)
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"Validation accuracy {val_metrics['accuracy']:.1%}, test accuracy {test_metrics['accuracy']:.1%}, "
          f"test macro F1 {test_metrics['macro_f1']:.3f}")
    print("Robustness (test set):", ", ".join(f"{k} {v:.1%}" for k, v in robust.items()))
    if new_signer:
        print(f"New-signer test: {new_signer['accuracy']:.1%} on {new_signer['n']} samples")
    print(f"Confidence threshold {threshold['threshold']:.2f} "
          f"(keeps {threshold.get('kept_correct', 0):.0%} of correct predictions, "
          f"rejects {threshold.get('rejected_errors', 0):.0%} of errors)")
    print(f"Saved model: {os.path.relpath(MODEL_PATH, APP_DIR)}")


if __name__ == "__main__":
    main()
