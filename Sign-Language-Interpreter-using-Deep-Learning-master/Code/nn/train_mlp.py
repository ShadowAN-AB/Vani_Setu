"""
Train a multi-layer perceptron (MLP) with Keras for letters, ASL phrases or ISL signs.

Network (same for every kind; only the input and output sizes change)
    input   88 numbers (letters: one hand) or 181 (phrases / ISL: two hands, see features.py)
    Dense 128, ReLU   -> Dropout 0.3
    Dense 64,  ReLU   -> Dropout 0.3
    Dense N,   softmax (one output per sign)

Training
    loss       sparse categorical cross-entropy
    optimiser  Adam, learning rate 0.001, batch size 64
    stopping   early stopping on validation loss (patience 10), best weights kept
    data       same 70 / 15 / 15 stratified split as the SVM; training part augmented;
               classes weighted by inverse frequency
    scaling    StandardScaler fitted on the training data only

The weights are exported to models/<kind>_mlp.npz and run on the server with
NumPy (nn/numpy_mlp.py), so TensorFlow is needed only on the training laptop:
    pip install -r requirements_train.txt

Usage (from Code/):
    .venv/bin/python nn/train_mlp.py --kind letter   compare with the SVM on letters
    .venv/bin/python nn/train_mlp.py --kind phrase   ASL phrases you recorded
    .venv/bin/python nn/train_mlp.py --kind isl      ISL signs you recorded

Writes models/<kind>_mlp.npz, reports/nn_<kind>.json, reports/nn_<kind>.md, reports/nn_<kind>_curves.png
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

import augment
import data
from features import batch_features, two_hand_features
from nn.numpy_mlp import NumpyMLP
from train_classifier import SEED, choose_threshold, load_letters, split_indices

MODELS_DIR = os.path.join(CODE_DIR, "models")
REPORT_DIR = os.path.join(CODE_DIR, "reports")
TITLES = {"letter": "Letters A–Z", "phrase": "ASL phrases", "isl": "ISL signs"}


def load(kind, min_per_label=10):
    """Return a dict with points, hands, labels, is_test, signers and a features() function."""
    if kind == "letter":
        points, labels, is_test, signers = load_letters(min_per_label)
        return {"points": points, "hands": None, "labels": labels, "is_test": is_test, "signers": signers,
                "features": lambda p, h: batch_features(p, 2), "feature_name": "v2"}
    points, hands, meta = data.load_two_hand_samples(kind)
    if not len(points):
        print(f"No {kind} samples yet. Record them in the app (see dataset/RECORDING_GUIDE.md).")
        sys.exit(1)
    labels = np.array([m["label"] for m in meta])
    is_test = np.array([m["split"] == "test" for m in meta])
    signers = np.array([m["signer"] for m in meta])
    counts = Counter(labels[~is_test])
    rare = sorted(k for k, v in counts.items() if v < min_per_label)
    if rare:
        print(f"Skipping signs with fewer than {min_per_label} training samples: {rare}")
    keep = ~np.isin(labels, rare)
    if len(set(labels[keep & ~is_test])) < 2:
        print("Need at least two signs with enough samples to train.")
        sys.exit(1)
    return {"points": points[keep], "hands": [h for h, k in zip(hands, keep) if k], "labels": labels[keep],
            "is_test": is_test[keep], "signers": signers[keep],
            "features": lambda p, h: two_hand_features(p, h), "feature_name": "two_hand"}


def build_model(n_in, n_out):
    import keras
    from keras import layers
    model = keras.Sequential([
        keras.Input((n_in,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_out, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def export(model, scaler, classes, info, path):
    dense = [l for l in model.layers if l.get_weights()]
    arrays = {"mean": scaler.mean_, "scale": scaler.scale_, "n_layers": np.array(len(dense)),
              "classes": np.array(classes), "info": np.array(json.dumps(info))}
    for i, layer in enumerate(dense):
        W, b = layer.get_weights()
        arrays[f"W{i}"], arrays[f"b{i}"] = W.astype(np.float32), b.astype(np.float32)
    np.savez_compressed(path, **arrays)


def plot_curves(history, kind, path):
    h = history.history
    epochs = np.arange(1, len(h["loss"]) + 1)
    best = int(np.argmin(h["val_loss"])) + 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, title in [(axes[0], "loss", "Cross-entropy loss"), (axes[1], "accuracy", "Accuracy")]:
        scale = 100 if key == "accuracy" else 1
        ax.plot(epochs, np.array(h[key]) * scale, color="#c96a12", lw=2, label="Training")
        ax.plot(epochs, np.array(h["val_" + key]) * scale, color="#008c99", lw=2, label="Validation")
        ax.axvline(best, color="#6b7780", lw=1, ls="--")
        ax.text(best, ax.get_ylim()[1], f" best epoch {best}", va="top", fontsize=8, color="#6b7780")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title + (" (%)" if key == "accuracy" else ""))
        ax.set_title(f"{TITLES[kind]}: {title.lower()}", loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6eaec")
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return best


def svm_on_same_test(points, labels):
    """Letters only: the live SVM's accuracy on exactly the same test samples."""
    import joblib
    path = os.path.join(MODELS_DIR, "sign_svm.joblib")
    if not os.path.exists(path):
        return None
    bundle = joblib.load(path)
    proba = bundle["model"].predict_proba(batch_features(points, bundle.get("feature_version", 2)))
    pred = np.array(bundle["model"].classes_)[proba.argmax(1)]
    return {"accuracy": round(float(accuracy_score(labels, pred)), 4),
            "macro_f1": round(float(f1_score(labels, pred, average="macro")), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["letter", "phrase", "isl"], default="letter")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()
    try:
        import keras
    except ImportError:
        print("TensorFlow / Keras is not installed. Run: pip install -r requirements_train.txt")
        sys.exit(2)
    keras.utils.set_random_seed(SEED)

    d = load(args.kind)
    pts, hands, labels, is_test = d["points"], d["hands"], d["labels"], d["is_test"]
    dev = np.where(~is_test)[0]
    tr_i, va_i, te_i = split_indices(labels[dev])
    tr, va, te = dev[tr_i], dev[va_i], dev[te_i]
    pick = (lambda idx: [hands[i] for i in idx]) if hands is not None else (lambda idx: None)

    if args.no_augment:
        tp, th, ty = pts[tr], pick(tr), labels[tr]
    elif hands is None:
        tp, ty = augment.augment(pts[tr], labels[tr]); th = None
    else:
        tp, th, ty = augment.augment_two_hands(pts[tr], pick(tr), labels[tr])
    feat = d["features"]
    scaler = StandardScaler().fit(feat(tp, th))
    x_tr = scaler.transform(feat(tp, th))
    x_va = scaler.transform(feat(pts[va], pick(va)))

    classes = sorted(set(labels))
    index = {c: i for i, c in enumerate(classes)}
    y_tr = np.array([index[c] for c in ty]); y_va = np.array([index[c] for c in labels[va]])
    counts = Counter(y_tr)
    class_weight = {k: len(y_tr) / (len(classes) * v) for k, v in counts.items()}

    print(f"{TITLES[args.kind]}: {len(tr)} train ({len(y_tr)} after augmentation) / {len(va)} validation / "
          f"{len(te)} test, {int(is_test.sum())} new-signer test, {len(classes)} classes, {x_tr.shape[1]} features")
    model = build_model(x_tr.shape[1], len(classes))
    start = time.perf_counter()
    history = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=args.epochs, batch_size=64,
                        class_weight=class_weight, verbose=0,
                        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
    fit_seconds = round(time.perf_counter() - start, 1)

    os.makedirs(MODELS_DIR, exist_ok=True); os.makedirs(REPORT_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{args.kind}_mlp.npz")
    info = {"kind": args.kind, "features": d["feature_name"], "threshold": 0.5,
            "layers": [x_tr.shape[1], 128, 64, len(classes)],
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    export(model, scaler, classes, info, model_path)

    # Everything below runs the exported NumPy model, i.e. exactly what the server will run
    net = NumpyMLP(model_path)
    val_proba = net.predict_proba(feat(pts[va], pick(va)))
    threshold = choose_threshold(val_proba, net.classes_, labels[va])
    info["threshold"] = threshold["threshold"]
    export(model, scaler, classes, info, model_path)
    keras_val = float(model.evaluate(x_va, y_va, verbose=0)[1])
    numpy_val = float(accuracy_score(labels[va], net.classes_[val_proba.argmax(1)]))

    test_pred = net.predict(feat(pts[te], pick(te)))
    best_epoch = plot_curves(history, args.kind, os.path.join(REPORT_DIR, f"nn_{args.kind}_curves.png"))
    result = {
        "kind": args.kind, "classes": classes, "features": d["feature_name"], "n_features": int(x_tr.shape[1]),
        "architecture": "Dense 128 ReLU -> Dropout 0.3 -> Dense 64 ReLU -> Dropout 0.3 -> Dense softmax",
        "parameters": int(model.count_params()),
        "split": {"train": int(len(tr)), "train_after_augmentation": int(len(y_tr)),
                  "validation": int(len(va)), "test": int(len(te)), "new_signer_test": int(is_test.sum())},
        "epochs_run": len(history.history["loss"]), "best_epoch": best_epoch, "fit_seconds": fit_seconds,
        "validation": {"accuracy": round(numpy_val, 4), "keras_accuracy": round(keras_val, 4)},
        "test": {"accuracy": round(float(accuracy_score(labels[te], test_pred)), 4),
                 "macro_f1": round(float(f1_score(labels[te], test_pred, average="macro")), 4)},
        "confidence_threshold": threshold,
    }
    if is_test.any():
        idx = np.where(is_test)[0]
        new_pred = net.predict(feat(pts[idx], pick(idx)))
        result["new_signer_test"] = {"accuracy": round(float(accuracy_score(labels[idx], new_pred)), 4), "n": int(len(idx))}
    if args.kind == "letter":
        result["svm_same_test"] = svm_on_same_test(pts[te], labels[te])
    one = feat(pts[te][:1], pick(te[:1]))
    start = time.perf_counter()
    for _ in range(500):
        net.predict_proba(one)
    result["numpy_ms_per_prediction"] = round((time.perf_counter() - start) / 500 * 1000, 4)

    with open(os.path.join(REPORT_DIR, f"nn_{args.kind}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    md = [f"# Neural network (MLP): {TITLES[args.kind]}", "",
          f"Architecture: {result['n_features']} inputs → Dense 128 (ReLU) → Dropout 0.3 → Dense 64 (ReLU) → "
          f"Dropout 0.3 → Dense {len(classes)} (softmax); {result['parameters']:,} parameters.",
          f"Adam (learning rate 0.001), cross-entropy loss, batch 64, early stopping after {result['epochs_run']} epochs "
          f"(best epoch {best_epoch}), {fit_seconds} s on CPU.", "",
          "| Set | Samples | Accuracy |", "|---|---|---|",
          f"| Validation | {len(va)} | {result['validation']['accuracy']:.1%} |",
          f"| Test | {len(te)} | {result['test']['accuracy']:.1%} (macro F1 {result['test']['macro_f1']:.3f}) |"]
    if result.get("new_signer_test"):
        md.append(f"| New signers | {result['new_signer_test']['n']} | {result['new_signer_test']['accuracy']:.1%} |")
    if result.get("svm_same_test"):
        md += ["", f"SVM on the same test samples: {result['svm_same_test']['accuracy']:.1%} "
               f"(macro F1 {result['svm_same_test']['macro_f1']:.3f})."]
    md += ["", f"Runs in NumPy on the server in {result['numpy_ms_per_prediction']} ms per prediction; "
           f"confidence threshold {threshold['threshold']:.2f}.", "",
           f"![Training curves](nn_{args.kind}_curves.png)", "", "```",
           classification_report(labels[te], test_pred, digits=3, zero_division=0), "```"]
    with open(os.path.join(REPORT_DIR, f"nn_{args.kind}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"Validation {numpy_val:.1%} (Keras {keras_val:.1%}), test {result['test']['accuracy']:.1%}, "
          f"best epoch {best_epoch}/{result['epochs_run']}, {fit_seconds} s")
    if result.get("svm_same_test"):
        print(f"SVM on the same test set: {result['svm_same_test']['accuracy']:.1%}")
    print(f"Saved {os.path.relpath(model_path, CODE_DIR)} and reports/nn_{args.kind}.md")


if __name__ == "__main__":
    main()
