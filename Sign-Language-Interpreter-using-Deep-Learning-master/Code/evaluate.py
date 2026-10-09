"""
Evaluation report for the letter classifier: the evidence behind every design choice.

  1. Before / after   original setup (63 features, no augmentation) against the new one,
                      on the normal test set and on harder copies of it
  2. Cross-validation 5-fold stratified CV on train + validation (augmenting inside each fold)
  3. Learning curve   does more training data still help?
  4. Validation curve how C moves the SVM from underfitting to overfitting
  5. 2-D views        PCA and t-SNE of the features, with the most-confused letters highlighted
  6. Speed            time to compute features and classify one frame
  7. New signers      accuracy per person on the held-out "Test set" recordings, if any

Usage (from Code/):  .venv/bin/python evaluate.py        (about 1-2 minutes)

Writes reports/evaluation.md, reports/evaluation.json and reports/eval_*.png
"""

import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import augment
from features import batch_features
from train_classifier import REPORT_DIR, SEED, load_letters, split_indices

TEAL, AMBER, PURPLE, GREY, INK = "#008c99", "#c96a12", "#7a5cc7", "#c9d0d3", "#1f2a30"
HIGHLIGHT = {"D": TEAL, "X": AMBER, "R": PURPLE}


def svm(C=10, gamma="scale"):
    return make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, class_weight="balanced", random_state=SEED))


def harder_versions(p):
    wrist = p[:, :1]
    return {
        "Original": p,
        "Other hand (mirrored)": augment.mirror(p),
        "Tilted 15°": augment.rotate(p, np.full(len(p), 15.0)),
        "Hand 40% smaller": (p - wrist) * 0.6 + wrist,
    }


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6eaec")
    ax.set_axisbelow(True)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    warnings.filterwarnings("ignore", category=FutureWarning)
    points, labels, is_test, signers = load_letters()
    dev_p, dev_y = points[~is_test], labels[~is_test]
    tr, va, te = split_indices(dev_y)
    out = {"split": {"train": len(tr), "validation": len(va), "test": len(te), "new_signer_test": int(is_test.sum())}}

    # 1. Before / after
    print("1/7 before and after")
    setups = {
        "Original (63 features, no augmentation)": (1, False),
        "63 features + augmentation": (1, True),
        "88 features, no augmentation": (2, False),
        "New (88 features + augmentation)": (2, True),
    }
    ablation = {}
    for name, (version, aug) in setups.items():
        tp, ty = augment.augment(dev_p[tr], dev_y[tr]) if aug else (dev_p[tr], dev_y[tr])
        model = svm().fit(batch_features(tp, version), ty)
        ablation[name] = {k: round(float(accuracy_score(dev_y[te], model.predict(batch_features(v, version)))), 4)
                          for k, v in harder_versions(dev_p[te]).items()}
    out["before_after"] = ablation

    conds = list(next(iter(ablation.values())).keys())
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.38
    xs = np.arange(len(conds))
    for i, (name, col) in enumerate([("Original (63 features, no augmentation)", GREY), ("New (88 features + augmentation)", TEAL)]):
        vals = [ablation[name][c] * 100 for c in conds]
        ax.bar(xs + (i - 0.5) * width, vals, width - 0.04, color=col, label=name)
        for x, v in zip(xs + (i - 0.5) * width, vals):
            ax.text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(conds); ax.set_ylim(0, 110); ax.set_ylabel("Test accuracy (%)")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    style(ax, "Before and after: the new model holds up when the hand changes")
    fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "eval_before_after.png"), dpi=150); plt.close(fig)

    # 2. Cross-validation, augmenting inside each fold so no copy crosses folds
    print("2/7 cross-validation")
    cv_p, cv_y = dev_p[np.concatenate([tr, va])], dev_y[np.concatenate([tr, va])]
    scores = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(cv_p, cv_y):
        tp, ty = augment.augment(cv_p[a], cv_y[a])
        scores.append(accuracy_score(cv_y[b], svm().fit(batch_features(tp, 2), ty).predict(batch_features(cv_p[b], 2))))
    out["cross_validation"] = {"folds": [round(float(s), 4) for s in scores],
                               "mean": round(float(np.mean(scores)), 4), "std": round(float(np.std(scores)), 4)}

    # 3. Learning curve
    print("3/7 learning curve")
    rng = np.random.default_rng(SEED)
    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    lc = []
    for frac in fractions:
        sub = rng.permutation(tr)[: max(52, int(len(tr) * frac))]
        tp, ty = augment.augment(dev_p[sub], dev_y[sub])
        model = svm().fit(batch_features(tp, 2), ty)
        lc.append({"train_samples": int(len(sub)),
                   "train_accuracy": round(float(accuracy_score(dev_y[sub], model.predict(batch_features(dev_p[sub], 2)))), 4),
                   "val_accuracy": round(float(accuracy_score(dev_y[va], model.predict(batch_features(dev_p[va], 2)))), 4)})
    out["learning_curve"] = lc
    fig, ax = plt.subplots(figsize=(7, 3.6))
    n = [r["train_samples"] for r in lc]
    ax.plot(n, [r["train_accuracy"] * 100 for r in lc], "-o", color=AMBER, lw=2, ms=6, label="Training accuracy")
    ax.plot(n, [r["val_accuracy"] * 100 for r in lc], "-o", color=TEAL, lw=2, ms=6, label="Validation accuracy")
    ax.set_xlabel("Original training samples (before augmentation)"); ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax, "Learning curve: validation accuracy as training data grows")
    fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "eval_learning_curve.png"), dpi=150); plt.close(fig)

    # 4. Validation curve over C
    print("4/7 validation curve")
    tp, ty = augment.augment(dev_p[tr], dev_y[tr])
    f_tp, f_tr, f_va = batch_features(tp, 2), batch_features(dev_p[tr], 2), batch_features(dev_p[va], 2)
    vc = []
    for C in [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]:
        model = svm(C).fit(f_tp, ty)
        vc.append({"C": C, "train_accuracy": round(float(accuracy_score(dev_y[tr], model.predict(f_tr))), 4),
                   "val_accuracy": round(float(accuracy_score(dev_y[va], model.predict(f_va))), 4)})
    out["validation_curve"] = vc
    fig, ax = plt.subplots(figsize=(7, 3.6))
    cs = [r["C"] for r in vc]
    ax.semilogx(cs, [r["train_accuracy"] * 100 for r in vc], "-o", color=AMBER, lw=2, ms=6, label="Training accuracy")
    ax.semilogx(cs, [r["val_accuracy"] * 100 for r in vc], "-o", color=TEAL, lw=2, ms=6, label="Validation accuracy")
    ax.set_xlabel("C (log scale)"); ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax, "Validation curve: small C underfits, the gap grows as C increases")
    fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "eval_validation_curve.png"), dpi=150); plt.close(fig)

    # 5. PCA and t-SNE
    print("5/7 PCA and t-SNE")
    f_all = StandardScaler().fit_transform(batch_features(dev_p, 2))
    pca = PCA(n_components=2, random_state=SEED).fit(f_all)
    views = {"PCA": pca.transform(f_all),
             "t-SNE": TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(f_all)}
    out["pca_explained_variance"] = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, xy) in zip(axes, views.items()):
        grey = ~np.isin(dev_y, list(HIGHLIGHT))
        ax.scatter(xy[grey, 0], xy[grey, 1], s=6, color=GREY, linewidths=0)
        for letter, col in HIGHLIGHT.items():
            m = dev_y == letter
            ax.scatter(xy[m, 0], xy[m, 1], s=12, color=col, linewidths=0, label=letter)
        for letter in sorted(set(dev_y)):
            cx, cy = np.median(xy[dev_y == letter], axis=0)
            ax.text(cx, cy, letter, fontsize=9, fontweight="bold", ha="center", va="center", color=INK)
        lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
        pad = (hi - lo) * 0.08
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name + (f" (2 components explain {sum(pca.explained_variance_ratio_):.0%} of variance)" if name == "PCA" else ""),
                     loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=9, loc="upper right", title="Most confused", title_fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(REPORT_DIR, "eval_pca_tsne.png"), dpi=150); plt.close(fig)

    # 6. Speed
    print("6/7 speed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        prob_model = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced",
                                                         probability=True, random_state=SEED)).fit(f_tp, ty)
    one = dev_p[te][:1]
    t0 = time.perf_counter()
    for _ in range(500):
        batch_features(one, 2)
    feat_ms = (time.perf_counter() - t0) / 500 * 1000
    t0 = time.perf_counter()
    for _ in range(500):
        prob_model.predict_proba(batch_features(one, 2))
    total_ms = (time.perf_counter() - t0) / 500 * 1000
    out["speed_ms"] = {"features": round(feat_ms, 3), "features_and_classify": round(total_ms, 3),
                       "support_vectors": int(prob_model[-1].n_support_.sum())}

    # 7. New signers
    print("7/7 new signers")
    if is_test.any():
        model = svm().fit(f_tp, ty)
        pred = model.predict(batch_features(points[is_test], 2))
        per = {}
        for s in sorted(set(signers[is_test])):
            m = signers[is_test] == s
            per[s] = {"n": int(m.sum()), "accuracy": round(float(accuracy_score(labels[is_test][m], pred[m])), 4)}
        out["new_signers"] = {"accuracy": round(float(accuracy_score(labels[is_test], pred)), 4),
                              "macro_f1": round(float(f1_score(labels[is_test], pred, average="macro")), 4),
                              "per_signer": per}
    else:
        out["new_signers"] = None

    with open(os.path.join(REPORT_DIR, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    md = ["# Evaluation report", "",
          f"Split: {len(tr)} train / {len(va)} validation / {len(te)} test (stratified, seed {SEED}); "
          f"{out['split']['new_signer_test']} new-signer test samples. SVM with C = 10 throughout so setups are comparable.", "",
          "## 1. Before and after (test accuracy)", "",
          "| Setup | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for name, r in ablation.items():
        md.append(f"| {name} | " + " | ".join(f"{r[c]:.1%}" for c in conds) + " |")
    cv = out["cross_validation"]
    md += ["", "![Before and after](eval_before_after.png)", "",
           "## 2. Cross-validation", "",
           f"5-fold stratified CV on train + validation, augmenting inside each fold: "
           f"**{cv['mean']:.1%} ± {cv['std']:.1%}** (folds: " + ", ".join(f"{s:.1%}" for s in cv["folds"]) + ").", "",
           "## 3. Learning curve", "", "| Training samples | Training accuracy | Validation accuracy |", "|---|---|---|"]
    md += [f"| {r['train_samples']} | {r['train_accuracy']:.1%} | {r['val_accuracy']:.1%} |" for r in lc]
    md += ["", "![Learning curve](eval_learning_curve.png)", "",
           "## 4. Validation curve over C", "", "| C | Training accuracy | Validation accuracy |", "|---|---|---|"]
    md += [f"| {r['C']} | {r['train_accuracy']:.1%} | {r['val_accuracy']:.1%} |" for r in vc]
    sp = out["speed_ms"]
    md += ["", "![Validation curve](eval_validation_curve.png)", "",
           "## 5. PCA and t-SNE", "",
           f"The first two principal components explain {sum(out['pca_explained_variance']):.0%} of the variance. "
           "D, X and R are highlighted because they are the most confused letters.", "",
           "![PCA and t-SNE](eval_pca_tsne.png)", "",
           "## 6. Speed", "",
           f"Features: {sp['features']:.2f} ms per frame. Features + SVM probabilities: {sp['features_and_classify']:.2f} ms per frame "
           f"({sp['support_vectors']} support vectors).", "",
           "## 7. New signers", ""]
    if out["new_signers"]:
        ns = out["new_signers"]
        md += [f"Accuracy on people not in the training data: **{ns['accuracy']:.1%}** (macro F1 {ns['macro_f1']:.3f}).", "",
               "| Signer | Samples | Accuracy |", "|---|---|---|"]
        md += [f"| {s} | {v['n']} | {v['accuracy']:.1%} |" for s, v in ns["per_signer"].items()]
    else:
        md += ["No new-signer samples yet. Record them in the app with **Test set** ticked (see dataset/RECORDING_GUIDE.md)."]
    with open(os.path.join(REPORT_DIR, "evaluation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("Wrote reports/evaluation.md")


if __name__ == "__main__":
    main()
