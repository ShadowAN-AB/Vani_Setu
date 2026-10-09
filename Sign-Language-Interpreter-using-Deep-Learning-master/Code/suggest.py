"""
Word prediction for Spell mode with a noisy-channel model.

The letters the app recognised are treated as a noisy copy of the letters the
signer meant. For every word w in the vocabulary that is at least as long as the
letters typed so far, Bayes' rule gives

    score(w) = log P(w)  +  sum_i log P(observed_i | intended = w_i)

  P(w)                      how common the word is (dataset/word_frequencies.csv)
  P(observed | intended)    how often the classifier reads one letter as another,
                            from its confusion matrix on held-out data (saved by
                            train_classifier.py inside the model file)

So if the app reads X-E-D, and R is often read as X, then RED scores higher than
words that really start with XED (there are none). Words that match exactly
still win, because the classifier is right most of the time.

Smoothing: P(o | c) = (1 - e) * (count(c -> o) + a) / (count(c) + 26a) + e / 26
  a = 0.1 keeps unseen confusions possible, e = 0.05 allows for live-camera errors
  that the test set never showed.
"""

import csv
import os
import string

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WORDS_CSV = os.path.join(APP_DIR, "dataset", "word_frequencies.csv")
LETTERS = string.ascii_uppercase
ALPHA, EPS = 0.1, 0.05


class Suggester:
    def __init__(self, words_csv=WORDS_CSV, confusion=None):
        words, freqs = [], []
        if os.path.exists(words_csv):
            with open(words_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    w = row["word"].strip().upper()
                    if w and all(c in LETTERS for c in w):
                        words.append(w)
                        freqs.append(float(row["frequency"]))
        self.words = np.array(words)
        self.log_prior = np.log(np.array(freqs) / max(sum(freqs), 1e-12)) if freqs else np.zeros(0)
        lengths = np.array([len(w) for w in words]) if words else np.zeros(0, dtype=int)
        self.lengths = lengths
        width = int(lengths.max()) if len(lengths) else 1
        # letters of each word as indices 0-25, padded with -1
        self.codes = np.full((len(words), width), -1, dtype=np.int16)
        for i, w in enumerate(words):
            self.codes[i, :len(w)] = [LETTERS.index(c) for c in w]
        self.set_confusion(confusion)

    def set_confusion(self, confusion):
        """confusion: {"classes": [...], "counts": [[...]]}, rows = actual letter, columns = predicted."""
        counts = np.zeros((26, 26))
        if confusion:
            classes = list(confusion.get("classes", []))
            raw = np.asarray(confusion.get("counts", []), dtype=float)
            for i, a in enumerate(classes):
                for j, p in enumerate(classes):
                    if a in LETTERS and p in LETTERS and raw.size:
                        counts[LETTERS.index(a), LETTERS.index(p)] = raw[i, j]
        if not counts.sum():
            counts = np.eye(26) * 20  # no confusion data: assume the classifier is usually right
        channel = (counts + ALPHA) / (counts.sum(1, keepdims=True) + 26 * ALPHA)
        channel = (1 - EPS) * channel + EPS / 26
        # log P(observed | intended), indexed [intended, observed]
        self.log_channel = np.log(channel)

    def suggest(self, letters, k=4):
        typed = "".join(c for c in str(letters).upper() if c in LETTERS)
        n = len(typed)
        if n < 2 or not len(self.words):
            return []
        cand = np.where(self.lengths >= n)[0]
        obs = np.array([LETTERS.index(c) for c in typed])
        intended = self.codes[cand, :n].astype(int)
        score = self.log_prior[cand] + self.log_channel[intended, obs[None, :]].sum(1)
        out = []
        for i in np.argsort(-score):
            w = str(self.words[cand[i]])
            if w == typed:
                continue
            out.append({"word": w, "corrected": not w.startswith(typed), "score": round(float(score[i]), 2)})
            if len(out) == k:
                break
        return out
