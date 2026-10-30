"""
Run a trained Keras MLP with plain NumPy.

TensorFlow is only needed to train (nn/train_mlp.py). The trained weights are
exported to a small .npz file, and this module repeats the forward pass:

    x  -> standardise:  z = (x - mean) / scale
       -> Dense + ReLU:  h1 = max(0, z  W1 + b1)
       -> Dense + ReLU:  h2 = max(0, h1 W2 + b2)
       -> Dense + softmax: p = softmax(h2 W3 + b3)

Dropout is only active during training, so it has no step here. This keeps the
server small enough for the free hosting tier (no 500 MB TensorFlow install).
"""

import json

import numpy as np


class NumpyMLP:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as f:
            self.mean = f["mean"]
            self.scale = f["scale"]
            n_layers = int(f["n_layers"])
            self.weights = [(f[f"W{i}"], f[f"b{i}"]) for i in range(n_layers)]
            self.classes_ = np.array([str(c) for c in f["classes"]])
            self.info = json.loads(str(f["info"]))

    @property
    def threshold(self):
        return float(self.info.get("threshold", 0.5))

    def predict_proba(self, x):
        z = (np.atleast_2d(np.asarray(x, dtype=np.float64)) - self.mean) / self.scale
        for i, (W, b) in enumerate(self.weights):
            z = z @ W + b
            if i < len(self.weights) - 1:
                z = np.maximum(z, 0.0)          # ReLU
        z = z - z.max(axis=1, keepdims=True)    # softmax, shifted for numerical stability
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]
