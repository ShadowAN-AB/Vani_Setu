# Model comparison

Same data and split for every model: 1484 train / 319 validation / 319 test (stratified, seed 42). Features: version 2 (88 numbers). Training data augmented to 8904 samples.

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 | Fit time | Time per prediction |
|-------|---------------------|---------------------|---------------|---------------|----------|---------------------|
| SVM (RBF) | 96.9% | 0.971 | 97.2% | 0.975 | 0.76s | 0.654 ms |
| Logistic Regression | 95.6% | 0.959 | 95.9% | 0.965 | 2.01s | 0.711 ms |
| Random Forest | 95.6% | 0.959 | 96.2% | 0.963 | 6.18s | 51.155 ms |
| MLP neural network | 95.9% | 0.958 | 96.2% | 0.963 | 4.02s | 0.453 ms |
| K-Nearest Neighbours | 94.4% | 0.945 | 95.0% | 0.954 | 0.01s | 0.979 ms |
| Decision Tree | 90.9% | 0.916 | 90.9% | 0.915 | 0.83s | 0.127 ms |

**Chosen by validation macro F1:** SVM (RBF).

The live app uses the SVM after tuning C and gamma in `train_classifier.py`; see `reports/metrics.json`.
