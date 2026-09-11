# Model comparison

Dataset: **2122 samples**, **26 classes** (1591 train / 531 test, stratified split, seed 42).

| Model | Test accuracy | Macro F1 | 5-fold CV | Fit time |
|-------|---------------|----------|-----------|----------|
| SVM (RBF) | 95.1% | 0.953 | 94.1% ± 0.5% | 0.05s |
| Logistic Regression | 94.0% | 0.940 | 92.5% ± 0.6% | 0.05s |
| Random Forest | 88.1% | 0.879 | 87.9% ± 2.0% | 0.17s |
| K-Nearest Neighbours | 85.7% | 0.853 | 84.9% ± 1.2% | 0.0s |
| Decision Tree | 74.6% | 0.739 | 74.8% ± 2.1% | 0.05s |

**Chosen model:** SVM (RBF) (95.1% test accuracy, 94.1% cross-validated).

Notes:

- All models use the same 63-D wrist-relative landmark features.
- Scaled models are wrapped in a `StandardScaler` pipeline.
- Cross-validation is 5-fold over the full dataset.
