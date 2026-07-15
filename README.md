# Breast Cancer Detection Pipeline

An end-to-end, production-ready machine learning pipeline that classifies breast tumors as Benign or Malignant. 

This project demonstrates how to handle critical medical datasets by actively prioritizing **Recall** to minimize false negatives, utilizing `scikit-learn` Pipelines to prevent data leakage, and implementing hyperparameter tuning via `GridSearchCV`.

## Project Objective & Clinical Context
In medical diagnostics, failing to detect a malignant tumor (a False Negative) carries a drastically higher cost than a false alarm (a False Positive). 

To reflect this real-world constraint, this pipeline is strictly optimized for **Recall** on the malignant class. The model actively up-weights the minority class during training to ensure high sensitivity to disease-positive cases.

## Tech Stack & Architecture
* **Language:** Python 3.9+
* **Libraries:** `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`
* **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic) Dataset
* **Model:** XGBoost Classifier (`XGBClassifier`)

### Pipeline Features:
1. **Leak-Proof Preprocessing:** Standardization (`StandardScaler`) is strictly bound within a `scikit-learn` `Pipeline`, ensuring that scaling metrics are fitted *only* on the training folds during Cross-Validation.
2. **Dynamic Class Balancing:** Automatically calculates and applies scaling weights (`scale_pos_weight`) to handle imbalanced datasets.
3. **Hyperparameter Tuning:** Utilizes `GridSearchCV` with Stratified K-Fold cross-validation.
4. **Standardized Logging:** Replaces standard print statements with a configured `logging` module for production-readiness.

## 🚀 How to Run Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/Vandahlen/breast-cancer-detection.git](https://github.com/Vandahlen/breast-cancer-detection.git)
cd breast-cancer-detection
