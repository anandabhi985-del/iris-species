# 🌸 Iris Species Classification ML Pipeline

An end-to-end Machine Learning classification pipeline and deployment project built on the classic Iris dataset using Python, `scikit-learn`, `joblib`, and `Streamlit`.

---

## 📌 Project Overview

This repository demonstrates a complete, production-grade Machine Learning classification lifecycle:
1. **Data Collection**: Fetching sample data via `sklearn.datasets.load_iris`.
2. **Data Cleaning**: Checking for duplicates, null values, and verifying class balance ($50$ samples per species).
3. **Data Preprocessing**: Implementing stratified train-test splits (`stratify=y`) to maintain target class balance across splits, followed by `StandardScaler` feature scaling.
4. **Feature Engineering**: Creating domain-specific botanical ratio features (`petal_ratio`, `sepal_ratio`, `petal_area`).
5. **Model Benchmarking**: Training and comparing Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Random Forest Classifiers via Stratified 5-Fold Cross-Validation.
6. **Model Evaluation**: Assessing model performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
7. **Hyperparameter Tuning**: Optimizing SVM hyperparameter combinations ($C$, $\gamma$, kernel) using `GridSearchCV`.
8. **Deployment**: Serializing compressed model artifacts (`joblib`) and serving an interactive web interface.
9. **Monitoring & Drift**: Implementing prediction distribution monitoring and accuracy performance triggers.

---

## 📊 Model Evaluation & Results

| Model | CV Mean Accuracy | Test Accuracy | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | ~0.958 | ~0.967 | ~0.970 | ~0.967 |
| **K-Nearest Neighbors (KNN)** | ~0.950 | ~0.967 | ~0.970 | ~0.967 |
| **Random Forest Classifier** | ~0.958 | ~0.967 | ~0.970 | ~0.967 |
| **Support Vector Machine (Tuned)** | **~0.975** | **~1.000** | **1.000** | **1.000** |

> **Key Result:** The **Tuned SVM Classifier** ($C=10, \gamma=\text{'scale'}$) achieved **100% test accuracy** on the holdout dataset, cleanly separating all three species (*Setosa*, *Versicolor*, and *Virginica*).

---

## 🛠️ Project Structure

```text
├── iris.py                     # Streamlit web application for real-time inference
├── iris_svm_model.pkl         # Serialized lightweight SVM model (compressed)
├── iris_scaler.pkl            # Serialized StandardScaler object (compressed)
├── README.md                  # Project documentation & summary
└── requirements.txt           # Required Python libraries
