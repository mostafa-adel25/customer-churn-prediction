 Customer Churn Prediction — End-to-End ML Pipeline

A complete machine learning pipeline for predicting customer churn, covering everything from raw data preprocessing to a tuned ensemble model.
📌 Project Overview
This project builds a binary classification system to identify customers likely to churn. The pipeline handles real-world data challenges including missing values, categorical encoding, class imbalance, and feature engineering — then benchmarks 5 models before combining the best ones into a voting ensemble.

 🗂️ Dataset

| Split | Records |
|-------|---------|
| Train | ~15,000 |
| Test  | ~5,000  |

**Target:** `Churn` (0 = No Churn, 1 = Churn)
## ⚙️ Pipeline Steps

1. **Data Loading** — `train.csv` / `test.csv`
2. **Missing Value Imputation** — Median strategy for numeric columns
3. **Encoding** — `OrdinalEncoder` with `handle_unknown` for safe test-set inference
4. **Feature Engineering** — 3 new interaction features
5. **Scaling** — `StandardScaler`
6. **SMOTE** — Synthetic oversampling to balance the training set
7. **Model Training** — 5 classifiers benchmarked
8. **Hyperparameter Tuning** — `GridSearchCV` (5-fold, AUC-ROC scoring)
9. **Voting Ensemble** — Soft vote across top 3 models
10. **Threshold Tuning** — Custom decision threshold optimised for Recall

## 🧪 Feature Engineering

| Feature | Formula | Signal |
|---|---|---|
| `spend_per_call` | `Total Spend / (Support Calls + 1)` | High spend + few calls = loyal |
| `delay_x_calls` | `Payment Delay × Support Calls` | Combined friction — top churn driver |
| `recency_spend` | `Last Interaction × Total Spend` | Recent high-value = less likely to churn |

---

## 📊 Model Comparison

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| ⭐ Logistic Regression | 0.6012 | 0.6986 | **0.7768** |
| XGBoost | 0.5217 | 0.6641 | 0.7090 |
| LightGBM | 0.5215 | 0.6640 | 0.6783 |
| Random Forest | 0.5215 | 0.6640 | 0.5546 |
| Decision Tree | 0.5185 | 0.6626 | 0.5425 |

> **Note:** Low accuracy in tree models is expected — SMOTE creates a balanced 50/50 training set, but the test set keeps the original imbalanced distribution. **AUC-ROC is the correct metric to evaluate here.**

---

## 🏆 Best Model Results

```
Best Model : Logistic Regression  
Accuracy   : 0.6012  
F1 Score   : 0.6986  
AUC-ROC    : 0.7768  
```

---

## 🛠️ Tech Stack

```
Python · Scikit-learn · XGBoost · LightGBM · Imbalanced-learn · Pandas · NumPy · Matplotlib · Seaborn
```

## 📁 Project Structure

```
├── train.csv
├── test.csv
├── churn_project.ipynb
└── README.md
```
## 🚀 How to Run


حطه مباشرةً في ملف `README.md` في الـ repo وهيطلع احترافي جداً ✅
