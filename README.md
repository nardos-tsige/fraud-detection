# Fraud Detection System for E-commerce & Bank Transactions

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red.svg)](https://xgboost.ai)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-green.svg)](https://shap.readthedocs.io)

## Overview

Machine learning system to detect fraudulent transactions across e-commerce and bank credit card streams. Built with SMOTE for class imbalance and SHAP for model explainability.

## Key Results

| Metric | Score |
|--------|-------|
| Best Model | Random Forest |
| F1-Score | 0.87 |
| Precision | 0.85 |
| Recall | 0.89 |

## Repository Structure
fraud-detection/
├── notebooks/ # 5 Jupyter notebooks (EDA to SHAP)
├── models/ # Trained model artifacts
├── reports/ # SHAP visualizations
├── data/ # Raw and processed datasets
└── requirements.txt # Dependencies


## Notebooks

| Notebook | Description |
|----------|-------------|
| 01_eda_fraud_data.ipynb | E-commerce EDA |
| 02_eda_creditcard.ipynb | Credit card EDA |
| 03_feature_engineering.ipynb | Time, velocity, device features |
| 04_modeling.ipynb | Model training & comparison |
| 05_shap_explainability.ipynb | SHAP analysis & recommendations |

## Features Engineered

- Time since signup (hours)
- Transaction velocity (1h, 24h windows)
- Users per device
- Hour of day / day of week
- IP to country mapping

## Model Performance

| Model | F1 | Precision | Recall |
|-------|----|-----------|--------|
| Random Forest | 0.87 | 0.85 | 0.89 |
| XGBoost | 0.85 | 0.83 | 0.87 |
| Logistic Regression | 0.72 | 0.70 | 0.74 |

## SHAP Insights

Top 5 fraud drivers from SHAP analysis:
1. Time since signup
2. Purchase value
3. Users per device
4. Hour of day
5. New user flag

## Business Recommendations

- 1-hour cooling period for new accounts
- Device reputation scoring for fraud rings
- $500+ transaction review threshold
- Off-hour purchase verification (1 AM - 5 AM)
- Velocity limits (3 transactions per hour)

## Setup

```bash
git clone https://github.com/nardos-tsige/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
jupyter notebook

License
   MIT

Author
    Nardos Tsige
    Software Engineering Student, AAU