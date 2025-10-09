# Heart-Disease-Prediction
Project: Heart Disease Prediction
Problem: Predict the likelihood of a patient having heart disease based on clinical features.

Dataset: UCI Heart Disease Dataset (Cleveland dataset is the most commonly used).

ML Task: Binary Classification.

Key Features: Age, sex, chest pain type, resting blood pressure, cholesterol levels, etc.

End-to-End Steps:

EDA: Check for correlations (e.g., age vs. max heart rate).

Preprocessing: Handle a few missing values, scale numerical features.

Models: Start with Logistic Regression, then try Random Forest and XGBoost.

Evaluation: Focus on Recall (we want to minimize false negatives - missing a patient with heart disease).

Deployment: Build a Streamlit app with sliders and dropdowns for input.
