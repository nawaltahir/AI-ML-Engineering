# AI-ML-Engineering
# Task 1: Telco Customer Churn Prediction – End-to-End ML Pipeline

## Objective
Build a **production-ready machine learning pipeline** to predict whether a telecom customer will churn (i.e., stop using the service) using the **Telco Customer Churn** dataset from Kaggle.

Key goals:
- Clean and preprocess customer data
- Train and tune classification models
- Package the entire pipeline using `scikit-learn` and export it for future use

---

## Methodology

1. **Data Source**
   - Dataset: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   - Target column: `Churn` (Yes = churned, No = stayed)

2. **Preprocessing**
   - Dropped `customerID` column
   - Converted `TotalCharges` to numeric and imputed missing values
   - Applied:
     - `StandardScaler` to numeric features
     - `OneHotEncoder` to categorical features
   - Used `ColumnTransformer` to apply transformations within a unified pipeline

3. **Model Pipelines**
   - Built two complete pipelines using `Pipeline`:
     - Logistic Regression
     - Random Forest Classifier
   - Each pipeline includes preprocessing + classifier

4. **Hyperparameter Tuning**
   - Used `GridSearchCV` with 5-fold cross-validation
   - Tuned key hyperparameters for each model:
     - Logistic Regression: `C`, `solver`
     - Random Forest: `n_estimators`, `max_depth`

5. **Model Evaluation**
   - Evaluated using `classification_report` on a test split
   - Compared precision, recall, F1-score, and accuracy

6. **Model Export**
   - Best models saved using `joblib` as `.pkl` files for future deployment

---
<img width="969" height="685" alt="image" src="https://github.com/user-attachments/assets/b6563a54-5ddd-402f-a124-8d8ac9695e23" />
