# Credit Risk Prediction

This notebook demonstrates a project for predicting loan default risk using a dataset of loan applicants. The key steps include data preprocessing, feature engineering, model training, and evaluating a Random Forest classifier.

---

## 1. Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
```

---

## 2. Load and Inspect Data
```python
# Load the data
application_df = pd.read_csv("application_record.csv")
credit_df = pd.read_csv("credit_record.csv")

# View shape and sample
print("Application Data Shape:", application_df.shape)
print("Credit Data Shape:", credit_df.shape)
application_df.head()
```

---

## 3. Examine and Clean Columns
```python
# Check missing values
application_df.isnull().sum()

# Drop columns with too many missing values (e.g., 'OCCUPATION_TYPE')
application_df.drop(columns=["OCCUPATION_TYPE"], inplace=True)
```

---

## 4. Create Target Variable
```python
# Flag default: 1 if STATUS has 2, 3, 4, or 5; 0 otherwise
credit_df['default'] = credit_df['STATUS'].isin(['2', '3', '4', '5']).astype(int)

# Aggregate to one default status per ID
status_df = credit_df.groupby("ID")["default"].max().reset_index()
```

---

## 5. Merge Data
```python
merged_df = pd.merge(application_df, status_df, on="ID", how="inner")
merged_df.shape
```

---

## 6. Feature Engineering
```python
# One-hot encode categorical features
cat_cols = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_EDUCATION_TYPE"]
merged_df = pd.get_dummies(merged_df, columns=cat_cols, drop_first=True)

# Drop ID column for modeling
model_df = merged_df.drop(columns=["ID"])
```

---

## 7. Train-Test Split
```python
X = model_df.drop("default", axis=1)
y = model_df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

---

## 8. Model Training
```python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
```

---

## 9. Model Evaluation
```python
y_pred = rf_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))
```

---

## 10. Predict on Full Data
```python
default_probs = rf_model.predict_proba(X)[:, 1]
X["default_probability"] = default_probs
X.head()
```

---

## 11. Save Predictions (Optional)
```python
X["ID"] = merged_df["ID"]  # If you want to keep ID
X[["ID", "default_probability"]].to_csv("default_probabilities.csv", index=False)
```

---

## ✅ Summary
- Built a binary classifier using application and credit history data
- Handled merging, cleaning, encoding, and imbalance
- Trained and evaluated a Random Forest model
- Predicted loan default probabilities

This notebook is now GitHub-ready.
