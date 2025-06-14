# Credit Risk Prediction Model

A machine learning project that predicts whether customers will default on their credit based on their application data and credit history.

## What This Does

This notebook helps banks and financial institutions identify customers who might have trouble paying back their loans. It looks at customer information like income, age, employment status, and past credit behavior to predict the risk of default.

## The Data

You'll need two CSV files to run this:
- `application_record.csv` - Customer application information (income, age, job, etc.)
- `credit_record.csv` - Credit history showing payment patterns

The model combines these datasets to create a complete picture of each customer's risk profile.

## How It Works

1. **Loads the data** from both CSV files
2. **Creates a target variable** by identifying customers who had late payments (status codes 1-5 mean trouble)
3. **Cleans and prepares the data** by handling missing values and converting text to numbers
4. **Balances the dataset** using SMOTE since most customers don't default (imbalanced data problem)
5. **Trains three different models**: Logistic Regression, Random Forest, and XGBoost
6. **Picks the best model** based on performance metrics
7. **Generates risk scores** for all customers

## What You Need to Install

```bash
pip install pandas scikit-learn xgboost imbalanced-learn matplotlib
```

## Running the Notebook

1. Make sure you have the two CSV files in the same folder as the notebook
2. Update the file paths in the data loading section if needed
3. Run all cells from top to bottom
4. The notebook will show you model performance and generate risk scores

## What You Get

- **Model performance reports** showing how accurate each model is
- **Feature importance analysis** telling you which customer characteristics matter most
- **Risk scores** for every customer (0 = low risk, 1 = high risk)
- **Top risky customers list** so you can take action

## Key Results

The notebook typically shows that Random Forest performs best for this type of data. You'll see metrics like:
- ROC-AUC score (higher is better, 0.5 is random guessing)
- Precision and recall for catching defaults
- Feature importance showing what drives risk

## Business Use

Use the risk scores to:
- Approve or reject loan applications
- Set interest rates based on risk level
- Monitor existing customers for early warning signs
- Focus collection efforts on high-risk accounts

## Notes

- The model assumes customers not in the credit history file have no defaults
- SMOTE is used to balance the training data since defaults are rare
- All three models are compared but Random Forest usually wins
- Risk scores are probabilities between 0 and 1

This is a complete end-to-end credit risk solution that you can adapt for your specific needs.