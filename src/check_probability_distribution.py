"""
Check Probability Distribution
"""
import pandas as pd
import numpy as np
import pickle

# Load test data
X_test = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\X_test.csv')
y_test = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\y_test.csv')['TARGET']

# Load SMOTE model
with open(r'c:\Users\gargd\Downloads\credit_risk_project\models\xgb_smote_model.pkl', 'rb') as f:
    xgb_smote = pickle.load(f)

# Get predictions
y_prob = xgb_smote.predict_proba(X_test)[:, 1]

print("=" * 60)
print("PROBABILITY DISTRIBUTION ANALYSIS")
print("=" * 60)

print("\n📊 Predicted Probability Distribution:")
print(pd.Series(y_prob).describe())

print("\n📊 Percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p:2d}th percentile: {np.percentile(y_prob, p):.4f}")

print("\n📊 Current (Broken) Tier Distribution:")
def assign_tier_old(prob):
    if prob < 0.05:
        return "Low Risk"
    elif prob < 0.15:
        return "Medium Risk"
    else:
        return "High Risk"

old_tiers = pd.Series([assign_tier_old(p) for p in y_prob])
print(old_tiers.value_counts(normalize=True).sort_index() * 100)

print("\n" + "=" * 60)
