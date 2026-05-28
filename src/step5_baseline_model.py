"""
STEP 5 — BASELINE MODEL (Logistic Regression)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

# Load train/test data
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv')['TARGET']
y_test = pd.read_csv('data/y_test.csv')['TARGET']

print("=" * 60)
print("STEP 5 — BASELINE MODEL (Logistic Regression)")
print("=" * 60)

# Create pipeline with StandardScaler and LogisticRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

print("\n🔧 Training Logistic Regression with StandardScaler...")
pipeline.fit(X_train, y_train)

# Predictions
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n📊 Test ROC-AUC: {roc_auc:.4f}")

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Default', 'Default']))

print("\n✅ STEP 5 COMPLETE\n")

# Save baseline metrics
baseline_metrics = {
    'roc_auc': roc_auc
}
import json
with open('data/baseline_metrics.json', 'w') as f:
    json.dump(baseline_metrics, f, indent=2)
print("💾 Saved baseline metrics")
