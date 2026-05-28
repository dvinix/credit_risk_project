"""
STEP 6 — MAIN MODEL (XGBoost)
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import json

# Load train/test data
X_train = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\X_train.csv')
X_test = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\X_test.csv')
y_train = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\y_train.csv')['TARGET']
y_test = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\y_test.csv')['TARGET']

print("=" * 60)
print("STEP 6 — MAIN MODEL (XGBoost)")
print("=" * 60)

# Calculate scale_pos_weight
count_0 = (y_train == 0).sum()
count_1 = (y_train == 1).sum()
scale_pos_weight = count_0 / count_1

print(f"\n⚖️  Class Balance:")
print(f"   Non-Default (0): {count_0:,}")
print(f"   Default (1):     {count_1:,}")
print(f"   scale_pos_weight: {scale_pos_weight:.4f}")

# Train XGBoost
print("\n🔧 Training XGBoost Classifier...")
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    random_state=42,
    verbosity=0
)

xgb_model.fit(X_train, y_train)

# Predictions
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred = xgb_model.predict(X_test)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n📊 Test ROC-AUC: {roc_auc:.4f}")

# Classification Report
print("\n📋 Classification Report (default threshold):")
print(classification_report(y_test, y_pred, target_names=['Non-Default', 'Default']))

print("\n✅ STEP 6 COMPLETE\n")

# Save model and metrics
import pickle
with open(r'c:\Users\gargd\Downloads\credit_risk_project\models\xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

xgb_metrics = {
    'roc_auc': roc_auc,
    'scale_pos_weight': scale_pos_weight
}
with open(r'c:\Users\gargd\Downloads\credit_risk_project\data\xgb_metrics.json', 'w') as f:
    json.dump(xgb_metrics, f, indent=2)

print("💾 Saved XGBoost model and metrics")
