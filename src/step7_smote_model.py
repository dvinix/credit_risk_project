"""
STEP 7 — HANDLE CLASS IMBALANCE (SMOTE)
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import json

# Load train/test data
X_train = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\X_train.csv')
X_test = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\X_test.csv')
y_train = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\y_train.csv')['TARGET']
y_test = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\y_test.csv')['TARGET']

# Load previous metrics
with open(r'c:\Users\gargd\Downloads\credit_risk_project\data\xgb_metrics.json', 'r') as f:
    prev_metrics = json.load(f)
scale_pos_weight = prev_metrics['scale_pos_weight']

print("=" * 60)
print("STEP 7 — HANDLE CLASS IMBALANCE (SMOTE)")
print("=" * 60)

# Apply SMOTE
print("\n🔄 Applying SMOTE (sampling_strategy=0.3)...")
smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\n📊 Before SMOTE:")
print(f"   Train shape: {X_train.shape}")
print(f"   Class 0: {(y_train == 0).sum():,}")
print(f"   Class 1: {(y_train == 1).sum():,}")

print(f"\n📊 After SMOTE:")
print(f"   Train shape: {X_train_smote.shape}")
print(f"   Class 0: {(y_train_smote == 0).sum():,}")
print(f"   Class 1: {(y_train_smote == 1).sum():,}")

# Train XGBoost on SMOTE data
print("\n🔧 Training XGBoost on SMOTE data...")
xgb_smote = XGBClassifier(
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

xgb_smote.fit(X_train_smote, y_train_smote)

# Predictions
y_pred_proba = xgb_smote.predict_proba(X_test)[:, 1]
y_pred = xgb_smote.predict(X_test)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n📊 Test ROC-AUC: {roc_auc:.4f}")

# Classification Report
print("\n📋 Classification Report:")
report = classification_report(y_test, y_pred, target_names=['Non-Default', 'Default'], output_dict=True)
print(classification_report(y_test, y_pred, target_names=['Non-Default', 'Default']))

# Compare recall improvement
recall_before = 0.66  # From step 6
recall_after = report['Default']['recall']

print(f"\n📈 Recall on default class improved from {recall_before:.2f} to {recall_after:.2f}")

print("\n✅ STEP 7 COMPLETE\n")

# Save SMOTE model and metrics
import pickle
with open(r'c:\Users\gargd\Downloads\credit_risk_project\models\xgb_smote_model.pkl', 'wb') as f:
    pickle.dump(xgb_smote, f)

smote_metrics = {
    'roc_auc': roc_auc,
    'recall_before': recall_before,
    'recall_after': recall_after
}
with open(r'c:\Users\gargd\Downloads\credit_risk_project\data\smote_metrics.json', 'w') as f:
    json.dump(smote_metrics, f, indent=2)

print("💾 Saved SMOTE model and metrics")
