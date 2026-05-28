"""
STEP 11 — SAVE EVERYTHING & FINAL SUMMARY
"""
import pandas as pd
import numpy as np
import pickle
import json
import shutil

print("=" * 60)
print("STEP 11 — SAVE EVERYTHING")
print("=" * 60)

# Load all metrics
with open(r'c:\Users\gargd\Downloads\credit_risk_project\data\baseline_metrics.json', 'r') as f:
    baseline_metrics = json.load(f)

with open(r'c:\Users\gargd\Downloads\credit_risk_project\data\xgb_metrics.json', 'r') as f:
    xgb_metrics = json.load(f)

with open(r'c:\Users\gargd\Downloads\credit_risk_project\data\smote_metrics.json', 'r') as f:
    smote_metrics = json.load(f)

with open(r'c:\Users\gargd\Downloads\credit_risk_project\data\threshold_metrics.json', 'r') as f:
    threshold_metrics = json.load(f)

with open(r'c:\Users\gargd\Downloads\credit_risk_project\data\shap_features.json', 'r') as f:
    shap_features = json.load(f)

with open(r'c:\Users\gargd\Downloads\credit_risk_project\data\business_metrics.json', 'r') as f:
    business_metrics = json.load(f)

# Load final model
with open(r'c:\Users\gargd\Downloads\credit_risk_project\models\xgb_smote_model.pkl', 'rb') as f:
    final_model = pickle.load(f)

# Copy final model
print("\n💾 Saving final model...")
shutil.copy(
    r'c:\Users\gargd\Downloads\credit_risk_project\models\xgb_smote_model.pkl',
    r'c:\Users\gargd\Downloads\credit_risk_project\models\xgb_final.pkl'
)
print("   ✅ Saved: models/xgb_final.pkl")

# Compile all metrics
final_metrics = {
    'dataset': 'Home Credit (application_train.csv)',
    'total_records': 307511,
    'features_used': 70,
    'engineered_features': 10,
    'baseline_lr_roc_auc': baseline_metrics['roc_auc'],
    'xgboost_roc_auc': xgb_metrics['roc_auc'],
    'xgboost_smote_roc_auc': smote_metrics['roc_auc'],
    'default_recall_before_smote': smote_metrics['recall_before'],
    'default_recall_after_smote': smote_metrics['recall_after'],
    'best_threshold': threshold_metrics['best_threshold'],
    'top_5_features': shap_features['top_5_features'],
    'default_rate': business_metrics['baseline_default_rate'],
    'recommended_threshold': business_metrics['recommended_threshold'],
    'approval_rate': business_metrics['approval_rate'],
    'npa_reduction_pct': business_metrics['npa_reduction_pct'],
    'tier_summary': business_metrics['tier_summary']
}

# Save final metrics
with open(r'c:\Users\gargd\Downloads\credit_risk_project\models\metrics.json', 'w') as f:
    json.dump(final_metrics, f, indent=2)
print("   ✅ Saved: models/metrics.json")

print("\n📊 All plots saved to outputs/ folder:")
plots = [
    'precision_recall_curve.png',
    'confusion_matrix.png',
    'shap_beeswarm.png',
    'shap_waterfall.png',
    'shap_feature_importance.png',
    'risk_segmentation.png'
]
for plot in plots:
    print(f"   ✅ {plot}")

print("\n✅ STEP 11 COMPLETE\n")

# Print final summary table
print("=" * 70)
print("FINAL METRICS SUMMARY TABLE")
print("=" * 70)
print(f"\n{'Metric':<40} {'Value':>25}")
print("-" * 70)
print(f"{'Dataset':<40} {'Home Credit':>25}")
print(f"{'Total Records':<40} {307511:>25,}")
print(f"{'Features Used':<40} {70:>25}")
print(f"{'Engineered Features':<40} {10:>25}")
print(f"{'Default Rate':<40} {final_metrics['default_rate']:>24.2%}")
print("-" * 70)
print(f"{'Baseline LR ROC-AUC':<40} {final_metrics['baseline_lr_roc_auc']:>25.4f}")
print(f"{'XGBoost ROC-AUC':<40} {final_metrics['xgboost_roc_auc']:>25.4f}")
print(f"{'XGBoost + SMOTE ROC-AUC':<40} {final_metrics['xgboost_smote_roc_auc']:>25.4f}")
print("-" * 70)
print(f"{'Default Recall (before SMOTE)':<40} {final_metrics['default_recall_before_smote']:>24.2%}")
print(f"{'Default Recall (after SMOTE)':<40} {final_metrics['default_recall_after_smote']:>24.2%}")
print("-" * 70)
print(f"{'Recommended Threshold':<40} {final_metrics['recommended_threshold']:>25.4f}")
print(f"{'Approval Rate':<40} {final_metrics['approval_rate']:>24.2%}")
print(f"{'Simulated NPA Reduction':<40} {final_metrics['npa_reduction_pct']:>24.2f}%")
print("=" * 70)
