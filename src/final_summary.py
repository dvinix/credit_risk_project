"""
FINAL PRINT — PROJECT SUMMARY
"""
import json

# Load final metrics
with open(r'c:\Users\gargd\Downloads\credit_risk_project\models\metrics.json', 'r') as f:
    metrics = json.load(f)

print("=" * 60)
print("CREDIT RISK PROJECT — FINAL RESULTS")
print("=" * 60)

print(f"\nDataset        : Home Credit (application_train.csv)")
print(f"Records        : {metrics['total_records']:,}")
print(f"Features used  : {metrics['features_used']}")
print(f"Engineered     : {metrics['engineered_features']} new features")

print(f"\nBaseline LR ROC-AUC   : {metrics['baseline_lr_roc_auc']:.4f}")
print(f"XGBoost ROC-AUC       : {metrics['xgboost_roc_auc']:.4f}")
print(f"XGBoost + SMOTE AUC   : {metrics['xgboost_smote_roc_auc']:.4f}")

print(f"\nDefault Recall (before SMOTE) : {metrics['default_recall_before_smote']*100:.2f}%")
print(f"Default Recall (after SMOTE)  : {metrics['default_recall_after_smote']*100:.2f}%")

print(f"\nTop 5 Default Drivers:")
for i, feature in enumerate(metrics['top_5_features'], 1):
    print(f"{i}. {feature}")

print(f"\nRisk Tiers:")
for tier_info in metrics['tier_summary']:
    print(f"{tier_info['tier']:12s} : {tier_info['percentage']:.1f}% of applicants")

print(f"\nRecommended Threshold : {metrics['recommended_threshold']:.4f}")
print(f"Simulated NPA Reduction : {metrics['npa_reduction_pct']:.2f}%")

print("=" * 60)
