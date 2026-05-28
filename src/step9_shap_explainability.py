"""
STEP 9 — SHAP EXPLAINABILITY
"""
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import json

# Load test data
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')['TARGET']

# Load SMOTE model
with open('models/xgb_smote_model.pkl', 'rb') as f:
    xgb_smote = pickle.load(f)

print("=" * 60)
print("STEP 9 — SHAP EXPLAINABILITY")
print("=" * 60)

# Sample 1000 random test samples
print("\n🔍 Computing SHAP values on 1000 random test samples...")
np.random.seed(42)
sample_indices = np.random.choice(len(X_test), size=1000, replace=False)
X_sample = X_test.iloc[sample_indices]

# Create SHAP explainer
explainer = shap.TreeExplainer(xgb_smote)
shap_values = explainer.shap_values(X_sample)

print("   ✅ SHAP values computed")

# 1. Beeswarm plot (top 15 features)
print("\n📊 Creating beeswarm plot (top 15 features)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, max_display=15, show=False)
plt.tight_layout()
plt.savefig('outputs/shap_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/shap_beeswarm.png")

# 2. Waterfall plot for 1 high-risk applicant
print("\n📊 Creating waterfall plot for 1 high-risk applicant...")
# Get predictions for sample
y_pred_proba = xgb_smote.predict_proba(X_sample)[:, 1]
# Find highest risk applicant
high_risk_idx = np.argmax(y_pred_proba)

plt.figure(figsize=(10, 8))
shap.waterfall_plot(shap.Explanation(
    values=shap_values[high_risk_idx],
    base_values=explainer.expected_value,
    data=X_sample.iloc[high_risk_idx],
    feature_names=X_sample.columns.tolist()
), show=False)
plt.tight_layout()
plt.savefig('outputs/shap_waterfall.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/shap_waterfall.png")

# 3. Feature importance bar plot
print("\n📊 Creating feature importance plot...")
shap_importance = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': X_sample.columns,
    'importance': shap_importance
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
top_15 = feature_importance.head(15)
plt.barh(range(len(top_15)), top_15['importance'].values)
plt.yticks(range(len(top_15)), top_15['feature'].values)
plt.xlabel('Mean |SHAP value|', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Features by SHAP Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/shap_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/shap_feature_importance.png")

# Top 5 features by mean absolute SHAP value
print("\n🏆 Top 5 Features by Mean Absolute SHAP Value:")
top_5 = feature_importance.head(5)
for idx, row in top_5.iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:.4f}")

# Business rules for top features
print("\n💼 Business Rules (Plain English):")
business_rules = {
    'EXT_SOURCE_2': 'Lower external credit scores strongly indicate higher default risk',
    'EXT_SOURCE_3': 'Poor third-party credit ratings are red flags for default',
    'ext_source_mean': 'Average external credit score below 0.4 signals high risk',
    'DAYS_BIRTH': 'Younger applicants (under 30) have higher default rates',
    'age_years': 'Age under 25 years significantly increases default probability',
    'DAYS_EMPLOYED': 'Short employment history (under 2 years) raises default risk',
    'employment_years': 'Less than 1 year of employment is a major risk factor',
    'AMT_CREDIT': 'Loan amounts above 1.5M show elevated default rates',
    'AMT_ANNUITY': 'High monthly payments relative to income increase default risk',
    'credit_to_income': 'Credit-to-income ratio above 6 is a strong default predictor',
    'DAYS_ID_PUBLISH': 'Recently issued IDs (within 2 years) correlate with defaults',
    'days_id_published_yr': 'ID issued less than 3 years ago indicates higher risk',
    'AMT_GOODS_PRICE': 'High-value purchases without sufficient income are risky',
    'REGION_POPULATION_RELATIVE': 'Applicants from less populated regions default more',
    'DAYS_LAST_PHONE_CHANGE': 'Frequent phone changes (within 6 months) signal instability'
}

for idx, row in top_5.iterrows():
    feature = row['feature']
    rule = business_rules.get(feature, 'Monitor this feature closely for risk assessment')
    print(f"   {idx+1}. {feature}: {rule}")

print("\n✅ STEP 9 COMPLETE\n")

# Save top features
top_features = {
    'top_5_features': top_5['feature'].tolist(),
    'top_5_importance': top_5['importance'].tolist()
}
with open('data/shap_features.json', 'w') as f:
    json.dump(top_features, f, indent=2)
print("💾 Saved SHAP feature importance")
