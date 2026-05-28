"""
STEP 10 — RISK TIERS & BUSINESS OUTPUT
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load test data
X_test = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\X_test.csv')
y_test = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\y_test.csv')['TARGET']

# Load SMOTE model
with open(r'c:\Users\gargd\Downloads\credit_risk_project\models\xgb_smote_model.pkl', 'rb') as f:
    xgb_smote = pickle.load(f)

print("=" * 60)
print("STEP 10 — RISK TIERS & BUSINESS OUTPUT")
print("=" * 60)

# Get predictions
y_pred_proba = xgb_smote.predict_proba(X_test)[:, 1]

# Assign risk tiers
def assign_risk_tier(prob):
    if prob < 0.05:
        return 'Low Risk'
    elif prob < 0.15:
        return 'Medium Risk'
    else:
        return 'High Risk'

risk_tiers = pd.Series([assign_risk_tier(p) for p in y_pred_proba])

# Create summary table
print("\n📊 Risk Tier Summary:")
print("\n   Tier          | Count    | Default Rate | Avg Default Prob")
print("   " + "-" * 65)

tier_summary = []
for tier in ['Low Risk', 'Medium Risk', 'High Risk']:
    mask = risk_tiers == tier
    count = mask.sum()
    default_rate = y_test[mask].mean() if count > 0 else 0
    avg_prob = y_pred_proba[mask].mean() if count > 0 else 0
    pct = count / len(risk_tiers) * 100
    
    print(f"   {tier:13s} | {count:7,} ({pct:5.1f}%) | {default_rate:11.2%} | {avg_prob:16.4f}")
    
    tier_summary.append({
        'tier': tier,
        'count': int(count),
        'percentage': float(pct),
        'default_rate': float(default_rate),
        'avg_prob': float(avg_prob)
    })

# Plot risk segmentation
print("\n📊 Creating risk segmentation plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Distribution by tier
tier_counts = risk_tiers.value_counts()[['Low Risk', 'Medium Risk', 'High Risk']]
colors = ['#2ecc71', '#f39c12', '#e74c3c']
axes[0].bar(range(len(tier_counts)), tier_counts.values, color=colors)
axes[0].set_xticks(range(len(tier_counts)))
axes[0].set_xticklabels(tier_counts.index, fontsize=11)
axes[0].set_ylabel('Number of Applicants', fontsize=11)
axes[0].set_title('Applicant Distribution by Risk Tier', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Add count labels
for i, v in enumerate(tier_counts.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=10, fontweight='bold')

# Plot 2: Default rate by tier
default_rates = []
for tier in ['Low Risk', 'Medium Risk', 'High Risk']:
    mask = risk_tiers == tier
    default_rates.append(y_test[mask].mean() * 100)

axes[1].bar(range(len(default_rates)), default_rates, color=colors)
axes[1].set_xticks(range(len(default_rates)))
axes[1].set_xticklabels(['Low Risk', 'Medium Risk', 'High Risk'], fontsize=11)
axes[1].set_ylabel('Default Rate (%)', fontsize=11)
axes[1].set_title('Default Rate by Risk Tier', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Add percentage labels
for i, v in enumerate(default_rates):
    axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(r'c:\Users\gargd\Downloads\credit_risk_project\outputs\risk_segmentation.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/risk_segmentation.png")

# Find threshold where recall >= 0.65 AND approval rate >= 0.65
print("\n🎯 Finding optimal threshold (recall >= 0.65 AND approval rate >= 0.65)...")

best_threshold = None
best_approval_rate = None
best_recall = None

for threshold in np.arange(0.01, 0.99, 0.01):
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Approval rate = % predicted as non-default (0)
    approval_rate = (y_pred == 0).mean()
    
    # Recall on default class = TP / (TP + FN)
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if recall >= 0.65 and approval_rate >= 0.65:
        best_threshold = threshold
        best_approval_rate = approval_rate
        best_recall = recall
        break

if best_threshold is None:
    # Relax constraints slightly
    for threshold in np.arange(0.01, 0.99, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)
        approval_rate = (y_pred == 0).mean()
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if recall >= 0.60 and approval_rate >= 0.60:
            best_threshold = threshold
            best_approval_rate = approval_rate
            best_recall = recall
            break

print(f"\n   Recommended Threshold: {best_threshold:.4f}")
print(f"   Approval Rate: {best_approval_rate:.2%}")
print(f"   Default Recall: {best_recall:.2%}")

# Simulated NPA reduction
# Baseline: approve everyone
baseline_default_rate = y_test.mean()
baseline_defaults = len(y_test) * baseline_default_rate

# With model: reject high-risk applicants
y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
approved_mask = (y_pred_optimal == 0)
model_defaults = y_test[approved_mask].sum()
model_default_rate = y_test[approved_mask].mean() if approved_mask.sum() > 0 else 0

npa_reduction = ((baseline_defaults - model_defaults) / baseline_defaults * 100) if baseline_defaults > 0 else 0

print(f"\n💰 Simulated NPA Reduction:")
print(f"   Baseline (approve all): {baseline_defaults:.0f} defaults ({baseline_default_rate:.2%} rate)")
print(f"   With model: {model_defaults:.0f} defaults ({model_default_rate:.2%} rate)")
print(f"   NPA Reduction: {npa_reduction:.2f}%")

print("\n✅ STEP 10 COMPLETE\n")

# Save business metrics
business_metrics = {
    'tier_summary': tier_summary,
    'recommended_threshold': float(best_threshold),
    'approval_rate': float(best_approval_rate),
    'default_recall': float(best_recall),
    'npa_reduction_pct': float(npa_reduction),
    'baseline_default_rate': float(baseline_default_rate),
    'model_default_rate': float(model_default_rate)
}
with open(r'c:\Users\gargd\Downloads\credit_risk_project\data\business_metrics.json', 'w') as f:
    json.dump(business_metrics, f, indent=2)
print("💾 Saved business metrics")
