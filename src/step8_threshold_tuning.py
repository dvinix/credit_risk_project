"""
STEP 8 — THRESHOLD TUNING
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load test data
X_test = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\X_test.csv')
y_test = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\y_test.csv')['TARGET']

# Load SMOTE model
with open(r'c:\Users\gargd\Downloads\credit_risk_project\models\xgb_smote_model.pkl', 'rb') as f:
    xgb_smote = pickle.load(f)

print("=" * 60)
print("STEP 8 — THRESHOLD TUNING")
print("=" * 60)

# Get predictions
y_pred_proba = xgb_smote.predict_proba(X_test)[:, 1]

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Plot Precision-Recall curve
print("\n📊 Plotting Precision-Recall curve...")
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, linewidth=2, label='PR Curve')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(r'c:\Users\gargd\Downloads\credit_risk_project\outputs\precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: outputs/precision_recall_curve.png")

# Find threshold where recall on default class >= 0.70
# Default class is 1, so we need recall for class 1
target_recall = 0.70

# Find the threshold - search from high to low threshold (low to high recall)
best_threshold = None
for i in range(len(thresholds)-1, -1, -1):
    if recall[i] >= target_recall and recall[i] <= 0.75:  # Find first one in reasonable range
        best_threshold = float(thresholds[i])
        best_precision = float(precision[i])
        best_recall = float(recall[i])
        break

if best_threshold is None:
    # Find closest to target recall
    idx = np.argmin(np.abs(recall[:-1] - target_recall))
    best_threshold = float(thresholds[idx])
    best_precision = float(precision[idx])
    best_recall = float(recall[idx])

print(f"\n🎯 Threshold where recall >= 0.70:")
print(f"   Threshold: {best_threshold:.4f}")
print(f"   Recall: {best_recall:.4f}")
print(f"   Precision: {best_precision:.4f}")

# Apply threshold
y_pred_tuned = (y_pred_proba >= best_threshold).astype(int)

# Classification Report
print(f"\n📋 Classification Report at threshold {best_threshold:.4f}:")
print(classification_report(y_test, y_pred_tuned, target_names=['Non-Default', 'Default']))

# Confusion Matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_tuned)
print(cm)
print(f"\n   True Negatives:  {cm[0,0]:,}")
print(f"   False Positives: {cm[0,1]:,}")
print(f"   False Negatives: {cm[1,0]:,}")
print(f"   True Positives:  {cm[1,1]:,}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Non-Default', 'Default'],
            yticklabels=['Non-Default', 'Default'])
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title(f'Confusion Matrix (Threshold={best_threshold:.4f})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'c:\Users\gargd\Downloads\credit_risk_project\outputs\confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n   ✅ Saved: outputs/confusion_matrix.png")

print("\n✅ STEP 8 COMPLETE\n")

# Save threshold
import json
threshold_metrics = {
    'best_threshold': best_threshold,
    'recall_at_threshold': best_recall,
    'precision_at_threshold': best_precision
}
with open(r'c:\Users\gargd\Downloads\credit_risk_project\data\threshold_metrics.json', 'w') as f:
    json.dump(threshold_metrics, f, indent=2)
print("💾 Saved threshold metrics")
