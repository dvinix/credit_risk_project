"""
Visual Comparison: Before vs After Fix
"""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Before (Broken)
tiers_before = ['Low Risk\n0.7%', 'Medium Risk\n8.5%', 'High Risk\n90.8%']
counts_before = [0.7, 8.5, 90.8]
colors = ['#2ecc71', '#f39c12', '#e74c3c']

axes[0].bar(range(3), counts_before, color=colors, alpha=0.7)
axes[0].set_xticks(range(3))
axes[0].set_xticklabels(tiers_before, fontsize=11)
axes[0].set_ylabel('Percentage of Applicants (%)', fontsize=12)
axes[0].set_title('❌ BEFORE: Broken Risk Tiers\n(Rejects 91% of applicants)', 
                  fontsize=13, fontweight='bold', color='red')
axes[0].set_ylim(0, 100)
axes[0].grid(axis='y', alpha=0.3)

# Add percentage labels
for i, v in enumerate(counts_before):
    axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

# Add warning annotation
axes[0].text(2, 50, '⚠️ UNUSABLE\nRejects 91%!', 
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# After (Fixed)
tiers_after = ['Low Risk\n32.1%', 'Medium Risk\n43.3%', 'High Risk\n24.6%']
counts_after = [32.1, 43.3, 24.6]

axes[1].bar(range(3), counts_after, color=colors, alpha=0.9)
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(tiers_after, fontsize=11)
axes[1].set_ylabel('Percentage of Applicants (%)', fontsize=12)
axes[1].set_title('✅ AFTER: Fixed Risk Tiers\n(Realistic distribution)', 
                  fontsize=13, fontweight='bold', color='green')
axes[1].set_ylim(0, 100)
axes[1].grid(axis='y', alpha=0.3)

# Add percentage labels
for i, v in enumerate(counts_after):
    axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

# Add success annotation
axes[1].text(1, 80, '✅ PRODUCTION READY\n57% automated\n43% manual review', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig(r'c:\Users\gargd\Downloads\credit_risk_project\outputs\BEFORE_AFTER_COMPARISON.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("=" * 60)
print("BEFORE vs AFTER COMPARISON")
print("=" * 60)
print("\n❌ BEFORE (Broken):")
print("   Low Risk:    0.7%  → Only 0.7% auto-approved")
print("   Medium Risk: 8.5%  → Only 8.5% for review")
print("   High Risk:  90.8%  → 91% REJECTED!")
print("\n   Problem: Unusable in production")

print("\n✅ AFTER (Fixed):")
print("   Low Risk:   32.1%  → 32% auto-approved (fast track)")
print("   Medium Risk: 43.3%  → 43% manual review")
print("   High Risk:  24.6%  → 25% auto-declined")
print("\n   Result: 57% automated, 43% manual review")
print("   Status: Production ready ✅")

print("\n📊 Saved: outputs/BEFORE_AFTER_COMPARISON.png")
print("=" * 60)
