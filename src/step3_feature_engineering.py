"""
STEP 3 — FEATURE ENGINEERING
"""
import pandas as pd
import numpy as np

# Load cleaned data
X = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\X_clean.csv')
y = pd.read_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\y_clean.csv')

print("=" * 60)
print("STEP 3 — FEATURE ENGINEERING")
print("=" * 60)

print(f"\n📊 Starting with {X.shape[1]} features")

# 1. age_years = DAYS_BIRTH / -365
X['age_years'] = X['DAYS_BIRTH'] / -365

# 2. employment_years = where DAYS_EMPLOYED < 0: DAYS_EMPLOYED/-365, else 0
X['employment_years'] = np.where(X['DAYS_EMPLOYED'] < 0, X['DAYS_EMPLOYED'] / -365, 0)

# 3. income_per_person = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS (clip min=1)
X['income_per_person'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS'].clip(lower=1)

# 4. annuity_to_credit = AMT_ANNUITY / AMT_CREDIT (clip min=1)
X['annuity_to_credit'] = X['AMT_ANNUITY'] / X['AMT_CREDIT'].clip(lower=1)

# 5. credit_to_income = AMT_CREDIT / AMT_INCOME_TOTAL
X['credit_to_income'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']

# 6. ext_source_mean = mean of EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
# Note: EXT_SOURCE_1 was dropped in step 2 (>40% missing), so we'll use EXT_SOURCE_2 and EXT_SOURCE_3
ext_cols = [col for col in ['EXT_SOURCE_2', 'EXT_SOURCE_3'] if col in X.columns]
if len(ext_cols) > 0:
    X['ext_source_mean'] = X[ext_cols].mean(axis=1)
else:
    X['ext_source_mean'] = 0

# 7. ext_source_min = min of EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
if len(ext_cols) > 0:
    X['ext_source_min'] = X[ext_cols].min(axis=1)
else:
    X['ext_source_min'] = 0

# 8. ext_source_std = std of EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
if len(ext_cols) > 0:
    X['ext_source_std'] = X[ext_cols].std(axis=1)
else:
    X['ext_source_std'] = 0

# 9. days_id_published_yr = DAYS_ID_PUBLISH / -365
X['days_id_published_yr'] = X['DAYS_ID_PUBLISH'] / -365

# 10. phone_change_yr = DAYS_LAST_PHONE_CHANGE / -365
X['phone_change_yr'] = X['DAYS_LAST_PHONE_CHANGE'] / -365

# Fill any new NaN values with median
new_features = ['age_years', 'employment_years', 'income_per_person', 'annuity_to_credit',
                'credit_to_income', 'ext_source_mean', 'ext_source_min', 'ext_source_std',
                'days_id_published_yr', 'phone_change_yr']

for col in new_features:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

print(f"\n✨ Created {len(new_features)} new features:")
print("\n   Feature Name              | Mean Value")
print("   " + "-" * 50)
for feat in new_features:
    print(f"   {feat:25s} | {X[feat].mean():10.4f}")

print(f"\n📦 Final feature count: {X.shape[1]} features")

print("\n✅ STEP 3 COMPLETE\n")

# Save engineered features
X.to_csv(r'c:\Users\gargd\Downloads\credit_risk_project\data\X_engineered.csv', index=False)
print("💾 Saved X_engineered.csv to data/ folder")
