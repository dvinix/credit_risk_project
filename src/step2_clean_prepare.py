"""
STEP 2 — CLEAN & PREPARE
"""
import pandas as pd
import numpy as np
import os

# Load the dataset
df = pd.read_csv('data/application_train.csv')

print("=" * 60)
print("STEP 2 — CLEAN & PREPARE")
print("=" * 60)

# Keep only numeric columns + TARGET
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_numeric = df[numeric_cols].copy()
print(f"\n✂️  Kept only numeric columns: {len(df_numeric.columns)} columns")

# Drop columns with more than 40% missing values
missing_pct = df_numeric.isnull().sum() / len(df_numeric) * 100
cols_to_drop = missing_pct[missing_pct > 40].index.tolist()
print(f"\n🗑️  Dropping {len(cols_to_drop)} columns with >40% missing values:")
for col in cols_to_drop[:5]:
    print(f"   - {col} ({missing_pct[col]:.2f}% missing)")
if len(cols_to_drop) > 5:
    print(f"   ... and {len(cols_to_drop) - 5} more")

df_clean = df_numeric.drop(columns=cols_to_drop)
print(f"\n   Remaining columns: {len(df_clean.columns)}")

# Fill remaining missing values with median
missing_before = df_clean.isnull().sum().sum()
print(f"\n💧 Missing values before filling: {missing_before:,}")

for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

missing_after = df_clean.isnull().sum().sum()
print(f"   Missing values after filling: {missing_after:,}")

# Separate X and y
y = df_clean['TARGET'].copy()
X = df_clean.drop(columns=['TARGET'])

print(f"\n📦 Final Shapes:")
print(f"   X shape: {X.shape} (rows, features)")
print(f"   y shape: {y.shape} (rows,)")
print(f"\n   Features: {X.shape[1]}")
print(f"   Samples: {X.shape[0]:,}")

print("\n✅ STEP 2 COMPLETE\n")

# Save for next steps
X.to_csv('data/X_clean.csv', index=False)
y.to_csv('data/y_clean.csv', index=False)
print("💾 Saved X_clean.csv and y_clean.csv to data/ folder")
