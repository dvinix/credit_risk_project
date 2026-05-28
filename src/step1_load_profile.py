"""
STEP 1 — LOAD & BASIC PROFILE
"""
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('data/application_train.csv')

print("=" * 60)
print("STEP 1 — LOAD & BASIC PROFILE")
print("=" * 60)

# Shape
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"   Rows: {df.shape[0]:,}")
print(f"   Columns: {df.shape[1]}")

# TARGET value counts and default rate
print("\n🎯 TARGET Distribution:")
target_counts = df['TARGET'].value_counts().sort_index()
print(target_counts)
default_rate = (df['TARGET'].sum() / len(df)) * 100
print(f"\n   Default Rate: {default_rate:.2f}%")
print(f"   Non-Default: {target_counts[0]:,} ({100-default_rate:.2f}%)")
print(f"   Default: {target_counts[1]:,} ({default_rate:.2f}%)")

# Top 10 columns by missing value %
print("\n❌ Top 10 Columns by Missing Value %:")
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False).head(10)
for col, pct in missing_pct.items():
    print(f"   {col:40s}: {pct:6.2f}%")

# Dtypes summary
print("\n📋 Data Types Summary:")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"   Numeric columns: {len(numeric_cols)}")
print(f"   Categorical columns: {len(categorical_cols)}")
print(f"   Total: {len(df.columns)}")

print("\n✅ STEP 1 COMPLETE\n")
