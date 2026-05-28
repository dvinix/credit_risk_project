"""
STEP 4 — TRAIN / TEST SPLIT
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load engineered data
X = pd.read_csv('data/X_engineered.csv')
y = pd.read_csv('data/y_clean.csv')['TARGET']

print("=" * 60)
print("STEP 4 — TRAIN / TEST SPLIT")
print("=" * 60)

# Split 80% train, 20% test, stratify=y, random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"\n📊 Split Results:")
print(f"   Train shape: {X_train.shape}")
print(f"   Test shape:  {X_test.shape}")

print(f"\n🎯 Class Balance in Train Set:")
train_counts = y_train.value_counts().sort_index()
print(f"   Non-Default (0): {train_counts[0]:,} ({train_counts[0]/len(y_train)*100:.2f}%)")
print(f"   Default (1):     {train_counts[1]:,} ({train_counts[1]/len(y_train)*100:.2f}%)")

print("\n✅ STEP 4 COMPLETE\n")

# Save splits
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
print("💾 Saved train/test splits to data/ folder")
