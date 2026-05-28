"""
MASTER SCRIPT - RUN ALL STEPS
Credit Risk Default Prediction Project
"""
import subprocess
import sys
import os

# Change to project root directory
project_root = r'c:\Users\gargd\Downloads\credit_risk_project'
os.chdir(project_root)

steps = [
    ('STEP 1 — LOAD & BASIC PROFILE', 'src\\step1_load_profile.py'),
    ('STEP 2 — CLEAN & PREPARE', 'src\\step2_clean_prepare.py'),
    ('STEP 3 — FEATURE ENGINEERING', 'src\\step3_feature_engineering.py'),
    ('STEP 4 — TRAIN / TEST SPLIT', 'src\\step4_train_test_split.py'),
    ('STEP 5 — BASELINE MODEL (Logistic Regression)', 'src\\step5_baseline_model.py'),
    ('STEP 6 — MAIN MODEL (XGBoost)', 'src\\step6_xgboost_model.py'),
    ('STEP 7 — HANDLE CLASS IMBALANCE (SMOTE)', 'src\\step7_smote_model.py'),
    ('STEP 8 — THRESHOLD TUNING', 'src\\step8_threshold_tuning.py'),
    ('STEP 9 — SHAP EXPLAINABILITY', 'src\\step9_shap_explainability.py'),
    ('STEP 10 — RISK TIERS & BUSINESS OUTPUT', 'src\\step10_risk_tiers.py'),
    ('STEP 11 — SAVE EVERYTHING', 'src\\step11_save_final.py'),
    ('FINAL SUMMARY', 'src\\final_summary.py'),
]

print("=" * 70)
print("CREDIT RISK DEFAULT PREDICTION PROJECT")
print("Running All Steps Sequentially")
print("=" * 70)

for i, (step_name, script_path) in enumerate(steps, 1):
    print(f"\n{'='*70}")
    print(f"Running {step_name}")
    print(f"{'='*70}\n")
    
    result = subprocess.run([sys.executable, script_path], 
                          capture_output=False, 
                          text=True)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {step_name} failed!")
        sys.exit(1)
    
    print(f"\n✅ {step_name} completed successfully")

print("\n" + "=" * 70)
print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\n📁 Outputs:")
print("   - Models saved in: models/")
print("   - Plots saved in: outputs/")
print("   - Metrics saved in: models/metrics.json")
print("\n" + "=" * 70)
