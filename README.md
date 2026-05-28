# Credit Risk Scoring & Default Prediction System
> End-to-end ML pipeline · Tabular financial data · Binary classification

---

## Project Overview

An end-to-end machine learning pipeline that predicts loan default probability on 50,000+ synthetic loan records modelled on LendingClub/Home Credit data.  The pipeline covers data ingestion → SQL profiling → feature engineering → EDA → modelling → explainability → business output.

---

## Results

| Metric | Value |
|--------|-------|
| Dataset | 50,000 loans (1:36 class imbalance) |
| Features engineered | 12 domain-specific |
| Best model | XGBoost + SMOTE |
| **ROC-AUC** | **0.727** |
| Default recall (tuned) | **~75%** |
| Risk tiers | Low / Medium / High |
| Simulated NPA reduction | ~65% at recommended threshold |

---

## Stack

```
Python 3.10+  ·  Pandas  ·  NumPy  ·  Scikit-learn  ·  XGBoost
SHAP  ·  Imbalanced-learn  ·  Matplotlib  ·  Seaborn  ·  SQLite
```

---

## Project Structure

```
credit_risk_project/
├── data/
│   ├── loan_data_raw.csv            # 50,000-row synthetic dataset
│   └── loan_data_engineered.csv     # 37 columns (12 new features)
├── src/
│   ├── generate_dataset.py          # Step 1 · Data generation
│   ├── sql_profiling.py             # Step 2 · SQL queries (missingness, outliers)
│   ├── feature_engineering.py       # Step 3 · 12 domain features
│   ├── eda.py                       # Step 4 · EDA + chi-sq / ANOVA
│   ├── modelling.py                 # Step 5 · LR → RF → XGBoost + SMOTE
│   ├── shap_explainability.py       # Step 6 · SHAP waterfall / beeswarm
│   └── business_output.py          # Step 7 · Risk tiers + threshold
├── models/
│   ├── xgb_pipeline.pkl            # Saved XGBoost pipeline (joblib)
│   └── metrics.json                # AUC, threshold, feature list
├── outputs/
│   ├── fig1_default_by_segment.png
│   ├── fig2_default_ltv_term.png
│   ├── fig3_feature_distributions.png
│   ├── fig4_correlation_matrix.png
│   ├── fig5_roc_pr_curves.png
│   ├── fig6_shap_beeswarm.png
│   ├── fig7_shap_waterfall.png
│   ├── fig8_shap_feature_importance.png
│   └── fig9_risk_segmentation.png
├── notebooks/
│   └── credit_risk_pipeline.ipynb  # Full interactive notebook
└── run_pipeline.py                 # Master runner (all 7 steps)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn xgboost shap imbalanced-learn matplotlib seaborn scipy

# 2. Run the full pipeline
python run_pipeline.py

# 3. Or run individual steps
python src/generate_dataset.py
python src/sql_profiling.py
python src/feature_engineering.py
python src/eda.py
python src/modelling.py
python src/shap_explainability.py
python src/business_output.py
```

---

## 12 Engineered Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | `dti_ratio` | Total debt / annual income |
| 2 | `credit_utilisation` | Credit used / credit limit |
| 3 | `ltv_ratio` | Loan amount / home value |
| 4 | `log_income` | log1p(annual income) — normalise skew |
| 5 | `monthly_payment` | Estimated monthly instalment |
| 6 | `payment_to_income` | Monthly payment / monthly income |
| 7 | `repayment_streak_norm` | On-time payment months / 60 |
| 8 | `credit_score_tier` | Ordinal band (1 Poor → 5 Exceptional) |
| 9 | `income_band` | Ordinal band (1 <30K → 5 200K+) |
| 10 | `has_delinquency` | Binary: any delinquency in 2 yrs |
| 11 | `high_dti_flag` | Binary: DTI ≥ 0.43 |
| 12 | `risk_composite` | Weighted risk score (0–1) |

---

## Business Risk Rules (from SHAP)

| Driver | Rule |
|--------|------|
| Credit Score Tier | Poor tier → 6.4% default rate vs 1.0% exceptional |
| Debt-to-Income | DTI ≥ 0.43 → flag for manual review |
| Credit Utilisation | >75% → require collateral or co-signer |
| Delinquencies | Any in 24 months → doubles default odds |
| Loan Term | 60-month loans carry 2× default rate of 12-month |

---

## Resume Bullets

> Built an end-to-end credit default prediction pipeline on 50K+ loan records using XGBoost, achieving 0.727 ROC-AUC; applied SMOTE to resolve 1:36 class imbalance, improving minority-class recall to 75%.

> Engineered 12 domain-specific features (debt-to-income ratio, LTV, repayment streak) and used SHAP explainability to identify top 5 default drivers, translating outputs into actionable credit policy rules.

> Conducted statistical EDA using chi-square tests and ANOVA; designed a 3-tier risk segmentation (Low/Medium/High) enabling simulated NPA reduction of 65% at the recommended approval threshold.
