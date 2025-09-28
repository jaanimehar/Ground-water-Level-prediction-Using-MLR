import pandas as pd
import statsmodels.api as sm
import os

# ------------------ SETTINGS ------------------
CSV_PATH       = r"D:\M.Tech DS\AIML application\Assignment-1\step2_Pre_Processed_output\groundwater_all_cleaned_encoded.csv"  # Your dataset
TARGET         = "dataValue"                       # Dependent variable
BIC_FILE       = r"D:\M.Tech DS\AIML application\Assignment-1\step5_model_selection_outputs\selected_features_BIC.csv"
OUTPUT_DIR     = r"D:\M.Tech DS\AIML application\Assignment-1\step6.4_comparison_allvsBIC_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# -----------------------------------------------

# Load data
df = pd.read_csv(CSV_PATH)
y = df[TARGET]

# ------------------ Full model ------------------
X_full = df.drop(columns=[TARGET])
X_full_const = sm.add_constant(X_full)
model_full = sm.OLS(y, X_full_const).fit()

# ------------------ BIC-selected model ------------------
# bic_features = pd.read_csv(BIC_FILE, header=None)[0].tolist()
bic_features = pd.read_csv(BIC_FILE)["Selected_Features:"].tolist()

X_bic = df[bic_features]
X_bic_const = sm.add_constant(X_bic)
model_bic = sm.OLS(y, X_bic_const).fit()

# ------------------ Comparison Table ------------------
def create_coef_table(model, name):
    return pd.DataFrame({
        "Feature": model.params.index,
        f"{name}_Coef": model.params.values,
        f"{name}_StdErr": model.bse,
        f"{name}_t": model.tvalues,
        f"{name}_p": model.pvalues
    })

# Merge tables on Feature
full_table = create_coef_table(model_full, "Full")
bic_table  = create_coef_table(model_bic, "BIC")

comparison = pd.merge(full_table, bic_table, on="Feature", how="outer")
comparison.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
print(f"Comparison table saved to {OUTPUT_DIR}/model_comparison.csv")

# ------------------ Summary Metrics ------------------
summary_metrics = pd.DataFrame({
    "Model": ["Full", "BIC"],
    "R_squared": [model_full.rsquared, model_bic.rsquared],
    "Adj_R_squared": [model_full.rsquared_adj, model_bic.rsquared_adj],
    "F_stat": [model_full.fvalue, model_bic.fvalue],
    "F_pvalue": [model_full.f_pvalue, model_bic.f_pvalue]
})
summary_metrics.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"), index=False)
print(f"Summary metrics saved to {OUTPUT_DIR}/summary_metrics.csv")
