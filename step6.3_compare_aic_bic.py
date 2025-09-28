import pandas as pd
import statsmodels.api as sm
import os

# ------------------ SETTINGS ------------------
CSV_PATH       = r"D:\M.Tech DS\AIML application\Assignment-1\step2_Pre_Processed_output\groundwater_all_cleaned_encoded.csv"  # Your dataset
TARGET         = "dataValue"

AIC_FILE       = r"D:\M.Tech DS\AIML application\Assignment-1\step5_model_selection_outputs\selected_features_AIC.csv"  # AIC-selected features
BIC_FILE       = r"D:\M.Tech DS\AIML application\Assignment-1\step5_model_selection_outputs\selected_features_BIC.csv"  # BIC-selected features

OUTPUT_DIR     = r"D:\M.Tech DS\AIML application\Assignment-1\step6.3_aic_bic_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# -----------------------------------------------

# Load dataset
df = pd.read_csv(CSV_PATH)
y = df[TARGET]

# ------------------ Full model ------------------
X_full = df.drop(columns=[TARGET])
X_full_const = sm.add_constant(X_full)
model_full = sm.OLS(y, X_full_const).fit()

# ------------------ Load feature lists ------------------
# Use header=None in case CSV has no column names
aic_features = pd.read_csv(AIC_FILE)["Selected_Features:"].tolist()
bic_features = pd.read_csv(BIC_FILE)["Selected_Features:"].tolist()


# ------------------ Fit AIC model ------------------
X_aic = df[aic_features]
X_aic_const = sm.add_constant(X_aic)
model_aic = sm.OLS(y, X_aic_const).fit()

# ------------------ Fit BIC model ------------------
X_bic = df[bic_features]
X_bic_const = sm.add_constant(X_bic)
model_bic = sm.OLS(y, X_bic_const).fit()

# ------------------ Create coefficient comparison ------------------
def create_coef_table(model, name):
    return pd.DataFrame({
        "Feature": model.params.index,
        f"{name}_Coef": model.params.values,
        f"{name}_StdErr": model.bse,
        f"{name}_t": model.tvalues,
        f"{name}_p": model.pvalues
    })

full_table = create_coef_table(model_full, "Full")
coef_aic = create_coef_table(model_aic, "AIC")
coef_bic = create_coef_table(model_bic, "BIC")

# Merge coefficient tables
coef_comparison = pd.merge(full_table, coef_aic, on="Feature", how="outer")
coef_comparison = pd.merge(coef_comparison, coef_bic, on="Feature", how="outer")
coef_comparison.to_csv(os.path.join(OUTPUT_DIR, "coef_comparison.csv"), index=False)
print(f"Coefficient comparison saved to {OUTPUT_DIR}/coef_comparison.csv")

# ------------------ Summary metrics comparison ------------------
summary_metrics = pd.DataFrame({
    "Model": ["Full", "AIC", "BIC"],
    "R_squared": [model_full.rsquared, model_aic.rsquared, model_bic.rsquared],
    "Adj_R_squared": [model_full.rsquared_adj, model_aic.rsquared_adj, model_bic.rsquared_adj],
    "F_stat": [model_full.fvalue, model_aic.fvalue, model_bic.fvalue],
    "F_pvalue": [model_full.f_pvalue, model_aic.f_pvalue, model_bic.f_pvalue],
    "Num_Predictors": [len(X_full.columns), len(X_aic.columns), len(X_bic.columns)]
})
summary_metrics.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics_comparison.csv"), index=False)
print(f"Summary metrics comparison saved to {OUTPUT_DIR}/summary_metrics_comparison.csv")
