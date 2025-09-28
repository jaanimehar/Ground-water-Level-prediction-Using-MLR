import pandas as pd
import os

# ---------------- SETTINGS ----------------
READ_DIR  = r"D:\M.Tech DS\AIML application\Assignment-1\step7_model_prediction"
WRITE_DIR = r"D:\M.Tech DS\AIML application\Assignment-1\step8_significant_interpretation"

# Output report file (CSV)
REPORT_FILE = os.path.join(WRITE_DIR, "Step10_Interpretation.csv")
# -------------------------------------------

# Load model coefficients, summary, and metrics
coef_df    = pd.read_csv(os.path.join(READ_DIR, "model_coefficients.csv"))
summary_df = pd.read_csv(os.path.join(READ_DIR, "trained-model_summary.csv"))
metrics_df = pd.read_csv(os.path.join(READ_DIR, "test-model-summary.csv"))

# --- Significant features (p < 0.05) ---
signif_df = coef_df[coef_df["p_value"] < 0.05].copy()
signif_df["Impact"] = signif_df["Coefficient"].apply(lambda x: "Positive" if x > 0 else "Negative")
signif_df["Section"] = "Significant Features"

# Keep only required columns
signif_df = signif_df[["Section", "Feature", "Impact", "Coefficient", "p_value"]]
signif_df.rename(columns={"p_value": "P-value", "Coefficient": "Coef"}, inplace=True)

# --- Model Fit Metrics ---
summary_df["Section"] = "Model Fit Metrics"
summary_df.rename(columns={"Metric": "Metric", "Value": "Value"}, inplace=True)
summary_df = summary_df[["Section", "Metric", "Value"]]

# --- Prediction Metrics ---
metrics_df["Section"] = "Prediction Metrics"
metrics_df.rename(columns={"Metric": "Metric", "Value": "Value"}, inplace=True)
metrics_df = metrics_df[["Section", "Metric", "Value"]]

# --- Confidence in Interpretation with next line ---

confidence_text = (
    "The model explains most of the variability in groundwater levels (high R-squared).\n "
    "Significant features with p < 0.05 are likely reliable predictors. \n"
    "Prediction metrics (RMSE for regression and Accuracy/F1 for categorized classes) indicate reasonable predictive power."
)
confidence_df = pd.DataFrame([{"Section": "Confidence in Interpretation", "Description": confidence_text}])
# --- Combine all sections ---
final_df = pd.concat([signif_df, summary_df, metrics_df, confidence_df], ignore_index=True, sort=False)

# Save to CSV
os.makedirs(WRITE_DIR, exist_ok=True)
final_df.to_csv(REPORT_FILE, index=False)

print(f"Step 10 interpretation report saved to {REPORT_FILE}")
