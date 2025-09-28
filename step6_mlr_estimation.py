import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

# ------------------ SETTINGS ------------------
CSV_PATH       = r"D:\M.Tech DS\AIML application\Assignment-1\step2_Pre_Processed_output\groundwater_all_cleaned_encoded.csv"       # your clean data CSV
TARGET         = "dataValue"                       # dependent variable
FEATURES_FILE  = r"D:\M.Tech DS\AIML application\Assignment-1\step5_model_selection_outputs\selected_features_AIC.csv"   # BIC or AIC selected features
OUTPUT_DIR     = r"D:\M.Tech DS\AIML application\Assignment-1\step6.1_Model_estimation_AIC_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# -----------------------------------------------

# Load data
df = pd.read_csv(CSV_PATH)
y = df[TARGET]

# Load selected features
selected_features = pd.read_csv(FEATURES_FILE)["Selected_Features:"].tolist()
X = df[selected_features]

# Add constant term for intercept
X_const = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X_const).fit()

# ------------------ 1. Save model summary ------------------
summary_file = os.path.join(OUTPUT_DIR, "mlr_model_summary.txt")
with open(summary_file, "w") as f:
    f.write(model.summary().as_text())
print(f"Model summary saved to {summary_file}")

# ------------------ 2. Coefficients table ------------------
coef_table = pd.DataFrame({
    "Feature": model.params.index,
    "Coefficient": model.params.values,
    "Std_Error": model.bse,
    "t_stat": model.tvalues,
    "p_value": model.pvalues
})
coef_table.to_csv(os.path.join(OUTPUT_DIR, "mlr_coefficients.csv"), index=False)
print("Coefficients table saved.")

# ------------------ 3. Predicted values & residuals ------------------
df['Predicted'] = model.predict(X_const)
df['Residuals'] = df[TARGET] - df['Predicted']

# ------------------ 4. Diagnostic Plots ------------------
# Residuals vs Predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Predicted'], y=df['Residuals'])
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residuals_vs_predicted.png"))
plt.close()

# Histogram of residuals
plt.figure(figsize=(8,6))
sns.histplot(df['Residuals'], bins=30, kde=True)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residuals_histogram.png"))
plt.close()

# Q-Q plot
sm.qqplot(df['Residuals'], line='s')
plt.title('Q-Q Plot of Residuals')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residuals_qqplot.png"))
plt.close()

# ------------------ 5. Print key metrics ------------------
print("\nKey model metrics:")
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
print(f"F-statistic: {model.fvalue:.2f}  (p-value: {model.f_pvalue:.4e})")

print(f"\nDiagnostics completed. All outputs saved in {OUTPUT_DIR}")