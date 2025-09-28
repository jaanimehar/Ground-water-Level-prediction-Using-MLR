import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

import numpy as np
import os

# ------------------ SETTINGS ------------------
CSV_PATH   = r"D:\M.Tech DS\AIML application\Assignment-1\step2_Pre_Processed_output\groundwater_all_cleaned_encoded.csv"
TARGET     = "dataValue"
BIC_FILE   = r"D:\M.Tech DS\AIML application\Assignment-1\step5_model_selection_outputs\selected_features_BIC.csv"  # File with BIC-selected features
OUTPUT_DIR = r"D:\M.Tech DS\AIML application\Assignment-1\step7_model_prediction"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# -----------------------------------------------

# Load data
df = pd.read_csv(CSV_PATH)
y = df[TARGET]

# Load BIC-selected features
bic_features = pd.read_csv(BIC_FILE)["Selected_Features:"].tolist()

X = df[bic_features]

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Fit OLS model on training data
X_train_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_const).fit()

# Predict on testing data
X_test_const = sm.add_constant(X_test)
y_pred = model.predict(X_test_const)

# Evaluate performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)


print("Model performance on test data:")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

# Save predictions
pred_df = pd.DataFrame({
    "Features": [", ".join(bic_features)] * len(y_test),
    "Actual": y_test,
    "Predicted": y_pred
})
pred_df.to_csv(os.path.join(OUTPUT_DIR, "model-predictions.csv"), index=False)
print(f"Predictions saved to {OUTPUT_DIR}/model-predictions.csv")
# save summary
summary_df = pd.DataFrame({
    "Metric": ["R-squared(testing data)", "RMSE", "MSE"],
    "Value": [r2, rmse, mean_squared_error(y_test, y_pred)]
})
summary_df.to_csv(os.path.join(OUTPUT_DIR, "test-model-summary.csv"), index=False)
print(f"Summary saved to {OUTPUT_DIR}/test-model-summary.csv")

#plot actual vs predicted
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"))
plt.show()
print(f"Actual vs Predicted plot saved to {os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png')}")

#histogram of residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residuals_histogram.png"))
plt.show()
print(f"Residuals histogram saved to {os.path.join(OUTPUT_DIR, 'residuals_histogram.png')}")

# Residuals vs Fitted
plt.figure(figsize=(8, 6))  
train_residuals = y_train - model.fittedvalues
plt.scatter(model.fittedvalues, train_residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residuals_vs_fitted.png"))
plt.show()
print(f"Residuals vs Fitted plot saved to {os.path.join(OUTPUT_DIR, 'residuals_vs_fitted.png')}")

# Residuals vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residuals_vs_predicted.png"))
plt.show()
print(f"Residuals vs Predicted plot saved to {os.path.join(OUTPUT_DIR, 'residuals_vs_predicted.png')}")

# Q-Q plot
sm.qqplot(residuals, line ='s')
plt.title("Q-Q Plot of Residuals")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "qq_plot_residuals.png"))
plt.show()      
print(f"Q-Q plot of residuals saved to {os.path.join(OUTPUT_DIR, 'qq_plot_residuals.png')}")

# model estimation
coef_df = pd.DataFrame({
    "Feature": model.params.index,
    "Coefficient": model.params.values,
    "Std_Error": model.bse.values,
    "t_value": model.tvalues.values,
    "p_value": model.pvalues.values
})
coef_df.to_csv(os.path.join(OUTPUT_DIR, "model_coefficients.csv"), index=False)
summary_df = pd.DataFrame({
    "Metric": ["R_squared(training)", "Adj_R_squared", "F_stat", "F_pvalue", "Num_Predictors"],
    "Value": [ model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue, len(bic_features)]
})
summary_df.to_csv(os.path.join(OUTPUT_DIR, "trained-model_summary.csv"), index=False)
print(f"Model coefficients and summary saved to {OUTPUT_DIR}")
