import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# ------------------ SETTINGS ------------------
CSV_PATH = r"D:\M.Tech DS\AIML application\Assignment-1\step2_Pre_Processed_output\groundwater_all_cleaned_encoded.csv"
TARGET = "dataValue"          # replace with your target column
OUTPUT_DIR = r"D:\M.Tech DS\AIML application\Assignment-1\step4_model_assumption_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ---------------------------------------------

# Load data
df = pd.read_csv(CSV_PATH)

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Only numeric features
X_numeric = X.select_dtypes(include=[np.number])

# ---------------- 1. Correlation Matrix ----------------
corr = X_numeric.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_correlation_matrix.png'))
plt.close()

# ---------------- 2. VIF (Multicollinearity) ----------------
X_const = sm.add_constant(X_numeric)  # add constant term for intercept
vif_data =  pd.DataFrame()
vif_data['Feature'] = X_numeric.columns
vif_data['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_numeric.shape[1])]
vif_data.sort_values(by='VIF', ascending=False, inplace=True)
vif_data.to_csv(os.path.join(OUTPUT_DIR, 'vif_values.csv'), index=False)
print("VIF values saved. High VIF (>10) indicates multicollinearity.")

# ---------------- 3. Fit Linear Regression Model ----------------
X_const = sm.add_constant(X_numeric)
model = sm.OLS(y, X_const).fit()
df['Predicted'] = model.predict(X_const)
df['Residuals'] = df[TARGET] - df['Predicted']

# ---------------- 4. Linearity & Homoscedasticity ----------------
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Predicted'], y=df['Residuals'])
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_vs_predicted.png'))
plt.close()

# ---------------- 5. Normality of Residuals ----------------
plt.figure(figsize=(8,6))
sns.histplot(df['Residuals'], kde=True, bins=30)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_histogram.png'))
plt.close()

sm.qqplot(df['Residuals'], line='s')
plt.title('Q-Q Plot of Residuals')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_qqplot.png'))
plt.close()

# ---------------- 6. Summary ----------------
with open(os.path.join(OUTPUT_DIR, 'model_summary.txt'), 'w') as f:
    f.write(model.summary().as_text())

print(f"Assumption checks completed. Outputs saved in {OUTPUT_DIR}")
