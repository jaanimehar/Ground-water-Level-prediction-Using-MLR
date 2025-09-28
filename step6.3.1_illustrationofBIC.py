import pandas as pd
import statsmodels.api as sm
from itertools import combinations
import os

# ------------------ SETTINGS ------------------
CSV_PATH   = r"D:\M.Tech DS\AIML application\Assignment-1\step2_Pre_Processed_output\groundwater_all_cleaned_encoded.csv"  # Your clean dataset
TARGET     = "dataValue"
OUTPUT_DIR = r"D:\M.Tech DS\AIML application\Assignment-1\step6.3_aic_bic_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# -----------------------------------------------

# Load data
df = pd.read_csv(CSV_PATH)
y = df[TARGET]
all_features = df.drop(columns=[TARGET]).columns.tolist()

# ------------------ Stepwise-like illustration ------------------
# WARNING: Using all combinations is impossible for 174 features
# So we illustrate with 3-feature combinations for example
example_features = all_features[:50]  # Take first 50 columns to demonstrate
results = []

# Create combinations of 1 to 50 features
for k in range(1, 51):  # Change 4 to a smaller number for practical runs
    for combo in combinations(example_features, k):
        X = df[list(combo)]
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        results.append({
            "Features": combo,
            "Num_Features": k,
            "AIC": model.aic,
            "BIC": model.bic,
            "R_squared": model.rsquared
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "bic_candidate_example.csv"), index=False)
print(f"Illustration saved to {OUTPUT_DIR}/bic_candidate_example.csv")

# ------------------ Identify lowest AIC and BIC ------------------
best_aic = results_df.loc[results_df['AIC'].idxmin()]
best_bic = results_df.loc[results_df['BIC'].idxmin()]

print("\nBest model by AIC:")
print(best_aic)

print("\nBest model by BIC:")
print(best_bic)
