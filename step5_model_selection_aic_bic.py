import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from itertools import combinations

# ------------------ SETTINGS ------------------
CSV_PATH   = r"D:\M.Tech DS\AIML application\Assignment-1\step2_Pre_Processed_output\groundwater_all_cleaned_encoded.csv"
TARGET     = "dataValue"          # replace with your target column
OUTPUT_DIR = r"D:\M.Tech DS\AIML application\Assignment-1\step5_model_selection_outputs"
CRITERION  = "BIC"     # choose "AIC" or "BIC"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ----------------------------------------------

df = pd.read_csv(CSV_PATH)
y  = df[TARGET]
X  = df.drop(columns=[TARGET])
X  = X.select_dtypes(include=[np.number])  # only numeric predictors

def calculate_ic(X, y, criterion="AIC"):
    """Fit OLS and return AIC or BIC."""
    model = sm.OLS(y, sm.add_constant(X)).fit()
    return model.aic if criterion.upper()=="AIC" else model.bic

def stepwise_selection(X, y, criterion="AIC"):
    """Forward-backward stepwise feature selection."""
    included = []
    changed  = True
    while changed:
        changed = False

        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pvals = pd.Series(index=excluded, dtype=float)
        for new_col in excluded:
            ic = calculate_ic(X[included + [new_col]], y, criterion)
            new_pvals[new_col] = ic
        if not new_pvals.empty:
            best_feature = new_pvals.idxmin()
            best_ic = new_pvals.min()
            current_ic = calculate_ic(X[included], y, criterion) if included else np.inf
            if best_ic < current_ic:
                included.append(best_feature)
                changed = True

        # backward step
        if included:
            ic_with_feature = pd.Series(index=included, dtype=float)
            for col in included:
                cols = list(set(included) - {col})
                ic = calculate_ic(X[cols], y, criterion) if cols else np.inf
                ic_with_feature[col] = ic
            worst_feature = ic_with_feature.idxmin()
            if ic_with_feature.min() < calculate_ic(X[included], y, criterion):
                included.remove(worst_feature)
                changed = True

    return included

best_features = stepwise_selection(X, y, criterion=CRITERION)
print(f"Selected features based on {CRITERION}:")
print(best_features)

# Save results
with open(os.path.join(OUTPUT_DIR, f"selected_features_{CRITERION}.txt"), "w") as f:
    f.write(f"Selected features based on {CRITERION}:\n")
    for feat in best_features:
        f.write(feat + "\n")
print(f"\nFeature list saved to selected_features_{CRITERION}.txt")  