import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import os
import statsmodels.api as sm

# ------------------ SETTINGS ------------------
CSV_PATH   = r"D:\M.Tech DS\AIML application\Assignment-1\step2_Pre_Processed_output\groundwater_all_cleaned_encoded.csv"  # Your clean dataset
TARGET     = "dataValue"
OUTPUT_DIR = r"D:\M.Tech DS\AIML application\Assignment-1\step6.3_aic_bic_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# -----------------------------------------------

# Load data
df = pd.read_csv(CSV_PATH)
y = df[TARGET]

# For demonstration, use first 10 features to illustrate
example_features = df.drop(columns=[TARGET]).columns[:10].tolist()

aic_values = []
bic_values = []
num_features = []

# Create all 1-5 feature combinations (demo)
for k in range(1, 6):
    for combo in combinations(example_features, k):
        X = df[list(combo)]
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        
        num_features.append(len(combo))
        aic_values.append(model.aic)
        bic_values.append(model.bic)

# Create DataFrame for plotting
plot_df = pd.DataFrame({
    "Num_Features": num_features,
    "AIC": aic_values,
    "BIC": bic_values
})

agg_df = plot_df.groupby("Num_Features")[["AIC", "BIC"]].min().reset_index()

plt.figure(figsize=(10,6))
sns.lineplot(x="Num_Features", y="AIC", data=agg_df, color='blue', label='AIC')
sns.lineplot(x="Num_Features", y="BIC", data=agg_df, color='red', label='BIC')
plt.xlabel("Number of Features in Model")
plt.ylabel("Minimum Information Criterion Value")
plt.title("AIC vs BIC (Minimum per Number of Features)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "aic_bic_comparison_line.png"))
plt.show()


print(f"Plot saved to {OUTPUT_DIR}/aic_bic_comparison_line.png")