import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# ------------------ SETTINGS ------------------
CSV_PATH = r"D:\M.Tech DS\AIML application\Assignment-1\step2_Pre_Processed_output\groundwater_all_cleaned_encoded.csv"
TARGET = "dataValue"          # replace with your target column
SPATIAL = "district"                  # optional, replace with your spatial column, or None
OUTPUT_DIR = r"D:\M.Tech DS\AIML application\Assignment-1\step3_eda_outputs\pairplots10Output"
TOP_N = 10                             # number of top correlated features to plot
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ---------------------------------------------

# Load data
df = pd.read_csv(CSV_PATH)

# Select numeric columns only
numeric_cols = df.select_dtypes(include='number').columns.tolist()
numeric_cols.remove(TARGET)  # remove target from features

# Compute correlations with target
corr_with_target = df[numeric_cols + [TARGET]].corr()[TARGET].abs().sort_values(ascending=False)
top_features = corr_with_target.head(TOP_N + 1).index.tolist()  # +1 in case target is included
if TARGET in top_features:
    top_features.remove(TARGET)

# Loop over top features and plot individually
for col in top_features:
    plt.figure(figsize=(6,4))
    
    if SPATIAL and SPATIAL in df.columns:
        sns.scatterplot(x=df[col], y=df[TARGET], hue=df[SPATIAL], palette='tab10', s=60, alpha=0.7, legend='full')
    else:
        sns.scatterplot(x=df[col], y=df[TARGET], color='steelblue', s=60, alpha=0.7, legend=False)
    
    # Regression line
    sns.regplot(x=df[col], y=df[TARGET], scatter=False, color='red')
    
    plt.xlabel(col)
    plt.ylabel(TARGET)
    plt.title(f'{TARGET} vs {col}')
    plt.tight_layout()
    
    
    # Save individual figure
    plt.savefig(os.path.join(OUTPUT_DIR, f'{TARGET}_vs_{col}.png'))
    plt.close()

print(f"Top {TOP_N} pair plots saved individually in {OUTPUT_DIR}")
