import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import Point

# ========== USER SETTINGS ==========
# Path to your clean CSV
CSV_PATH = r"D:\M.Tech DS\AIML application\Assignment-1\step2_Pre_Processed_output\groundwater_all_cleaned_encoded.csv"
# Name of the dependent variable (target)
TARGET = "dataValue"     # <-- replace with your actual target column name
# Directory to save all plots
OUTPUT_DIR = r"D:\M.Tech DS\AIML application\Assignment-1\step3_eda_outputs"
OUTPUT_DIR_PairsPlot = r"D:\M.Tech DS\AIML application\Assignment-1\step3_eda_outputs\PairsPlotOutput"

# ===================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_PairsPlot, exist_ok=True)


# ---------- 1. Load Data ----------
df = pd.read_csv(CSV_PATH)

print("\n--- BASIC INFO ---")
print(df.info())
print("\n--- FIRST 5 ROWS ---")
print(df.head())

# ---------- 2. Summary Statistics ----------
print("\n--- DESCRIPTIVE STATS ---")
print(df.describe().T)

# Save summary to CSV
df.describe().T.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))

# ---------- 3. Missing Values Check ----------
missing = df.isna().sum().sort_values(ascending=False)
missing.to_csv(os.path.join(OUTPUT_DIR, "missing_values_report.csv"))
print("\nMissing values report saved.")

# ---------- 4. Correlation Analysis ----------
corr = df.corr(numeric_only=True)
corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))
plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# ---------- 5. Target Distribution ----------
plt.figure(figsize=(8, 5))
sns.histplot(df[TARGET], kde=True, bins=30)
plt.title(f"Distribution of {TARGET}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{TARGET}_distribution.png"))
plt.close()

# ---------- 6. Pairwise Plots with Target ----------
# To avoid huge plots, pick top 5 correlated features with target
top_features = (
    corr[TARGET]
    .abs()
    .sort_values(ascending=False)
    .drop(labels=[TARGET])
    .head(5)
    .index
    .tolist()
)
sns.pairplot(df[[TARGET] + top_features])
plt.savefig(os.path.join(OUTPUT_DIR, "pairplot_top5.png"))
plt.close()

# Select numeric columns only (independent variables)
numeric_cols = df.select_dtypes(include='number').columns.tolist()
numeric_cols.remove(TARGET)  # remove target from features

# Loop over independent variables and create separate plots
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[col], y=df[TARGET])
    sns.regplot(x=df[col], y=df[TARGET], scatter=False, color='red')  # optional regression line
    plt.xlabel(col)
    plt.ylabel(TARGET)
    plt.title(f'{TARGET} vs {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_PairsPlot, f'{TARGET}_vs_{col}.png'))
    plt.close()

# ---------- 7. Outlier Detection ----------
# Using Z-score > 3 as simple check
numeric_cols = df.select_dtypes(include=[np.number]).columns
z_scores = np.abs(stats.zscore(df[numeric_cols]))
outliers = (z_scores > 3).sum(axis=1)
print(f"\nRows with potential outliers (z>3) : {np.sum(outliers > 0)}")
outlier_rows = df[outliers > 0]
outlier_rows.to_csv(os.path.join(OUTPUT_DIR, "potential_outliers.csv"), index=False)

# ---------- 8. Time/Spatial Trend Example ----------
df['date'] = pd.to_datetime(df['date'])
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="date", y=TARGET)
plt.title("Temporal Trend of Groundwater Level")
plt.xlabel("Date")
plt.ylabel(TARGET)
plt.savefig(os.path.join(OUTPUT_DIR, "temporal_trend.png"))
plt.close()

#-------------9. Create GeoDataFrame-----------------
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],
    crs="EPSG:4326"
)

# Simple scatter map
gdf.plot(column=TARGET, cmap='coolwarm', legend=True, figsize=(8,8),
         markersize=40, alpha=0.7, edgecolor='k', linewidth=0.5, legend_kwds={'label': TARGET} )
plt.title(f'Spatial Distribution of {TARGET}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig(os.path.join(OUTPUT_DIR, 'spatial_distribution_points.png'))
plt.close()

print("\nEDA completed. Plots and reports saved to:", OUTPUT_DIR)
