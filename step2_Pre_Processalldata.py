import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# === 1️⃣ Load file ===
input_file = r"D:\M.Tech DS\AIML application\Assignment-1\step1_Load_Convert_Data\GWL.csv"  # Excel or CSV
sheet_name = 0  # For Excel

if input_file.lower().endswith(".csv"):
    df = pd.read_csv(input_file)
else:
    df = pd.read_excel(input_file, sheet_name=sheet_name)

print(f"Original dataset: {df.shape[0]} rows x {df.shape[1]} columns")

# === 2️⃣ Replace undefined values with NaN ===
df.replace(["undefined", "Undefined", "NA", "N/A", ""], pd.NA, inplace=True)

# === 3️⃣ Fill missing values ===
# Numeric columns: fill with median
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns: fill with 'Unknown'
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# === 4️⃣ Encode categorical columns ===
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# === 5️⃣ Save cleaned CSV ===
output_file = r"D:\M.Tech DS\AIML application\Assign\step2_Pre_Processed_output\groundwater_all_cleaned_encoded.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_csv(output_file, index=False)
  
print(f"✅ All columns cleaned and encoded. Saved as: {output_file}")
print(f"Final dataset: {df.shape[0]} rows x {df.shape[1]} columns")

print("✅ Data preprocessing completed.")
print(f"✅ Saved: {output_file}")
