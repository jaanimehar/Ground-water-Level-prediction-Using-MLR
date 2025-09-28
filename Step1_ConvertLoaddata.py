import os
import pandas as pd

# Path to your Excel file
input_file = r"D:\M.Tech DS\AIML application\Assignment-1\GWL.xlsx"
# Path to save CSV
output_file = r"D:\M.Tech DS\AIML application\Assignment-1\step1_Load_Convert_Data\GWL.csv"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Read the first sheet of the Excel file
df = pd.read_excel(input_file, sheet_name=0)  # Use sheet_name="Sheet1" if needed

# Save as CSV
df.to_csv(output_file, index=False)

print(f"✅ Converted and saved as CSV: {output_file}")
