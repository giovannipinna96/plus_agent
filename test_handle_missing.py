"""Test script for handle_missing_values tool."""

import pandas as pd
from tools.manipulation_tools import handle_missing_values

# Dataset path
file_path = "/u/gpinna/phd_projects/plusAgent/plus_agent/data/titanic.csv"

# First, let's check the missing values in the Age column
print("=" * 60)
print("INITIAL DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv(file_path)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nAge column info:")
print(f"  - Missing values: {df['age'].isnull().sum()}")
print(f"  - Total values: {len(df['age'])}")
print(f"  - Missing percentage: {df['age'].isnull().sum() / len(df['age']) * 100:.2f}%")
print(f"  - Current median: {df['age'].median()}")
print(f"  - Current mean: {df['age'].mean():.2f}")

# Now test the handle_missing_values tool
print("\n" + "=" * 60)
print("TESTING handle_missing_values TOOL")
print("=" * 60)

# Note: The column in the dataset is 'age' (lowercase), not 'Age'
# LangChain tools must be invoked using .invoke() with a dictionary
result = handle_missing_values.invoke({
    "file_path": file_path,
    "column_name": "age",  # lowercase as in the dataset
    "method": "median"      # lowercase as required by the tool (not "Median")
})

print(f"\nResult: {result}")

# Verify the result
print("\n" + "=" * 60)
print("VERIFICATION OF RESULTS")
print("=" * 60)

# Check if the output file was created
import os
output_path = file_path.replace('.csv', '_missing_handled.csv')
if os.path.exists(output_path):
    print(f"\n✅ Output file created: {output_path}")

    df_handled = pd.read_csv(output_path)
    print(f"\nHandled dataset shape: {df_handled.shape}")
    print(f"Missing values in age column after handling: {df_handled['age'].isnull().sum()}")

    # Compare before and after
    print(f"\nComparison:")
    print(f"  - Before: {df['age'].isnull().sum()} missing values")
    print(f"  - After: {df_handled['age'].isnull().sum()} missing values")
    print(f"  - Handled: {df['age'].isnull().sum() - df_handled['age'].isnull().sum()} values")
    print(f"  - Median used for filling: {df['age'].median()}")
else:
    print(f"\n❌ Output file not found: {output_path}")

print("\n" + "=" * 60)
print("TEST COMPLETED")
print("=" * 60)
