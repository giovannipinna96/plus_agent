#!/usr/bin/env python3
"""
Script to systematically improve all tool functions in smolagents_singleagent.py
This script enhances docstrings and return statements for better clarity and usability.
"""

import re

# Read the original file
with open('/u/gpinna/phd_projects/plusAgent/plus_agent/smolagents_singleagent.py', 'r') as f:
    content = f.read()

# Define comprehensive improvements for each tool function
# Format: (old_return_pattern, new_return_code, new_docstring)

improvements = {
    'read_csv_file': {
        'docstring': '''    """
    Loads and analyzes a CSV file, providing essential dataset information at a glance.

    This tool performs the initial step of any data analysis workflow by reading a CSV file
    and extracting key structural information. It's the first tool to use when working with
    new datasets, providing a quick overview of the data's shape, columns, and memory footprint.

    **Use Cases:**
    - Initial data exploration and validation
    - Verifying successful data import
    - Quick assessment of dataset size and structure
    - Memory usage estimation for large datasets

    **Capabilities:**
    - Reads CSV files with pandas for robust parsing
    - Calculates dataset dimensions (rows x columns)
    - Lists all column names in order
    - Computes memory usage in KB

    Args:
        file_path (str): Absolute or relative path to the CSV file to load.
                        Example: 'data/titanic.csv' or '/home/user/datasets/sales.csv'

    Returns:
        str: A formatted string containing:
             - Success confirmation message
             - Dataset shape as (rows, columns)
             - Complete list of column names
             - Memory usage in KB

             Success format:
             "CSV FILE LOADED SUCCESSFULLY

              Dataset Shape: X rows × Y columns

              Columns (Y total):
                1. column_name_1
                2. column_name_2
                ...

              Memory Usage: Z.ZZ KB"

             Error format:
             "ERROR: Failed to read CSV file
              Reason: [error details]
              Suggestion: [actionable fix]"

    **Important Notes:**
    - File must be in valid CSV format
    - Large files (>1GB) may take longer to load
    - Handles various CSV dialects automatically
    - Preserves column order from source file
    """''',
        'return_code': '''        df = pd.read_csv(file_path)

        n_rows, n_cols = df.shape
        columns = list(df.columns)
        memory_kb = df.memory_usage(deep=True).sum() / 1024

        output = [
            "CSV FILE LOADED SUCCESSFULLY",
            "",
            f"Dataset Shape: {n_rows:,} rows × {n_cols} columns",
            "",
            f"Columns ({n_cols} total):"
        ]

        for idx, col in enumerate(columns, 1):
            output.append(f"  {idx}. {col}")

        output.append("")
        output.append(f"Memory Usage: {memory_kb:.2f} KB")

        return "\\n".join(output)

    except FileNotFoundError:
        return f"ERROR: Failed to read CSV file\\nReason: File not found at path '{file_path}'\\nSuggestion: Verify the file path and ensure the file exists"
    except pd.errors.EmptyDataError:
        return f"ERROR: Failed to read CSV file\\nReason: The file at '{file_path}' is empty\\nSuggestion: Ensure the CSV file contains data"
    except pd.errors.ParserError as e:
        return f"ERROR: Failed to read CSV file\\nReason: CSV parsing error - {str(e)}\\nSuggestion: Check if the file is a valid CSV format"
    except Exception as e:
        return f"ERROR: Failed to read CSV file\\nReason: {type(e).__name__}: {str(e)}\\nSuggestion: Verify file permissions and format"'''
    }
}

# Apply improvements
print("This script outline shows the improvement approach.")
print("Due to the complexity, I'll need to improve the file directly with Edit operations.")
print(f"\\nTotal tool functions to improve: 51")
print("Improvements include:")
print("  - Enhanced docstrings with detailed descriptions")
print("  - Use case explanations")
print("  - Better parameter documentation")
print("  - Detailed return value formats")
print("  - Improved error messages with actionable suggestions")
print("  - Formatted output with clear sections and headers")
