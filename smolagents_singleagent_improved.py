#!/usr/bin/env python3
"""
Multi-Agent Data Analysis System using smolagents - IMPROVED VERSION

A comprehensive multi-agent system for data analysis powered by smolagents and Hugging Face models.
This system orchestrates specialized AI agents to perform end-to-end data science workflows.

This improved version features:
- Enhanced docstrings with detailed explanations for all 50+ tools
- Improved return statements with better formatting and more information
- Additional tools for specific use cases (e.g., survival prediction)
- Comprehensive error handling with informative messages

Architecture:
- Manager Agent: Orchestrates the overall workflow
- Data Reader Agent: Analyzes datasets and provides structure information
- Data Manipulation Agent: Handles data preprocessing and transformations
- Data Operations Agent: Performs mathematical operations and aggregations
- ML Prediction Agent: Trains and evaluates machine learning models

Author: Multi-Agent System (Enhanced)
Framework: smolagents (Hugging Face)
"""

import os
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Statistical imports
from scipy.stats import pearsonr, spearmanr, ttest_ind, chi2_contingency
from scipy import stats

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# smolagents imports
from smolagents import CodeAgent, InferenceClientModel, tool, TransformersModel

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "model_id": os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-14B-Instruct"),
    "max_new_tokens": int(os.getenv("MAX_TOKENS", "1024")),
    "temperature": float(os.getenv("TEMPERATURE", "0.1")),
    "default_dataset": os.getenv("DEFAULT_DATASET_PATH", "data/titanic.csv"),
    "huggingface_token": os.getenv("HUGGINGFACE_TOKEN"),
}

print(f"🚀 Multi-Agent System Configuration:")
print(f"   Model: {CONFIG['model_id']}")
print(f"   Max Tokens: {CONFIG['max_new_tokens']}")
print(f"   Temperature: {CONFIG['temperature']}")
print(f"   Default Dataset: {CONFIG['default_dataset']}")


# ============================================================================
# DATA READING TOOLS - ENHANCED
# ============================================================================

@tool
def load_dataset(file_path: str) -> str:
    """
    Loads a CSV dataset and returns comprehensive basic information about its structure.

    This is typically the first tool to use when starting data analysis. It provides
    a quick overview of the dataset size, helping you understand the scale of data
    you're working with before diving into detailed analysis.

    **Use Cases:**
    - Initial dataset exploration to verify successful loading
    - Quick sanity check before starting analysis
    - Confirming dataset dimensions match expectations
    - Understanding data volume for performance considerations

    **Capabilities:**
    - Loads CSV files from any valid file path
    - Reports total number of rows (observations/samples)
    - Reports total number of columns (features/variables)
    - Provides basic shape information for planning next steps

    Args:
        file_path (str): Path to the CSV file to load. Can be absolute or relative path.
                        Example: 'data/titanic.csv' or '/home/user/data/sales.csv'

    Returns:
        str: A formatted string containing:
             - Confirmation of successful loading
             - Number of rows (data points/observations)
             - Number of columns (features/variables)
             - Basic shape tuple

             Example format:
             "✓ Dataset loaded successfully!

              Dataset Shape:
              - Rows (observations): 891
              - Columns (features): 12
              - Shape: (891, 12)

              Ready for analysis!"

    **Error Handling:**
    - File not found: Returns clear error message with the attempted path
    - Invalid CSV format: Returns parsing error details
    - Permission issues: Returns access denied message

    **Note:** This tool only loads and reports basic info. Use other tools like
    get_column_names(), get_data_types(), or get_first_rows() for more details.
    """
    try:
        df = pd.read_csv(file_path)
        n_rows, n_cols = df.shape
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        output = [
            "✓ Dataset loaded successfully!",
            "",
            "Dataset Shape:",
            f"  • Rows (observations): {n_rows:,}",
            f"  • Columns (features): {n_cols}",
            f"  • Total cells: {n_rows * n_cols:,}",
            f"  • Memory usage: {memory_mb:.2f} MB",
            "",
            "Ready for analysis!"
        ]

        return "\n".join(output)

    except FileNotFoundError:
        return f"✗ Error: File not found at path '{file_path}'\n\nPlease check:\n  • Path is correct\n  • File exists\n  • You have read permissions"
    except pd.errors.EmptyDataError:
        return f"✗ Error: File '{file_path}' is empty"
    except pd.errors.ParserError as e:
        return f"✗ Error: Invalid CSV format in '{file_path}'\n\nDetails: {str(e)}"
    except Exception as e:
        return f"✗ Error loading dataset: {type(e).__name__}: {str(e)}"


@tool
def get_column_names(file_path: str) -> str:
    """
    Retrieves and displays all column names in the dataset with their data types.

    This tool provides a structured overview of all variables in your dataset,
    categorizing them by type (numeric or categorical). This is essential for
    understanding what data you have available and planning your analysis strategy.

    **Use Cases:**
    - Discovering all available features in a new dataset
    - Verifying column names before operations (avoiding typos)
    - Understanding the mix of numeric vs categorical variables
    - Planning feature engineering or model training
    - Checking if specific columns exist before using them

    **Capabilities:**
    - Lists all column names in order
    - Categorizes each column as numeric or categorical
    - Provides column count
    - Shows data types for each column

    Args:
        file_path (str): Path to the CSV file to analyze.
                        Example: 'data/titanic.csv'

    Returns:
        str: A formatted string containing:
             - Total number of columns
             - Each column name with its type (numeric/categorical)
             - Summary statistics (# numeric vs # categorical)

             Example format:
             "Dataset contains 12 columns:

              Numeric Columns (5):
                1. PassengerId (int64)
                2. Age (float64)
                3. Fare (float64)
                ...

              Categorical Columns (7):
                1. Name (object)
                2. Sex (object)
                3. Embarked (object)
                ...

              Summary: 5 numeric, 7 categorical columns"

    **Tips:**
    - Numeric columns can be used for mathematical operations and regression
    - Categorical columns may need encoding before use in ML models
    - Object dtype usually indicates text/categorical data
    """
    try:
        df = pd.read_csv(file_path)
        n_cols = len(df.columns)

        numeric_cols = []
        categorical_cols = []

        for col in df.columns:
            dtype_str = str(df[col].dtype)
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append((col, dtype_str))
            else:
                categorical_cols.append((col, dtype_str))

        output = [
            f"Dataset contains {n_cols} columns:",
            ""
        ]

        if numeric_cols:
            output.append(f"Numeric Columns ({len(numeric_cols)}):")
            for idx, (col, dtype) in enumerate(numeric_cols, 1):
                output.append(f"  {idx}. {col} ({dtype})")
            output.append("")

        if categorical_cols:
            output.append(f"Categorical Columns ({len(categorical_cols)}):")
            for idx, (col, dtype) in enumerate(categorical_cols, 1):
                output.append(f"  {idx}. {col} ({dtype})")
            output.append("")

        output.append(f"Summary: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical columns")

        return "\n".join(output)

    except FileNotFoundError:
        return f"✗ Error: File not found at '{file_path}'"
    except Exception as e:
        return f"✗ Error getting column names: {type(e).__name__}: {str(e)}"


@tool
def get_data_types(file_path: str) -> str:
    """
    Provides detailed information about data types of all columns in the dataset.

    Understanding data types is crucial for proper data analysis and transformation.
    This tool gives you a complete breakdown of how pandas interprets each column,
    which affects what operations you can perform and how memory is used.

    **Use Cases:**
    - Identifying columns that need type conversion
    - Understanding memory usage patterns
    - Planning data preprocessing steps
    - Detecting columns that should be categorical but are numeric (or vice versa)
    - Verifying data loaded correctly with expected types

    **Capabilities:**
    - Shows pandas dtype for each column
    - Groups columns by data type category
    - Provides counts for each type category
    - Identifies potential type issues

    Args:
        file_path (str): Path to the CSV file to analyze.

    Returns:
        str: A formatted string containing:
             - Complete list of columns with their pandas dtypes
             - Grouped summary by type (int, float, object, etc.)
             - Type distribution counts
             - Memory implications

             Example format:
             "Data Types Analysis:

              Column Details:
                • PassengerId: int64
                • Survived: int64
                • Pclass: int64
                • Name: object
                • Sex: object
                • Age: float64
                • Fare: float64

              Type Distribution:
                • int64: 3 columns
                • float64: 2 columns
                • object: 2 columns

              Notes:
                - object dtype typically contains text/categorical data
                - float64 may indicate numeric data with decimals or missing values
                - int64 represents whole numbers"
    """
    try:
        df = pd.read_csv(file_path)

        output = [
            "Data Types Analysis:",
            "",
            "Column Details:"
        ]

        # List all columns with types
        for col in df.columns:
            output.append(f"  • {col}: {df[col].dtype}")

        output.append("")

        # Type distribution
        type_counts = df.dtypes.value_counts()
        output.append("Type Distribution:")
        for dtype, count in type_counts.items():
            output.append(f"  • {dtype}: {count} column{'s' if count > 1 else ''}")

        output.append("")
        output.append("Notes:")
        output.append("  - object dtype typically contains text/categorical data")
        output.append("  - float64 may indicate numeric data with decimals or missing values")
        output.append("  - int64 represents whole numbers without missing values")

        return "\n".join(output)

    except Exception as e:
        return f"✗ Error getting data types: {type(e).__name__}: {str(e)}"


@tool
def get_null_counts(file_path: str) -> str:
    """
    Analyzes and reports missing values (nulls/NaN) across all columns in the dataset.

    Missing data is one of the most common data quality issues. This tool provides
    a comprehensive analysis of where data is missing, how much is missing, and the
    percentage impact. This information is critical for deciding on imputation strategies
    or whether to drop columns/rows.

    **Use Cases:**
    - Data quality assessment
    - Planning missing value imputation strategies
    - Deciding whether to drop columns with too much missing data
    - Understanding data collection issues
    - Prioritizing data cleaning efforts

    **Capabilities:**
    - Counts missing values in each column
    - Calculates percentage of missing data
    - Ranks columns by amount of missing data
    - Provides actionable recommendations
    - Identifies if dataset is complete (no missing values)

    Args:
        file_path (str): Path to the CSV file to analyze.

    Returns:
        str: A formatted string containing:
             - Total dataset completeness percentage
             - List of columns with missing values (sorted by severity)
             - Count and percentage of missing values per column
             - Recommendations based on missing data patterns

             Example format:
             "Missing Values Analysis:

              Overall Completeness: 88.5%
              Total cells: 10,692
              Missing cells: 1,234 (11.5%)

              Columns with Missing Data (ranked by severity):

                1. Cabin: 687 missing (77.1%)
                   → Recommendation: Consider dropping or use 'Unknown' category

                2. Age: 177 missing (19.9%)
                   → Recommendation: Impute with median or mean

                3. Embarked: 2 missing (0.2%)
                   → Recommendation: Impute with mode or drop rows

              Complete Columns (no missing data): 9 columns"

    **Recommendations Guide:**
    - < 5% missing: Safe to impute or drop rows
    - 5-30% missing: Impute with statistical methods
    - 30-60% missing: Consider dropping or special handling
    - > 60% missing: Likely should drop column unless critical
    """
    try:
        df = pd.read_csv(file_path)
        null_counts = df.isnull().sum()
        total_rows = len(df)
        total_cells = df.size
        total_missing = null_counts.sum()

        output = [
            "Missing Values Analysis:",
            "",
            f"Overall Completeness: {((total_cells - total_missing) / total_cells * 100):.1f}%",
            f"Total cells: {total_cells:,}",
            f"Missing cells: {total_missing:,} ({(total_missing / total_cells * 100):.1f}%)",
            ""
        ]

        # Find columns with missing values
        missing_cols = null_counts[null_counts > 0].sort_values(ascending=False)

        if len(missing_cols) > 0:
            output.append("Columns with Missing Data (ranked by severity):")
            output.append("")

            for idx, (col, count) in enumerate(missing_cols.items(), 1):
                pct = (count / total_rows) * 100
                output.append(f"  {idx}. {col}: {count:,} missing ({pct:.1f}%)")

                # Add recommendation
                if pct < 5:
                    rec = "Safe to impute or drop rows"
                elif pct < 30:
                    rec = "Impute with statistical methods (mean/median/mode)"
                elif pct < 60:
                    rec = "Consider dropping or advanced imputation"
                else:
                    rec = "Consider dropping column unless critical"
                output.append(f"     → Recommendation: {rec}")
                output.append("")

            complete_cols = len(df.columns) - len(missing_cols)
            output.append(f"Complete Columns (no missing data): {complete_cols} columns")
        else:
            output.append("✓ No missing values found! Dataset is complete.")

        return "\n".join(output)

    except Exception as e:
        return f"✗ Error counting nulls: {type(e).__name__}: {str(e)}"


@tool
def get_unique_values(file_path: str, column_name: str) -> str:
    """
    Analyzes and displays unique values in a specific column with frequency distribution.

    Understanding the unique values in a column is essential for:
    - Identifying categorical variables and their categories
    - Detecting data quality issues (typos, inconsistencies)
    - Deciding encoding strategies for machine learning
    - Finding rare categories that might need grouping

    **Use Cases:**
    - Exploring categorical variables (Gender, Class, etc.)
    - Identifying data entry errors or inconsistencies
    - Understanding cardinality (number of unique values)
    - Planning one-hot encoding or label encoding
    - Detecting rare categories or outliers
    - Checking value distributions

    **Capabilities:**
    - Lists all unique values (for low cardinality columns)
    - Shows sample of values (for high cardinality columns)
    - Provides frequency counts for each value
    - Calculates percentages
    - Identifies most and least common values

    Args:
        file_path (str): Path to the CSV file.
        column_name (str): Name of the column to analyze.
                          Example: 'Sex', 'Pclass', 'Embarked'

    Returns:
        str: A formatted string containing:
             - Number of unique values (cardinality)
             - Number of missing values
             - Complete list (if ≤ 20 unique values) or sample (if > 20)
             - Frequency distribution with counts and percentages
             - Most/least common values

             Example format for categorical:
             "Unique Values Analysis: Sex

              Cardinality: 2 unique values
              Missing values: 0
              Total non-null values: 891

              Value Distribution:

                1. male: 577 occurrences (64.8%)
                   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

                2. female: 314 occurrences (35.2%)
                   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

              Most common: male (64.8%)
              Least common: female (35.2%)"

    **Note:** For columns with > 20 unique values, shows top 10 and bottom 10
    to keep output manageable while still providing insights.
    """
    try:
        df = pd.read_csv(file_path)

        if column_name not in df.columns:
            available_cols = ", ".join(df.columns[:10])
            return f"✗ Error: Column '{column_name}' not found.\n\nAvailable columns: {available_cols}..."

        col = df[column_name]
        n_missing = col.isnull().sum()
        n_total = len(col)
        n_non_null = col.count()

        # Get value counts
        value_counts = col.value_counts(dropna=False)
        n_unique = len(value_counts)

        output = [
            f"Unique Values Analysis: {column_name}",
            "",
            f"Cardinality: {n_unique:,} unique values",
            f"Missing values: {n_missing:,} ({(n_missing/n_total*100):.1f}%)",
            f"Total non-null values: {n_non_null:,}",
            ""
        ]

        if n_unique == 0:
            output.append("⚠ Column contains only missing values")
            return "\n".join(output)

        output.append("Value Distribution:")
        output.append("")

        # Show all values if ≤ 20, otherwise show top 10
        show_count = min(n_unique, 20)

        for idx, (value, count) in enumerate(value_counts.head(show_count).items(), 1):
            pct = (count / n_non_null) * 100
            # Create a simple bar visualization
            bar_length = int(pct / 2)  # Scale to max 50 chars
            bar = "▓" * bar_length

            val_str = str(value) if not pd.isna(value) else "<missing>"
            output.append(f"  {idx}. {val_str}: {count:,} occurrences ({pct:.1f}%)")
            if bar:
                output.append(f"     {bar}")
            output.append("")

        if n_unique > 20:
            output.append(f"... and {n_unique - 20} more unique values")
            output.append("")

        # Summary statistics
        most_common = value_counts.index[0]
        most_common_pct = (value_counts.iloc[0] / n_non_null) * 100
        output.append(f"Most common: {most_common} ({most_common_pct:.1f}%)")

        if n_unique > 1:
            least_common = value_counts.index[-1]
            least_common_pct = (value_counts.iloc[-1] / n_non_null) * 100
            output.append(f"Least common: {least_common} ({least_common_pct:.1f}%)")

        return "\n".join(output)

    except Exception as e:
        return f"✗ Error analyzing unique values: {type(e).__name__}: {str(e)}"


# I'll continue with the rest of the tools in the next section...
# This file is getting large, so I'll create it in multiple parts

