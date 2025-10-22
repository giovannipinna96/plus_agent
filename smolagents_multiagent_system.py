#!/usr/bin/env python3
"""
Multi-Agent Data Analysis System using smolagents - Enhanced Version

Architecture:
- Manager Agent: Orchestrates the overall workflow
- Data Reader Agent: Analyzes datasets and provides structure information  
- Data Manipulation Agent: Handles data preprocessing and transformations
- Data Operations Agent: Performs mathematical operations and aggregations
- Data Visualization Agent: Creates charts and visualizations
- ML Prediction Agent: Trains and evaluates machine learning models
- Statistical Analysis Agent: Performs statistical tests and correlations
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, r2_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib

# smolagents imports
from smolagents import CodeAgent, tool, TransformersModel, VLLMModel

from smolagents_tools import *


# from transformers import BitsAndBytesConfig
import torch

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # "model_id": os.getenv("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
    "model_id": os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),  # Agents use Llama-3.1-8B
    "max_new_tokens": int(os.getenv("MAX_TOKENS", "4096")),
    "temperature": float(os.getenv("TEMPERATURE", "0.6")),
    "default_dataset": os.getenv("DEFAULT_DATASET_PATH", "data/titanic.csv"),
    "max_length": os.getenv("MAX_LENGTH", "10000"),
    "huggingface_token": os.getenv("HUGGINGFACE_TOKEN"),
}

# ============================================================================
# COLUMN DESCRIPTION GENERATION WITH LLM
# ============================================================================

def analyze_dataset_columns(file_path: str) -> List[Dict[str, str]]:
    """
    Analyze dataset columns and extract their characteristics.

    Args:
        file_path: Path to the dataset file

    Returns:
        List of dictionaries with column information
    """
    try:
        df = pd.read_csv(file_path)
        columns_info = []

        for col in df.columns:
            dtype = df[col].dtype

            # Classify column type
            if pd.api.types.is_numeric_dtype(dtype):
                if pd.api.types.is_integer_dtype(dtype):
                    col_type = "numeric (integer)"
                else:
                    col_type = "numeric (float)"
            elif pd.api.types.is_categorical_dtype(dtype) or df[col].nunique() < 10:
                col_type = "categorical"
            elif pd.api.types.is_object_dtype(dtype):
                col_type = "string"
            else:
                col_type = "object"

            # Additional statistics
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()

            columns_info.append({
                "name": col,
                "type": col_type,
                "unique_values": unique_count,
                "null_count": null_count
            })

        return columns_info

    except Exception as e:
        print(f"Error analyzing columns: {str(e)}")
        return []


def generate_column_descriptions_with_llm(columns_info: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Return static, detailed descriptions for Titanic dataset columns.

    This function provides comprehensive, well-documented descriptions for each column
    in the Titanic dataset, based on official Kaggle competition documentation.

    Args:
        columns_info: List of dictionaries with column information

    Returns:
        Dictionary mapping column names to their detailed descriptions
    """
    # Static detailed descriptions for Titanic dataset columns (in English)
    # Source: https://www.kaggle.com/competitions/titanic/data
    TITANIC_COLUMN_DESCRIPTIONS = {
        "PassengerId": "Unique identifier assigned to each passenger in the dataset. This is an integer value that serves as a primary key for tracking individual passengers throughout the analysis.",

        "Survived": "Binary survival indicator showing whether the passenger survived the Titanic disaster. Values are 0 (did not survive/died) or 1 (survived). This is the target variable for predictive modeling.",

        "Pclass": "Ticket class serving as a proxy for socio-economic status. Values are 1 (First class - Upper class), 2 (Second class - Middle class), or 3 (Third class - Lower class). First class passengers had access to the best accommodations and higher survival rates.",

        "Name": "Full name of the passenger including title (Mr., Mrs., Miss., Master., Dr., Rev., etc.). The title can provide insights into social status, gender, and age group. Format typically follows: 'Surname, Title. Firstname Middlename'.",

        "Sex": "Biological sex/gender of the passenger. Values are 'male' or 'female'. Gender was a significant factor in survival rates due to the 'women and children first' evacuation protocol.",

        "Age": "Age of the passenger in years, with fractional values for passengers less than 1 year old (e.g., 0.42 for infants). This column contains missing values (NaN) for passengers whose age was not recorded. Age is important for identifying vulnerable groups like children and elderly.",

        "SibSp": "Number of siblings or spouses the passenger had aboard the Titanic. Sibling relationships include brother, sister, stepbrother, and stepsister. Spouse relationships include husband and wife (mistresses and fiancés were not counted). This helps understand family group sizes.",

        "Parch": "Number of parents or children the passenger had aboard the Titanic. Parent relationships include mother and father. Child relationships include daughter, son, stepdaughter, and stepson. Note that some children traveled only with a nanny, resulting in Parch=0 despite being minors.",

        "Ticket": "Ticket number, which can be either numeric or alphanumeric. Some tickets were shared among family members or groups traveling together. The ticket prefix sometimes indicates the point of purchase or ticket type.",

        "Fare": "Passenger fare paid for the ticket, denominated in British pounds (£). Higher fares generally correlate with higher class accommodations. Some passengers shared tickets, so fare may represent cost for multiple people. This column can have 0 values for crew members or complimentary passengers.",

        "Cabin": "Cabin number assigned to the passenger, typically alphanumeric (e.g., 'C85', 'B96 B98'). The letter indicates the deck (A-G, with A being the highest and G the lowest). This column has many missing values as cabin information was not recorded for all passengers, particularly third class.",

        "Embarked": "Port of embarkation where the passenger boarded the Titanic. Values are C (Cherbourg, France), Q (Queenstown, now Cobh, Ireland), or S (Southampton, England). The embarkation port can correlate with passenger demographics and ticket class."
    }

    print(f"\n📚 Using static detailed column descriptions from Kaggle Titanic documentation...")

    # Create descriptions dictionary matching available columns
    descriptions = {}
    for col_info in columns_info:
        col_name = col_info['name']
        if col_name in TITANIC_COLUMN_DESCRIPTIONS:
            descriptions[col_name] = TITANIC_COLUMN_DESCRIPTIONS[col_name]
        else:
            # Fallback for any non-standard columns
            descriptions[col_name] = f"A {col_info['type']} column with {col_info['unique_values']} unique values and {col_info['null_count']} missing values."

    print(f"✅ Loaded detailed descriptions for {len(descriptions)} columns\n")

    return descriptions


def format_column_descriptions(descriptions: Dict[str, str]) -> str:
    """
    Format column descriptions for inclusion in prompts.

    Args:
        descriptions: Dictionary mapping column names to descriptions

    Returns:
        Formatted string with column descriptions
    """
    formatted = "Database Column Information:\n"
    for col_name, description in descriptions.items():
        formatted += f"  - {col_name}: {description}\n"

    return formatted

# ============================================================================
# DATA READING TOOLS - Simplified and Specific
# ============================================================================

# @tool
# def load_dataset(file_path: str) -> str:
#     """
#     Load a dataset and return basic shape information.
    
#     Args:
#         file_path: Path to the CSV file
        
#     Returns:
#         String with dataset shape (rows, columns)
#     """
#     try:
#         df = pd.read_csv(file_path)
#         return f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns"
#     except Exception as e:
#         return f"Error loading dataset: {str(e)}"


# @tool
# def get_column_names(file_path: str) -> str:
#     """
#     Get list of all column names in the dataset.
    
#     Args:
#         file_path: Path to the CSV file
        
#     Returns:
#         Comma-separated list of column names
#     """
#     try:
#         df = pd.read_csv(file_path)
#         return f"Columns: {', '.join(df.columns.tolist())}"
#     except Exception as e:
#         return f"Error getting columns: {str(e)}"


# @tool
# def get_data_types(file_path: str) -> str:
#     """
#     Get data types of all columns.
    
#     Args:
#         file_path: Path to the CSV file
        
#     Returns:
#         String with column names and their data types
#     """
#     try:
#         df = pd.read_csv(file_path)
#         dtypes = df.dtypes.to_dict()
#         result = []
#         for col, dtype in dtypes.items():
#             result.append(f"{col}: {dtype}")
#         return "Data types:\n" + "\n".join(result)
#     except Exception as e:
#         return f"Error getting data types: {str(e)}"


# @tool
# def get_null_counts(file_path: str) -> str:
#     """
#     Count missing values in each column.
    
#     Args:
#         file_path: Path to the CSV file
        
#     Returns:
#         String with missing value counts per column
#     """
#     try:
#         df = pd.read_csv(file_path)
#         null_counts = df.isnull().sum()
#         result = []
#         for col, count in null_counts.items():
#             if count > 0:
#                 pct = (count / len(df)) * 100
#                 result.append(f"{col}: {count} ({pct:.1f}%)")
        
#         if result:
#             return "Missing values:\n" + "\n".join(result)
#         else:
#             return "No missing values found"
#     except Exception as e:
#         return f"Error counting nulls: {str(e)}"


# @tool
# def get_unique_values(file_path: str, column_name: str) -> str:
#     """
#     Get unique values in a specific column.
    
#     Args:
#         file_path: Path to the CSV file
#         column_name: Name of the column
        
#     Returns:
#         String with unique values and their count
#     """
#     try:
#         df = pd.read_csv(file_path)
#         if column_name not in df.columns:
#             return f"Column '{column_name}' not found"
        
#         unique_vals = df[column_name].dropna().unique()
#         n_unique = len(unique_vals)
        
#         if n_unique <= 20:
#             return f"Column '{column_name}' has {n_unique} unique values: {', '.join(map(str, unique_vals))}"
#         else:
#             sample = unique_vals[:10]
#             return f"Column '{column_name}' has {n_unique} unique values. First 10: {', '.join(map(str, sample))}"
#     except Exception as e:
#         return f"Error getting unique values: {str(e)}"


# @tool
# def get_numeric_summary(file_path: str, column_name: str) -> str:
#     """
#     Get statistical summary for a numeric column.
    
#     Args:
#         file_path: Path to the CSV file
#         column_name: Name of the numeric column
        
#     Returns:
#         String with mean, median, std, min, max, quartiles
#     """
#     try:
#         df = pd.read_csv(file_path)
#         if column_name not in df.columns:
#             return f"Column '{column_name}' not found"
        
#         col = df[column_name].dropna()
        
#         stats = {
#             "count": len(col),
#             "mean": col.mean(),
#             "median": col.median(),
#             "std": col.std(),
#             "min": col.min(),
#             "25%": col.quantile(0.25),
#             "50%": col.quantile(0.50),
#             "75%": col.quantile(0.75),
#             "max": col.max()
#         }
        
#         result = f"Statistics for '{column_name}':\n"
#         for key, val in stats.items():
#             result += f"  {key}: {val:.2f}\n"
        
#         return result
#     except Exception as e:
#         return f"Error getting summary: {str(e)}"


# @tool
# def get_first_rows(file_path: str, n_rows: int = 5) -> str:
#     """
#     Get first N rows of the dataset.
    
#     Args:
#         file_path: Path to the CSV file
#         n_rows: Number of rows to return
        
#     Returns:
#         String representation of first N rows
#     """
#     try:
#         df = pd.read_csv(file_path)
#         preview = df.head(n_rows).to_string()
#         return f"First {n_rows} rows:\n{preview}"
#     except Exception as e:
#         return f"Error getting rows: {str(e)}"


# # ============================================================================
# # DATA MANIPULATION TOOLS - More Specific
# # ============================================================================

# @tool
# def drop_column(file_path: str, column_name: str) -> str:
#     """
#     Drop a specific column from the dataset.
    
#     Args:
#         file_path: Path to the CSV file
#         column_name: Name of column to drop
        
#     Returns:
#         String confirming column drop and new file path
#     """
#     try:
#         df = pd.read_csv(file_path)
#         if column_name not in df.columns:
#             return f"Column '{column_name}' not found"
        
#         df = df.drop(columns=[column_name])
#         output_path = file_path.replace('.csv', '_dropped.csv')
#         df.to_csv(output_path, index=False)
        
#         return f"Dropped column '{column_name}'. New shape: {df.shape}. Saved to: {output_path}"
#     except Exception as e:
#         return f"Error dropping column: {str(e)}"


# @tool
# def drop_null_rows(file_path: str, column_name: Optional[str] = None) -> str:
#     """
#     Drop rows with missing values.
    
#     Args:
#         file_path: Path to the CSV file
#         column_name: Specific column to check for nulls (optional, if None drops any row with null)
        
#     Returns:
#         String confirming rows dropped
#     """
#     try:
#         df = pd.read_csv(file_path)
#         original_len = len(df)
        
#         if column_name:
#             if column_name not in df.columns:
#                 return f"Column '{column_name}' not found"
#             df = df.dropna(subset=[column_name])
#         else:
#             df = df.dropna()
        
#         dropped = original_len - len(df)
#         output_path = file_path.replace('.csv', '_no_nulls.csv')
#         df.to_csv(output_path, index=False)
        
#         return f"Dropped {dropped} rows with nulls. New shape: {df.shape}. Saved to: {output_path}"
#     except Exception as e:
#         return f"Error dropping null rows: {str(e)}"


# @tool
# def fill_numeric_nulls(file_path: str, column_name: str, method: str = "mean") -> str:
#     """
#     Fill missing values in a numeric column.
    
#     Args:
#         file_path: Path to the CSV file
#         column_name: Name of the numeric column
#         method: Fill method - 'mean', 'median', 'zero', or numeric value
        
#     Returns:
#         String confirming fill operation
#     """
#     try:
#         df = pd.read_csv(file_path)
#         if column_name not in df.columns:
#             return f"Column '{column_name}' not found"
        
#         null_count = df[column_name].isnull().sum()
#         if null_count == 0:
#             return f"No missing values in '{column_name}'"
        
#         if method == "mean":
#             fill_value = df[column_name].mean()
#         elif method == "median":
#             fill_value = df[column_name].median()
#         elif method == "zero":
#             fill_value = 0
#         else:
#             try:
#                 fill_value = float(method)
#             except:
#                 return f"Invalid method: {method}"
        
#         df[column_name] = df[column_name].fillna(fill_value)
        
#         output_path = file_path.replace('.csv', '_filled.csv')
#         df.to_csv(output_path, index=False)
        
#         return f"Filled {null_count} nulls in '{column_name}' with {method} ({fill_value:.2f}). Saved to: {output_path}"
#     except Exception as e:
#         return f"Error filling nulls: {str(e)}"


# @tool
# def fill_categorical_nulls(file_path: str, column_name: str, method: str = "mode") -> str:
#     """
#     Fill missing values in a categorical column.
    
#     Args:
#         file_path: Path to the CSV file
#         column_name: Name of the categorical column
#         method: Fill method - 'mode', 'unknown', or specific value
        
#     Returns:
#         String confirming fill operation
#     """
#     try:
#         df = pd.read_csv(file_path)
#         if column_name not in df.columns:
#             return f"Column '{column_name}' not found"
        
#         null_count = df[column_name].isnull().sum()
#         if null_count == 0:
#             return f"No missing values in '{column_name}'"
        
#         if method == "mode":
#             fill_value = df[column_name].mode()[0] if len(df[column_name].mode()) > 0 else "unknown"
#         elif method == "unknown":
#             fill_value = "unknown"
#         else:
#             fill_value = method
        
#         df[column_name] = df[column_name].fillna(fill_value)
        
#         output_path = file_path.replace('.csv', '_filled.csv')
#         df.to_csv(output_path, index=False)
        
#         return f"Filled {null_count} nulls in '{column_name}' with '{fill_value}'. Saved to: {output_path}"
#     except Exception as e:
#         return f"Error filling categorical nulls: {str(e)}"


# @tool
# def encode_categorical(file_path: str, column_name: str, encoding_type: str = "onehot") -> str:
#     """
#     Encode a categorical column.
    
#     Args:
#         file_path: Path to the CSV file
#         column_name: Name of categorical column
#         encoding_type: 'onehot' or 'label'
        
#     Returns:
#         String confirming encoding operation
#     """
#     try:
#         df = pd.read_csv(file_path)
#         if column_name not in df.columns:
#             return f"Column '{column_name}' not found"
        
#         if encoding_type == "onehot":
#             dummies = pd.get_dummies(df[column_name], prefix=column_name, dtype=int)
#             df = pd.concat([df, dummies], axis=1)
#             df = df.drop(columns=[column_name])
#             msg = f"One-hot encoded '{column_name}' into {len(dummies.columns)} columns"
#         elif encoding_type == "label":
#             le = LabelEncoder()
#             df[f"{column_name}_encoded"] = le.fit_transform(df[column_name].fillna('missing'))
#             mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#             msg = f"Label encoded '{column_name}'. Mapping: {mapping}"
#         else:
#             return f"Unknown encoding type: {encoding_type}"
        
#         output_path = file_path.replace('.csv', '_encoded.csv')
#         df.to_csv(output_path, index=False)
        
#         return f"{msg}. Saved to: {output_path}"
#     except Exception as e:
#         return f"Error encoding: {str(e)}"


# @tool
# def create_new_feature(file_path: str, new_column: str, column1: str, operation: str, column2_or_value: Union[str, float]) -> str:
#     """
#     Create a new feature from existing columns.
    
#     Args:
#         file_path: Path to the CSV file
#         new_column: Name for the new feature
#         column1: First column name
#         operation: Operation to perform (+, -, *, /, >, <, ==)
#         column2_or_value: Second column name or numeric value
        
#     Returns:
#         String confirming feature creation
#     """
#     try:
#         df = pd.read_csv(file_path)
        
#         if column1 not in df.columns:
#             return f"Column '{column1}' not found"
        
#         # Check if column2_or_value is a column name or a value
#         if isinstance(column2_or_value, str) and column2_or_value in df.columns:
#             operand2 = df[column2_or_value]
#         else:
#             try:
#                 operand2 = float(column2_or_value)
#             except:
#                 operand2 = column2_or_value
        
#         operand1 = df[column1]
        
#         if operation == '+':
#             df[new_column] = operand1 + operand2
#         elif operation == '-':
#             df[new_column] = operand1 - operand2
#         elif operation == '*':
#             df[new_column] = operand1 * operand2
#         elif operation == '/':
#             df[new_column] = operand1 / operand2
#         elif operation == '>':
#             df[new_column] = (operand1 > operand2).astype(int)
#         elif operation == '<':
#             df[new_column] = (operand1 < operand2).astype(int)
#         elif operation == '==':
#             df[new_column] = (operand1 == operand2).astype(int)
#         else:
#             return f"Unknown operation: {operation}"
        
#         output_path = file_path.replace('.csv', '_new_feature.csv')
#         df.to_csv(output_path, index=False)
        
#         return f"Created new feature '{new_column}' using {column1} {operation} {column2_or_value}. Saved to: {output_path}"
#     except Exception as e:
#         return f"Error creating feature: {str(e)}"


# @tool
# def normalize_column(file_path: str, column_name: str, method: str = "minmax") -> str:
#     """
#     Normalize a numeric column.
    
#     Args:
#         file_path: Path to the CSV file
#         column_name: Name of column to normalize
#         method: 'minmax' (0-1) or 'zscore' (standardization)
        
#     Returns:
#         String confirming normalization
#     """
#     try:
#         df = pd.read_csv(file_path)
#         if column_name not in df.columns:
#             return f"Column '{column_name}' not found"
        
#         col = df[column_name].values.reshape(-1, 1)
        
#         if method == "minmax":
#             scaler = MinMaxScaler()
#             df[f"{column_name}_normalized"] = scaler.fit_transform(col)
#             msg = f"Min-max normalized '{column_name}' to range [0, 1]"
#         elif method == "zscore":
#             scaler = StandardScaler()
#             df[f"{column_name}_standardized"] = scaler.fit_transform(col)
#             msg = f"Z-score standardized '{column_name}' (mean=0, std=1)"
#         else:
#             return f"Unknown method: {method}"
        
#         output_path = file_path.replace('.csv', '_normalized.csv')
#         df.to_csv(output_path, index=False)
        
#         return f"{msg}. Saved to: {output_path}"
#     except Exception as e:
#         return f"Error normalizing: {str(e)}"


# # ============================================================================
# # DATA FILTERING AND SELECTION TOOLS
# # ============================================================================

# @tool
# def filter_rows_numeric(file_path: str, column_name: str, operator: str, value: float) -> str:
#     """
#     Filter rows based on numeric condition.
    
#     Args:
#         file_path: Path to the CSV file
#         column_name: Column to filter on
#         operator: Comparison operator (>, <, >=, <=, ==, !=)
#         value: Numeric value to compare
        
#     Returns:
#         String with filtering results
#     """
#     try:
#         df = pd.read_csv(file_path)
#         if column_name not in df.columns:
#             return f"Column '{column_name}' not found"
        
#         original_len = len(df)
        
#         if operator == '>':
#             df_filtered = df[df[column_name] > value]
#         elif operator == '<':
#             df_filtered = df[df[column_name] < value]
#         elif operator == '>=':
#             df_filtered = df[df[column_name] >= value]
#         elif operator == '<=':
#             df_filtered = df[df[column_name] <= value]
#         elif operator == '==':
#             df_filtered = df[df[column_name] == value]
#         elif operator == '!=':
#             df_filtered = df[df[column_name] != value]
#         else:
#             return f"Unknown operator: {operator}"
        
#         output_path = file_path.replace('.csv', '_filtered.csv')
#         df_filtered.to_csv(output_path, index=False)
        
#         return f"Filtered {original_len} -> {len(df_filtered)} rows where {column_name} {operator} {value}. Saved to: {output_path}"
#     except Exception as e:
#         return f"Error filtering: {str(e)}"


# @tool
# def filter_rows_categorical(file_path: str, column_name: str, values: str, include: bool = True) -> str:
#     """
#     Filter rows based on categorical values.
    
#     Args:
#         file_path: Path to the CSV file
#         column_name: Column to filter on
#         values: Comma-separated list of values
#         include: If True, keep rows with these values. If False, exclude them
        
#     Returns:
#         String with filtering results
#     """
#     try:
#         df = pd.read_csv(file_path)
#         if column_name not in df.columns:
#             return f"Column '{column_name}' not found"
        
#         value_list = [v.strip() for v in values.split(',')]
#         original_len = len(df)
        
#         if include:
#             df_filtered = df[df[column_name].isin(value_list)]
#             action = "included"
#         else:
#             df_filtered = df[~df[column_name].isin(value_list)]
#             action = "excluded"
        
#         output_path = file_path.replace('.csv', '_filtered.csv')
#         df_filtered.to_csv(output_path, index=False)
        
#         return f"Filtered {original_len} -> {len(df_filtered)} rows. {action}: {value_list}. Saved to: {output_path}"
#     except Exception as e:
#         return f"Error filtering: {str(e)}"


# @tool
# def select_columns(file_path: str, columns: str) -> str:
#     """
#     Select specific columns from the dataset.
    
#     Args:
#         file_path: Path to the CSV file
#         columns: Comma-separated list of column names to keep
        
#     Returns:
#         String confirming column selection
#     """
#     try:
#         df = pd.read_csv(file_path)
#         col_list = [c.strip() for c in columns.split(',')]
        
#         missing = [c for c in col_list if c not in df.columns]
#         if missing:
#             return f"Columns not found: {missing}"
        
#         df_selected = df[col_list]
        
#         output_path = file_path.replace('.csv', '_selected.csv')
#         df_selected.to_csv(output_path, index=False)
        
#         return f"Selected {len(col_list)} columns. New shape: {df_selected.shape}. Saved to: {output_path}"
#     except Exception as e:
#         return f"Error selecting columns: {str(e)}"


# # ============================================================================
# # STATISTICAL ANALYSIS TOOLS
# # ============================================================================

# @tool
# def calculate_correlation(file_path: str, column1: str, column2: str, method: str = "pearson") -> str:
#     """
#     Calculate correlation between two numeric columns.
    
#     Args:
#         file_path: Path to the CSV file
#         column1: First column name
#         column2: Second column name
#         method: 'pearson' or 'spearman'
        
#     Returns:
#         String with correlation coefficient and p-value
#     """
#     try:
#         df = pd.read_csv(file_path)
        
#         for col in [column1, column2]:
#             if col not in df.columns:
#                 return f"Column '{col}' not found"
        
#         # Remove rows with missing values
#         data = df[[column1, column2]].dropna()
        
#         if method == "pearson":
#             corr, p_value = pearsonr(data[column1], data[column2])
#         elif method == "spearman":
#             corr, p_value = spearmanr(data[column1], data[column2])
#         else:
#             return f"Unknown method: {method}"
        
#         interpretation = ""
#         abs_corr = abs(corr)
#         if abs_corr < 0.3:
#             strength = "weak"
#         elif abs_corr < 0.7:
#             strength = "moderate"
#         else:
#             strength = "strong"
        
#         direction = "positive" if corr > 0 else "negative"
        
#         return f"{method.capitalize()} correlation between '{column1}' and '{column2}': {corr:.4f} (p-value: {p_value:.4f}). This is a {strength} {direction} correlation."
#     except Exception as e:
#         return f"Error calculating correlation: {str(e)}"


# @tool
# def perform_ttest(file_path: str, column_name: str, group_column: str) -> str:
#     """
#     Perform t-test between two groups.
    
#     Args:
#         file_path: Path to the CSV file
#         column_name: Numeric column to test
#         group_column: Column with two groups
        
#     Returns:
#         String with t-statistic and p-value
#     """
#     try:
#         df = pd.read_csv(file_path)
        
#         if column_name not in df.columns or group_column not in df.columns:
#             return "Column not found"
        
#         groups = df[group_column].unique()
#         if len(groups) != 2:
#             return f"T-test requires exactly 2 groups. Found {len(groups)}: {groups}"
        
#         group1 = df[df[group_column] == groups[0]][column_name].dropna()
#         group2 = df[df[group_column] == groups[1]][column_name].dropna()
        
#         t_stat, p_value = stats.ttest_ind(group1, group2)
        
#         mean1, mean2 = group1.mean(), group2.mean()
        
#         sig = "statistically significant" if p_value < 0.05 else "not statistically significant"
        
#         return f"T-test for '{column_name}' between groups:\n  {groups[0]}: mean={mean1:.2f}, n={len(group1)}\n  {groups[1]}: mean={mean2:.2f}, n={len(group2)}\n  t-statistic: {t_stat:.4f}\n  p-value: {p_value:.4f}\n  Result: {sig} (α=0.05)"
#     except Exception as e:
#         return f"Error performing t-test: {str(e)}"


# @tool
# def chi_square_test(file_path: str, column1: str, column2: str) -> str:
#     """
#     Perform chi-square test between two categorical variables.
    
#     Args:
#         file_path: Path to the CSV file
#         column1: First categorical column
#         column2: Second categorical column
        
#     Returns:
#         String with chi-square statistic and p-value
#     """
#     try:
#         df = pd.read_csv(file_path)
        
#         if column1 not in df.columns or column2 not in df.columns:
#             return "Column not found"
        
#         # Create contingency table
#         contingency = pd.crosstab(df[column1], df[column2])
        
#         chi2, p_value, dof, expected = chi2_contingency(contingency)
        
#         sig = "dependent" if p_value < 0.05 else "independent"
        
#         return f"Chi-square test between '{column1}' and '{column2}':\n  Chi-square statistic: {chi2:.4f}\n  p-value: {p_value:.4f}\n  Degrees of freedom: {dof}\n  Result: Variables are {sig} (α=0.05)"
#     except Exception as e:
#         return f"Error performing chi-square test: {str(e)}"


# @tool
# def calculate_group_statistics(file_path: str, value_column: str, group_column: str) -> str:
#     """
#     Calculate statistics for each group.
    
#     Args:
#         file_path: Path to the CSV file
#         value_column: Numeric column to analyze
#         group_column: Column to group by
        
#     Returns:
#         String with statistics for each group
#     """
#     try:
#         df = pd.read_csv(file_path)
        
#         if value_column not in df.columns or group_column not in df.columns:
#             return "Column not found"
        
#         grouped = df.groupby(group_column)[value_column].agg([
#             'count', 'mean', 'std', 'min', 'max'
#         ])
        
#         result = f"Group statistics for '{value_column}' by '{group_column}':\n"
#         result += grouped.to_string()
        
#         return result
#     except Exception as e:
#         return f"Error calculating group statistics: {str(e)}"


# # ============================================================================
# # DATA VISUALIZATION TOOLS
# # ============================================================================

# @tool
# def create_histogram(file_path: str, column_name: str, bins: int = 20) -> str:
#     """
#     Create histogram for a numeric column.
    
#     Args:
#         file_path: Path to the CSV file
#         column_name: Column to plot
#         bins: Number of bins
        
#     Returns:
#         String confirming plot creation
#     """
#     try:
#         df = pd.read_csv(file_path)
#         if column_name not in df.columns:
#             return f"Column '{column_name}' not found"
        
#         plt.figure(figsize=(10, 6))
#         plt.hist(df[column_name].dropna(), bins=bins, edgecolor='black', alpha=0.7)
#         plt.title(f'Histogram of {column_name}')
#         plt.xlabel(column_name)
#         plt.ylabel('Frequency')
#         plt.grid(True, alpha=0.3)
        
#         output_path = file_path.replace('.csv', f'_histogram_{column_name}.png')
#         plt.savefig(output_path, dpi=100, bbox_inches='tight')
#         plt.close()
        
#         # Calculate statistics
#         data = df[column_name].dropna()
#         stats_info = f"Mean: {data.mean():.2f}, Median: {data.median():.2f}, Std: {data.std():.2f}"
        
#         return f"Histogram created for '{column_name}'. {stats_info}. Saved to: {output_path}"
#     except Exception as e:
#         return f"Error creating histogram: {str(e)}"


# @tool
# def create_scatter_plot(file_path: str, x_column: str, y_column: str, color_column: Optional[str] = None) -> str:
#     """
#     Create scatter plot between two numeric columns.
    
#     Args:
#         file_path: Path to the CSV file
#         x_column: Column for x-axis
#         y_column: Column for y-axis
#         color_column: Optional column for color coding
        
#     Returns:
#         String confirming plot creation
#     """
#     try:
#         df = pd.read_csv(file_path)
        
#         for col in [x_column, y_column]:
#             if col not in df.columns:
#                 return f"Column '{col}' not found"
        
#         plt.figure(figsize=(10, 6))
        
#         if color_column and color_column in df.columns:
#             for category in df[color_column].unique():
#                 mask = df[color_column] == category
#                 plt.scatter(df[mask][x_column], df[mask][y_column], label=category, alpha=0.6)
#             plt.legend()
#         else:
#             plt.scatter(df[x_column], df[y_column], alpha=0.6)
        
#         plt.title(f'Scatter Plot: {x_column} vs {y_column}')
#         plt.xlabel(x_column)
#         plt.ylabel(y_column)
#         plt.grid(True, alpha=0.3)
        
#         output_path = file_path.replace('.csv', f'_scatter_{x_column}_{y_column}.png')
#         plt.savefig(output_path, dpi=100, bbox_inches='tight')
#         plt.close()
        
#         # Calculate correlation
#         corr = df[[x_column, y_column]].corr().iloc[0, 1]
        
#         return f"Scatter plot created. Correlation: {corr:.3f}. Saved to: {output_path}"
#     except Exception as e:
#         return f"Error creating scatter plot: {str(e)}"


# @tool
# def create_bar_chart(file_path: str, column_name: str, top_n: int = 10) -> str:
#     """
#     Create bar chart for categorical column.
    
#     Args:
#         file_path: Path to the CSV file
#         column_name: Categorical column to plot
#         top_n: Number of top categories to show
        
#     Returns:
#         String confirming plot creation
#     """
#     try:
#         df = pd.read_csv(file_path)
#         if column_name not in df.columns:
#             return f"Column '{column_name}' not found"
        
#         value_counts = df[column_name].value_counts().head(top_n)
        
#         plt.figure(figsize=(10, 6))
#         value_counts.plot(kind='bar', edgecolor='black', alpha=0.7)
#         plt.title(f'Bar Chart: Top {top_n} values in {column_name}')
#         plt.xlabel(column_name)
#         plt.ylabel('Count')
#         plt.xticks(rotation=45, ha='right')
#         plt.grid(True, alpha=0.3, axis='y')
        
#         output_path = file_path.replace('.csv', f'_bar_{column_name}.png')
#         plt.savefig(output_path, dpi=100, bbox_inches='tight')
#         plt.close()
        
#         return f"Bar chart created for '{column_name}'. Top value: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences). Saved to: {output_path}"
#     except Exception as e:
#         return f"Error creating bar chart: {str(e)}"


# @tool
# def create_correlation_heatmap(file_path: str, columns: Optional[str] = None) -> str:
#     """
#     Create correlation heatmap for numeric columns.
    
#     Args:
#         file_path: Path to the CSV file
#         columns: Optional comma-separated list of columns (if None, uses all numeric)
        
#     Returns:
#         String confirming plot creation
#     """
#     try:
#         df = pd.read_csv(file_path)
        
#         if columns:
#             col_list = [c.strip() for c in columns.split(',')]
#             df_numeric = df[col_list]
#         else:
#             df_numeric = df.select_dtypes(include=[np.number])
        
#         if df_numeric.shape[1] < 2:
#             return "Need at least 2 numeric columns for correlation heatmap"
        
#         plt.figure(figsize=(12, 8))
#         corr_matrix = df_numeric.corr()
        
#         sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
#                     square=True, linewidths=1, cbar_kws={"shrink": 0.8})
#         plt.title('Correlation Heatmap')
        
#         output_path = file_path.replace('.csv', '_correlation_heatmap.png')
#         plt.savefig(output_path, dpi=100, bbox_inches='tight')
#         plt.close()
        
#         # Find strongest correlations
#         upper_tri = np.triu(np.ones_like(corr_matrix), k=1)
#         upper_tri_corr = corr_matrix.where(upper_tri.astype(bool))
#         strongest = upper_tri_corr.abs().unstack().nlargest(3)
        
#         return f"Correlation heatmap created for {corr_matrix.shape[0]} variables. Saved to: {output_path}"
#     except Exception as e:
#         return f"Error creating heatmap: {str(e)}"


# # ============================================================================
# # ENHANCED MACHINE LEARNING TOOLS
# # ============================================================================

# @tool
# def train_random_forest_model(file_path: str, target_column: str, feature_columns: str, task_type: str = "classification") -> str:
#     """
#     Train a Random Forest model.

#     Args:
#         file_path: Path to the data file
#         target_column: Name of the target variable column
#         feature_columns: Comma-separated list of feature columns
#         task_type: Type of task (classification, regression)

#     Returns:
#         String describing the model training results including feature importance
#     """
#     try:
#         df = pd.read_csv(file_path)

#         # Parse feature columns
#         features = [col.strip() for col in feature_columns.split(',')]

#         # Check if columns exist
#         for col in features + [target_column]:
#             if col not in df.columns:
#                 return f"Column '{col}' not found in dataset"

#         # Prepare data
#         X = df[features]
#         y = df[target_column]

#         # Handle missing values
#         X = X.fillna(X.mean())
#         if task_type == "classification":
#             y = y.fillna(y.mode()[0])
#         else:
#             y = y.fillna(y.mean())

#         # Handle categorical variables
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].astype(str))

#         if task_type == "classification" and y.dtype == 'object':
#             le_target = LabelEncoder()
#             y = le_target.fit_transform(y.astype(str))

#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Train model
#         if task_type == "classification":
#             model = RandomForestClassifier(n_estimators=100, random_state=42)
#             scoring_metric = 'accuracy'
#         else:
#             model = RandomForestRegressor(n_estimators=100, random_state=42)
#             scoring_metric = 'r2'

#         model.fit(X_train, y_train)

#         # Make predictions
#         y_pred = model.predict(X_test)

#         # Calculate metrics
#         if task_type == "classification":
#             accuracy = accuracy_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
#             recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
#             f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

#             metrics = {
#                 "accuracy": round(accuracy, 4),
#                 "precision": round(precision, 4),
#                 "recall": round(recall, 4),
#                 "f1_score": round(f1, 4)
#             }
#         else:
#             mse = mean_squared_error(y_test, y_pred)
#             r2 = r2_score(y_test, y_pred)

#             metrics = {
#                 "mse": round(mse, 4),
#                 "r2_score": round(r2, 4)
#             }

#         # Feature importance
#         feature_importance = dict(zip(features, [round(imp, 4) for imp in model.feature_importances_]))

#         # Cross-validation
#         cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring_metric)

#         # Save model
#         model_path = file_path.replace('.csv', f'_rf_{task_type}_model.joblib')
#         joblib.dump(model, model_path)

#         results = {
#             "model_type": f"Random Forest {task_type}",
#             "features": features,
#             "target": target_column,
#             "train_size": len(X_train),
#             "test_size": len(X_test),
#             **metrics,
#             "cv_score_mean": round(cv_scores.mean(), 4),
#             "cv_score_std": round(cv_scores.std(), 4),
#             "feature_importance": feature_importance,
#             "model_saved": model_path
#         }

#         return f"Random Forest model trained: {results}"

#     except Exception as e:
#         return f"Error training Random Forest model: {str(e)}"


# @tool
# def train_knn_model(file_path: str, target_column: str, feature_columns: str, task_type: str = "classification", n_neighbors: int = 5) -> str:
#     """
#     Train a K-Nearest Neighbors model.

#     Args:
#         file_path: Path to the data file
#         target_column: Name of the target variable column
#         feature_columns: Comma-separated list of feature columns
#         task_type: Type of task (classification, regression)
#         n_neighbors: Number of neighbors to use

#     Returns:
#         String describing the model training results
#     """
#     try:
#         df = pd.read_csv(file_path)

#         # Parse feature columns
#         features = [col.strip() for col in feature_columns.split(',')]

#         # Check if columns exist
#         for col in features + [target_column]:
#             if col not in df.columns:
#                 return f"Column '{col}' not found in dataset"

#         # Prepare data
#         X = df[features]
#         y = df[target_column]

#         # Handle missing values
#         X = X.fillna(X.mean())
#         if task_type == "classification":
#             y = y.fillna(y.mode()[0])
#         else:
#             y = y.fillna(y.mean())

#         # Handle categorical variables
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].astype(str))

#         if task_type == "classification" and y.dtype == 'object':
#             le_target = LabelEncoder()
#             y = le_target.fit_transform(y.astype(str))

#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Train model
#         if task_type == "classification":
#             model = KNeighborsClassifier(n_neighbors=n_neighbors)
#             scoring_metric = 'accuracy'
#         else:
#             model = KNeighborsRegressor(n_neighbors=n_neighbors)
#             scoring_metric = 'r2'

#         model.fit(X_train, y_train)

#         # Make predictions
#         y_pred = model.predict(X_test)

#         # Calculate metrics
#         if task_type == "classification":
#             accuracy = accuracy_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
#             recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
#             f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

#             metrics = {
#                 "accuracy": round(accuracy, 4),
#                 "precision": round(precision, 4),
#                 "recall": round(recall, 4),
#                 "f1_score": round(f1, 4)
#             }
#         else:
#             mse = mean_squared_error(y_test, y_pred)
#             r2 = r2_score(y_test, y_pred)

#             metrics = {
#                 "mse": round(mse, 4),
#                 "r2_score": round(r2, 4)
#             }

#         # Cross-validation
#         cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring_metric)

#         # Save model
#         model_path = file_path.replace('.csv', f'_knn_{task_type}_model.joblib')
#         joblib.dump(model, model_path)

#         results = {
#             "model_type": f"KNN {task_type}",
#             "n_neighbors": n_neighbors,
#             "features": features,
#             "target": target_column,
#             "train_size": len(X_train),
#             "test_size": len(X_test),
#             **metrics,
#             "cv_score_mean": round(cv_scores.mean(), 4),
#             "cv_score_std": round(cv_scores.std(), 4),
#             "model_saved": model_path
#         }

#         return f"KNN model trained: {results}"

#     except Exception as e:
#         return f"Error training KNN model: {str(e)}"


# @tool
# def train_svm_model(file_path: str, target_column: str, feature_columns: str, task_type: str = "classification") -> str:
#     """
#     Train a Support Vector Machine model.

#     Args:
#         file_path: Path to the data file
#         target_column: Name of the target variable column
#         feature_columns: Comma-separated list of feature columns
#         task_type: Type of task (classification, regression)

#     Returns:
#         String describing the model training results
#     """
#     try:
#         df = pd.read_csv(file_path)

#         # Parse feature columns
#         features = [col.strip() for col in feature_columns.split(',')]

#         # Check if columns exist
#         for col in features + [target_column]:
#             if col not in df.columns:
#                 return f"Column '{col}' not found in dataset"

#         # Prepare data
#         X = df[features]
#         y = df[target_column]

#         # Handle missing values
#         X = X.fillna(X.mean())
#         if task_type == "classification":
#             y = y.fillna(y.mode()[0])
#         else:
#             y = y.fillna(y.mean())

#         # Handle categorical variables
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].astype(str))

#         if task_type == "classification" and y.dtype == 'object':
#             le_target = LabelEncoder()
#             y = le_target.fit_transform(y.astype(str))

#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Train model
#         if task_type == "classification":
#             model = SVC(kernel='rbf', random_state=42)
#             scoring_metric = 'accuracy'
#         else:
#             model = SVR(kernel='rbf')
#             scoring_metric = 'r2'

#         model.fit(X_train, y_train)

#         # Make predictions
#         y_pred = model.predict(X_test)

#         # Calculate metrics
#         if task_type == "classification":
#             accuracy = accuracy_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
#             recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
#             f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

#             metrics = {
#                 "accuracy": round(accuracy, 4),
#                 "precision": round(precision, 4),
#                 "recall": round(recall, 4),
#                 "f1_score": round(f1, 4)
#             }
#         else:
#             mse = mean_squared_error(y_test, y_pred)
#             r2 = r2_score(y_test, y_pred)

#             metrics = {
#                 "mse": round(mse, 4),
#                 "r2_score": round(r2, 4)
#             }

#         # Cross-validation
#         cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring_metric)

#         # Save model
#         model_path = file_path.replace('.csv', f'_svm_{task_type}_model.joblib')
#         joblib.dump(model, model_path)

#         results = {
#             "model_type": f"SVM {task_type}",
#             "features": features,
#             "target": target_column,
#             "train_size": len(X_train),
#             "test_size": len(X_test),
#             **metrics,
#             "cv_score_mean": round(cv_scores.mean(), 4),
#             "cv_score_std": round(cv_scores.std(), 4),
#             "model_saved": model_path
#         }

#         return f"SVM model trained: {results}"

#     except Exception as e:
#         return f"Error training SVM model: {str(e)}"


# @tool
# def evaluate_model(model_path: str, test_data_path: str, target_column: str, feature_columns: str) -> str:
#     """
#     Evaluate a trained model on new test data.

#     Args:
#         model_path: Path to the saved model file
#         test_data_path: Path to the test data file
#         target_column: Name of the target variable column
#         feature_columns: Comma-separated list of feature columns

#     Returns:
#         String describing the model evaluation results
#     """
#     try:
#         # Load the model
#         model = joblib.load(model_path)

#         # Load test data
#         df = pd.read_csv(test_data_path)

#         # Parse feature columns
#         features = [col.strip() for col in feature_columns.split(',')]

#         # Check if columns exist
#         for col in features + [target_column]:
#             if col not in df.columns:
#                 return f"Column '{col}' not found in test dataset"

#         # Prepare test data
#         X = df[features]
#         y = df[target_column]

#         # Handle missing values
#         X = X.fillna(X.mean())

#         # Handle categorical variables (same as training)
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].astype(str))

#         # Make predictions
#         y_pred = model.predict(X)

#         # Determine if this is classification or regression based on model type
#         model_name = str(type(model).__name__)
#         is_classification = any(clf in model_name for clf in ['Classifier', 'SVC'])

#         # Calculate appropriate metrics
#         if is_classification:
#             accuracy = accuracy_score(y, y_pred)
#             precision = precision_score(y, y_pred, average='weighted', zero_division=0)
#             recall = recall_score(y, y_pred, average='weighted', zero_division=0)
#             f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

#             results = {
#                 "model_type": model_name,
#                 "task": "classification",
#                 "test_samples": len(X),
#                 "accuracy": round(accuracy, 4),
#                 "precision": round(precision, 4),
#                 "recall": round(recall, 4),
#                 "f1_score": round(f1, 4)
#             }
#         else:
#             mse = mean_squared_error(y, y_pred)
#             r2 = r2_score(y, y_pred)
#             rmse = np.sqrt(mse)

#             results = {
#                 "model_type": model_name,
#                 "task": "regression",
#                 "test_samples": len(X),
#                 "mse": round(mse, 4),
#                 "rmse": round(rmse, 4),
#                 "r2_score": round(r2, 4)
#             }

#         return f"Model evaluation results: {results}"

#     except Exception as e:
#         return f"Error evaluating model: {str(e)}"



# @tool
# def train_logistic_regression(file_path: str, target_column: str, feature_columns: str) -> str:
#     """
#     Train logistic regression for binary classification.
    
#     Args:
#         file_path: Path to the CSV file
#         target_column: Target variable column
#         feature_columns: Comma-separated list of feature columns
        
#     Returns:
#         String with model performance metrics
#     """
#     try:
#         df = pd.read_csv(file_path)
#         features = [col.strip() for col in feature_columns.split(',')]
        
#         # Prepare data
#         X = df[features]
#         y = df[target_column]
        
#         # Handle categorical features
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].fillna('missing'))
        
#         # Handle missing values
#         X = X.fillna(X.mean())
        
#         # Encode target if categorical
#         if y.dtype == 'object':
#             le_target = LabelEncoder()
#             y = le_target.fit_transform(y)
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Train model
#         model = LogisticRegression(max_iter=1000, random_state=42)
#         model.fit(X_train, y_train)
        
#         # Predictions
#         y_pred = model.predict(X_test)
#         y_proba = model.predict_proba(X_test)[:, 1]
        
#         # Metrics
#         acc = accuracy_score(y_test, y_pred)
#         prec = precision_score(y_test, y_pred, average='binary')
#         rec = recall_score(y_test, y_pred, average='binary')
#         f1 = f1_score(y_test, y_pred, average='binary')
        
#         # Check if binary classification for AUC
#         if len(np.unique(y)) == 2:
#             auc = roc_auc_score(y_test, y_proba)
#             auc_str = f", AUC: {auc:.3f}"
#         else:
#             auc_str = ""
        
#         # Feature importance (coefficients)
#         importance = dict(zip(features, model.coef_[0]))
#         top_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
#         # Save model
#         model_path = file_path.replace('.csv', '_logistic_model.joblib')
#         joblib.dump(model, model_path)
        
#         return f"Logistic Regression trained:\n  Accuracy: {acc:.3f}\n  Precision: {prec:.3f}\n  Recall: {rec:.3f}\n  F1: {f1:.3f}{auc_str}\n  Top features: {top_features}\n  Model saved: {model_path}"
#     except Exception as e:
#         return f"Error training logistic regression: {str(e)}"


# @tool
# def train_decision_tree(file_path: str, target_column: str, feature_columns: str, max_depth: int = 5) -> str:
#     """
#     Train decision tree classifier.
    
#     Args:
#         file_path: Path to the CSV file
#         target_column: Target variable column
#         feature_columns: Comma-separated list of feature columns
#         max_depth: Maximum depth of the tree
        
#     Returns:
#         String with model performance and feature importance
#     """
#     try:
#         df = pd.read_csv(file_path)
#         features = [col.strip() for col in feature_columns.split(',')]
        
#         # Prepare data
#         X = df[features]
#         y = df[target_column]
        
#         # Handle categorical features
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].fillna('missing'))
        
#         X = X.fillna(X.mean())
        
#         if y.dtype == 'object':
#             le_target = LabelEncoder()
#             y = le_target.fit_transform(y)
        
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Train model
#         model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
#         model.fit(X_train, y_train)
        
#         # Predictions
#         y_pred = model.predict(X_test)
        
#         # Metrics
#         acc = accuracy_score(y_test, y_pred)
        
#         # Feature importance
#         importance = dict(zip(features, model.feature_importances_))
#         top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
#         # Save model
#         model_path = file_path.replace('.csv', '_decision_tree.joblib')
#         joblib.dump(model, model_path)
        
#         return f"Decision Tree trained:\n  Accuracy: {acc:.3f}\n  Max depth: {max_depth}\n  Top features: {top_features}\n  Model saved: {model_path}"
#     except Exception as e:
#         return f"Error training decision tree: {str(e)}"


# @tool
# def train_gradient_boosting(file_path: str, target_column: str, feature_columns: str, n_estimators: int = 100) -> str:
#     """
#     Train gradient boosting classifier.
    
#     Args:
#         file_path: Path to the CSV file
#         target_column: Target variable column
#         feature_columns: Comma-separated list of feature columns
#         n_estimators: Number of boosting stages
        
#     Returns:
#         String with model performance
#     """
#     try:
#         df = pd.read_csv(file_path)
#         features = [col.strip() for col in feature_columns.split(',')]
        
#         # Prepare data
#         X = df[features]
#         y = df[target_column]
        
#         # Handle categorical features
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].fillna('missing'))
        
#         X = X.fillna(X.mean())
        
#         if y.dtype == 'object':
#             le_target = LabelEncoder()
#             y = le_target.fit_transform(y)
        
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Train model
#         model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
#         model.fit(X_train, y_train)
        
#         # Predictions
#         y_pred = model.predict(X_test)
        
#         # Metrics
#         acc = accuracy_score(y_test, y_pred)
        
#         # Feature importance
#         importance = dict(zip(features, model.feature_importances_))
#         top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
#         # Save model
#         model_path = file_path.replace('.csv', '_gradient_boosting.joblib')
#         joblib.dump(model, model_path)
        
#         return f"Gradient Boosting trained:\n  Accuracy: {acc:.3f}\n  N estimators: {n_estimators}\n  Top features: {top_features}\n  Model saved: {model_path}"
#     except Exception as e:
#         return f"Error training gradient boosting: {str(e)}"


# @tool
# def make_prediction(model_path: str, data_path: str, feature_columns: str) -> str:
#     """
#     Make predictions using a trained model.
    
#     Args:
#         model_path: Path to saved model
#         data_path: Path to data for prediction
#         feature_columns: Comma-separated list of feature columns
        
#     Returns:
#         String with predictions
#     """
#     try:
#         model = joblib.load(model_path)
#         df = pd.read_csv(data_path)
#         features = [col.strip() for col in feature_columns.split(',')]
        
#         X = df[features]
        
#         # Handle categorical features
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].fillna('missing'))
        
#         X = X.fillna(X.mean())
        
#         # Make predictions
#         predictions = model.predict(X)
        
#         # Add predictions to dataframe
#         df['prediction'] = predictions
        
#         # If probabilistic model, add probabilities
#         if hasattr(model, 'predict_proba'):
#             probas = model.predict_proba(X)
#             if probas.shape[1] == 2:
#                 df['probability'] = probas[:, 1]
        
#         output_path = data_path.replace('.csv', '_predictions.csv')
#         df.to_csv(output_path, index=False)
        
#         # Summary
#         unique, counts = np.unique(predictions, return_counts=True)
#         pred_summary = dict(zip(unique, counts))
        
#         return f"Predictions made for {len(predictions)} samples:\n  Distribution: {pred_summary}\n  Results saved to: {output_path}"
#     except Exception as e:
#         return f"Error making predictions: {str(e)}"


# @tool
# def perform_cross_validation(file_path: str, target_column: str, feature_columns: str, model_type: str = "random_forest", cv_folds: int = 5) -> str:
#     """
#     Perform cross-validation for model evaluation.
    
#     Args:
#         file_path: Path to the CSV file
#         target_column: Target variable column
#         feature_columns: Comma-separated list of feature columns
#         model_type: Type of model (random_forest, logistic, svm)
#         cv_folds: Number of cross-validation folds
        
#     Returns:
#         String with cross-validation scores
#     """
#     try:
#         df = pd.read_csv(file_path)
#         features = [col.strip() for col in feature_columns.split(',')]
        
#         # Prepare data
#         X = df[features]
#         y = df[target_column]
        
#         # Handle categorical features
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].fillna('missing'))
        
#         X = X.fillna(X.mean())
        
#         if y.dtype == 'object':
#             le_target = LabelEncoder()
#             y = le_target.fit_transform(y)
        
#         # Select model
#         if model_type == "random_forest":
#             model = RandomForestClassifier(n_estimators=100, random_state=42)
#         elif model_type == "logistic":
#             model = LogisticRegression(max_iter=1000, random_state=42)
#         elif model_type == "svm":
#             model = SVC(random_state=42)
#         else:
#             return f"Unknown model type: {model_type}"
        
#         # Perform cross-validation
#         cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        
#         return f"Cross-validation results ({cv_folds} folds):\n  Model: {model_type}\n  Scores: {cv_scores.round(3)}\n  Mean: {cv_scores.mean():.3f}\n  Std: {cv_scores.std():.3f}\n  Min: {cv_scores.min():.3f}\n  Max: {cv_scores.max():.3f}"
#     except Exception as e:
#         return f"Error in cross-validation: {str(e)}"


# @tool
# def feature_selection(file_path: str, target_column: str, feature_columns: str, n_features: int = 5) -> str:
#     """
#     Select top features based on importance.
    
#     Args:
#         file_path: Path to the CSV file
#         target_column: Target variable column
#         feature_columns: Comma-separated list of feature columns
#         n_features: Number of top features to select
        
#     Returns:
#         String with selected features and their scores
#     """
#     try:
#         df = pd.read_csv(file_path)
#         features = [col.strip() for col in feature_columns.split(',')]
        
#         # Prepare data
#         X = df[features]
#         y = df[target_column]
        
#         # Handle categorical features
#         for col in X.columns:
#             if X[col].dtype == 'object':
#                 le = LabelEncoder()
#                 X[col] = le.fit_transform(X[col].fillna('missing'))
        
#         X = X.fillna(X.mean())
        
#         if y.dtype == 'object':
#             le_target = LabelEncoder()
#             y = le_target.fit_transform(y)
        
#         # Feature selection
#         selector = SelectKBest(score_func=f_classif, k=min(n_features, len(features)))
#         X_selected = selector.fit_transform(X, y)
        
#         # Get selected features
#         selected_mask = selector.get_support()
#         selected_features = [f for f, s in zip(features, selected_mask) if s]
        
#         # Get scores
#         feature_scores = dict(zip(features, selector.scores_))
#         top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
        
#         # Save selected features dataset
#         df_selected = pd.concat([df[selected_features], df[target_column]], axis=1)
#         output_path = file_path.replace('.csv', '_selected_features.csv')
#         df_selected.to_csv(output_path, index=False)
        
#         return f"Feature selection complete:\n  Selected {len(selected_features)} features: {selected_features}\n  Top scores: {top_features}\n  Saved to: {output_path}"
#     except Exception as e:
#         return f"Error in feature selection: {str(e)}"


# # ============================================================================
# # ANSWER GENERATION TOOLS
# # ============================================================================

# @tool
# def answer_survival_question(file_path: str, passenger_info: str) -> str:
#     """
#     Answer survival questions about Titanic passengers.
    
#     Args:
#         file_path: Path to the CSV file
#         passenger_info: Description of passenger (e.g., "male, age 30, first class")
        
#     Returns:
#         String with survival prediction and analysis
#     """
#     try:
#         df = pd.read_csv(file_path)
        
#         # Parse passenger info
#         info_lower = passenger_info.lower()
        
#         # Extract features
#         conditions = []
        
#         if 'male' in info_lower or 'man' in info_lower:
#             conditions.append(df['Sex'] == 'male')
#         elif 'female' in info_lower or 'woman' in info_lower:
#             conditions.append(df['Sex'] == 'female')
        
#         if 'first class' in info_lower or 'class 1' in info_lower:
#             conditions.append(df['Pclass'] == 1)
#         elif 'second class' in info_lower or 'class 2' in info_lower:
#             conditions.append(df['Pclass'] == 2)
#         elif 'third class' in info_lower or 'class 3' in info_lower:
#             conditions.append(df['Pclass'] == 3)
        
#         # Age extraction
#         import re
#         age_match = re.search(r'age (\d+)', info_lower)
#         if age_match:
#             age = int(age_match.group(1))
#             conditions.append((df['Age'] >= age - 5) & (df['Age'] <= age + 5))
        
#         # Filter data
#         if conditions:
#             mask = conditions[0]
#             for cond in conditions[1:]:
#                 mask = mask & cond
#             filtered = df[mask]
#         else:
#             filtered = df
        
#         if len(filtered) == 0:
#             return "No passengers found matching the description"
        
#         # Calculate survival rate
#         survival_rate = filtered['Survived'].mean()
#         total_matching = len(filtered)
#         survived = filtered['Survived'].sum()
        
#         # Additional context
#         overall_survival = df['Survived'].mean()
        
#         return f"Survival analysis for '{passenger_info}':\n  Matching passengers: {total_matching}\n  Survived: {survived}\n  Survival rate: {survival_rate:.1%}\n  Overall survival rate: {overall_survival:.1%}\n  Relative survival: {survival_rate/overall_survival:.2f}x the average"
#     except Exception as e:
#         return f"Error analyzing survival: {str(e)}"


# @tool
# def get_dataset_insights(file_path: str) -> str:
#     """
#     Generate comprehensive insights about the dataset.
    
#     Args:
#         file_path: Path to the CSV file
        
#     Returns:
#         String with key insights about the data
#     """
#     try:
#         df = pd.read_csv(file_path)
        
#         insights = []
        
#         # Basic info
#         insights.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns")
        
#         # Missing data
#         missing = df.isnull().sum()
#         if missing.sum() > 0:
#             worst_missing = missing.nlargest(3)
#             insights.append(f"Missing data in: {worst_missing.to_dict()}")
        
#         # For Titanic specific insights
#         if 'Survived' in df.columns:
#             survival_rate = df['Survived'].mean()
#             insights.append(f"Overall survival rate: {survival_rate:.1%}")
            
#             if 'Sex' in df.columns:
#                 female_survival = df[df['Sex'] == 'female']['Survived'].mean()
#                 male_survival = df[df['Sex'] == 'male']['Survived'].mean()
#                 insights.append(f"Female survival: {female_survival:.1%}, Male survival: {male_survival:.1%}")
            
#             if 'Pclass' in df.columns:
#                 class_survival = df.groupby('Pclass')['Survived'].mean()
#                 insights.append(f"Survival by class: {class_survival.to_dict()}")
        
#         # Numeric columns
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         insights.append(f"Numeric columns: {list(numeric_cols)}")
        
#         # Categorical columns
#         cat_cols = df.select_dtypes(include=['object']).columns
#         insights.append(f"Categorical columns: {list(cat_cols)}")
        
#         return "Dataset Insights:\n" + "\n".join(f"• {insight}" for insight in insights)
#     except Exception as e:
#         return f"Error generating insights: {str(e)}"


# ============================================================================
# SPECIALIZED AGENTS CREATION
# ============================================================================

def create_agents():
    """Create and return all specialized agents."""
    
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    
    # memory_context = []  # lista che accumula il contesto

    # def memory_callback(step):
    #     """Callback invocata a ogni step del CodeAgent"""
    #     if step.output is not None:
    #         memory_context.append({
    #             "thought": step.thought,
    #             "code": step.code,
    #             "output": step.output,
    #         })
    #     print(f"Message {i} Thought: {step.thought}\nCode:\n{step.code}\nOutput:\n{step.output}\n{'-'*40}" for i, step in enumerate(memory_context))

    # OLD ARCHITECTURE - This function is now DEPRECATED
    # Replaced by TODO 3: Planner-Executor-Answerer pattern
    # See new functions below: create_planner_agent(), create_executor_agent(), create_answerer_agent()

    print("🔧 Initializing Model...")
    # model = VLLMModel(model_id=CONFIG["model_id"])
    model = TransformersModel(
        model_id=CONFIG["model_id"],
        temperature=CONFIG["temperature"],
        max_new_tokens=CONFIG["max_new_tokens"],
        trust_remote_code=True,
        token=CONFIG["huggingface_token"],
        max_length=CONFIG["max_length"],
        # quantization_config=bnb_config,
        device_map="auto"
    )
    print(f"✅ Model initialized: {CONFIG['model_id']}")
    
    print("\n🤖 Creating Specialized Agents...")
    
    # Data Exploration Agent - Focused on comprehensive data discovery and understanding
    print("   📊 Creating Data Exploration Agent...")
    data_reader_agent = CodeAgent(
        tools=[
            load_dataset,
            get_column_names,
            get_data_types,
            get_null_counts,
            get_unique_values,
            get_numeric_summary,
            get_first_rows,
            get_dataset_insights
        ],
        model=model,
        # step_callbacks=[memory_callback],
        # use_structured_outputs_internally=True,
        additional_authorized_imports=["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "scipy"],
        name="DataExplorationAgent",
        description="""Expert agent specialized in comprehensive data exploration and initial dataset analysis.

Core Capabilities:
- **Data Loading & Validation**: Loads CSV/JSON datasets and validates their structure, reporting dimensions (rows × columns)
- **Schema Analysis**: Examines column names, data types (numeric, categorical, datetime) and identifies potential data quality issues
- **Missing Data Detection**: Counts and reports null values per column with percentage calculations to assess data completeness
- **Statistical Profiling**: Generates descriptive statistics for numeric columns (mean, median, std, quartiles, min/max)
- **Categorical Analysis**: Identifies unique values in categorical columns with frequency distributions
- **Data Preview**: Provides first N rows preview to understand data structure and sample values
- **Automated Insights**: Generates comprehensive dataset insights including correlations, patterns, and anomalies
- **Data Quality Assessment**: Evaluates overall data quality and suggests preprocessing strategies

Use this agent when you need to understand a new dataset, perform initial exploratory data analysis (EDA),
or get a comprehensive overview of data characteristics before applying transformations or models.""",
        max_steps=8,
        planning_interval=5
    )
    
    # Data Preprocessing Agent - Focused on comprehensive data cleaning and transformation
    print("   🔧 Creating Data Preprocessing Agent...")
    data_manipulation_agent = CodeAgent(
        tools=[
            drop_column,
            drop_null_rows,
            fill_numeric_nulls,
            fill_categorical_nulls,
            encode_categorical,
            create_new_feature,
            normalize_column,
            filter_rows_numeric,
            filter_rows_categorical,
            select_columns
        ],
        model=model,
        # step_callbacks=[memory_callback],
        # use_structured_outputs_internally=True,
        additional_authorized_imports=["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "scipy"],
        name="DataPreprocessingAgent",
        description="""Expert agent specialized in data cleaning, transformation, and feature engineering for ML-ready datasets.

Core Capabilities:
- **Missing Value Management**: Handles null values with multiple strategies (mean/median/mode imputation, forward/backward fill, or removal)
- **Column Operations**: Drops irrelevant or redundant columns, selects specific feature subsets for analysis
- **Categorical Encoding**: Transforms categorical variables using one-hot encoding, label encoding, or ordinal encoding
- **Feature Engineering**: Creates new features through mathematical operations, combinations, or domain-specific transformations
- **Data Normalization**: Applies scaling techniques (Min-Max, Z-score standardization) to ensure features are on comparable scales
- **Row Filtering**: Filters data based on numeric conditions (>, <, ==) or categorical values (inclusion/exclusion)
- **Data Type Conversion**: Converts columns to appropriate data types for optimal processing and memory usage
- **Outlier Treatment**: Identifies and handles outliers through capping, removal, or transformation
- **Data Quality Enhancement**: Ensures data consistency, handles duplicates, and validates data integrity

Use this agent when you need to prepare raw data for analysis, clean messy datasets, engineer features,
or transform data into formats suitable for statistical analysis and machine learning models.""",
        max_steps=10,
        planning_interval=5
    )
    
    # Statistical Inference Agent - Focused on hypothesis testing and statistical analysis
    print("   📈 Creating Statistical Inference Agent...")
    statistical_agent = CodeAgent(
        tools=[
            calculate_correlation,
            perform_ttest,
            chi_square_test,
            calculate_group_statistics
        ],
        model=model,
        # step_callbacks=[memory_callback],
        # use_structured_outputs_internally=True,
        name="StatisticalInferenceAgent",
        additional_authorized_imports=["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "scipy"],
        description="""Expert agent specialized in statistical hypothesis testing, correlation analysis, and inferential statistics.

Core Capabilities:
- **Correlation Analysis**: Calculates Pearson and Spearman correlation coefficients between variables with p-value significance testing
- **Hypothesis Testing**: Performs parametric (t-tests, ANOVA) and non-parametric tests with detailed interpretation of results
- **Chi-Square Tests**: Analyzes relationships between categorical variables and tests for independence with contingency tables
- **Group Comparisons**: Compares statistical measures (mean, median, variance) across different groups or categories
- **Distribution Analysis**: Tests for normality, skewness, kurtosis, and identifies distribution patterns
- **Confidence Intervals**: Calculates confidence intervals for population parameters at various significance levels
- **Effect Size Calculation**: Measures practical significance beyond statistical significance (Cohen's d, eta-squared)
- **Statistical Reporting**: Generates comprehensive statistical reports with interpretation and recommendations
- **Assumption Validation**: Checks statistical test assumptions (normality, homogeneity of variance, independence)

Use this agent when you need to test hypotheses, validate assumptions, measure relationships between variables,
or perform rigorous statistical analysis to draw data-driven conclusions with quantified confidence levels.""",
        max_steps=8,
        planning_interval=5
    )
    
    # Data Visualization Agent - Focused on creating insightful visual analytics
    print("   📊 Creating Data Visualization Agent...")
    visualization_agent = CodeAgent(
        tools=[
            create_histogram,
            create_scatter_plot,
            create_bar_chart,
            create_correlation_heatmap
        ],
        model=model,
        # step_callbacks=[memory_callback],
        # use_structured_outputs_internally=True,
        additional_authorized_imports=["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "scipy"],
        name="DataVisualizationAgent",
        description="""Expert agent specialized in creating publication-quality data visualizations and visual analytics.

Core Capabilities:
- **Distribution Visualization**: Creates histograms with density curves to show data distributions, identify skewness, and detect outliers
- **Relationship Analysis**: Generates scatter plots with regression lines, confidence intervals, and color-coded categories
- **Categorical Comparison**: Produces bar charts, grouped bars, and stacked bars for comparing categorical data across groups
- **Correlation Heatmaps**: Builds annotated correlation matrices with color gradients to visualize multivariate relationships
- **Time Series Plots**: Creates line charts with trend lines for temporal data analysis and forecasting visualization
- **Advanced Plotting**: Supports violin plots, box plots, pair plots, and multi-panel figures for comprehensive analysis
- **Statistical Overlays**: Adds statistical annotations (mean lines, confidence bands, significance markers) to charts
- **Customization**: Applies professional styling, custom color palettes, labels, titles, and legends for presentation-ready graphics
- **Interactive Features**: Supports zoom, pan, tooltips, and interactive legend toggling for exploratory analysis

Use this agent when you need to communicate data insights visually, create charts for reports and presentations,
explore relationships between variables graphically, or generate publication-quality figures for research papers.""",
        max_steps=8,
        planning_interval=5
    )
    
    # Machine Learning Agent - Focused on end-to-end ML model development and deployment
    print("   🎯 Creating Machine Learning Agent...")
    ml_prediction_agent = CodeAgent(
        tools=[
            train_logistic_regression,
            train_decision_tree,
            train_random_forest_model,
            train_gradient_boosting,
            train_knn_model,
            train_svm_model,
            perform_cross_validation,
            feature_selection,
            make_prediction,
            evaluate_model
        ],
        model=model,
        # step_callbacks=[memory_callback],
        # use_structured_outputs_internally=True,
        additional_authorized_imports=["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "scipy"],
        name="MachineLearningAgent",
        description="""Expert agent specialized in end-to-end machine learning model development, training, evaluation, and deployment.

Core Capabilities:
- **Model Training Suite**: Trains multiple ML algorithms including Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, KNN, and SVM
- **Classification & Regression**: Handles both supervised learning tasks with automatic task type detection and appropriate metric selection
- **Cross-Validation**: Performs k-fold cross-validation with stratification to ensure robust model performance estimates
- **Feature Selection**: Identifies most important features using statistical tests, mutual information, and model-based importance scores
- **Hyperparameter Tuning**: Optimizes model parameters using grid search and random search with automated scoring metrics
- **Model Evaluation**: Computes comprehensive metrics (accuracy, precision, recall, F1, AUC-ROC for classification; MSE, RMSE, R² for regression)
- **Ensemble Methods**: Combines multiple models for improved predictions through voting and stacking techniques
- **Prediction Pipeline**: Generates predictions on new data with confidence scores and probability estimates
- **Model Persistence**: Saves trained models to disk for deployment and future use with versioning support
- **Feature Importance Analysis**: Extracts and visualizes which features contribute most to predictions

Use this agent when you need to build predictive models, compare algorithm performance, deploy ML solutions,
or create production-ready machine learning pipelines for classification or regression tasks.""",
        max_steps=12,
        planning_interval=5
    )
    
    # Question Answering Agent - Specialized in domain-specific data queries
    print("   💡 Creating Question Answering Agent...")
    answer_agent = CodeAgent(
        tools=[
            answer_survival_question,
            predict_single_passenger_survival
        ],
        model=model,
        # step_callbacks=[memory_callback],
        # use_structured_outputs_internally=True,
        additional_authorized_imports=["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "scipy"],
        name="QuestionAnsweringAgent",
        description="""Expert agent specialized in answering domain-specific questions and providing contextual insights from datasets.

Core Capabilities:
- **Natural Language Queries**: Interprets and answers user questions in natural language about dataset characteristics
- **Survival Analysis**: Specifically designed for Titanic dataset, calculates survival probabilities based on passenger attributes
- **Contextual Prediction**: Makes individual predictions with detailed explanations based on similar historical cases
- **Comparative Analysis**: Compares individual cases against population statistics and provides relative likelihood assessments
- **Pattern Recognition**: Identifies patterns in data that answer specific "what if" and "how likely" questions
- **Probability Estimation**: Calculates and explains probability distributions for specific scenarios or passenger profiles
- **Historical Matching**: Finds similar historical cases and uses them to answer predictive questions
- **Result Interpretation**: Translates statistical results into clear, actionable answers with confidence levels
- **Domain Knowledge Integration**: Applies domain-specific knowledge (e.g., Titanic disaster factors) to enhance answers

Use this agent when you need direct, interpretable answers to specific questions about the data,
probability estimates for individual cases, or contextual explanations based on historical patterns in the dataset.""",
        max_steps=5,
        planning_interval=3
    )
    
    print("✅ All specialized agents created successfully!")

    return model, data_reader_agent, data_manipulation_agent, statistical_agent, visualization_agent, ml_prediction_agent, answer_agent


# ============================================================================
# TODO 3: NEW PLANNER-EXECUTOR-ANSWERER ARCHITECTURE
# ============================================================================
# Implementation of the three-agent pattern as required by todo.md
# This replaces the previous Manager + 6 specialized agents architecture

def _initialize_model():
    """Initialize and return the TransformersModel (smolagents framework)."""
    print("🔧 Initializing Model...")
    model = TransformersModel(
        model_id=CONFIG["model_id"],
        temperature=CONFIG["temperature"],
        max_new_tokens=CONFIG["max_new_tokens"],
        trust_remote_code=True,
        token=CONFIG["huggingface_token"],
        max_length=CONFIG["max_length"],
        device_map="auto"
    )
    print(f"✅ Model initialized: {CONFIG['model_id']}")
    return model


def create_planner_agent():
    """
    Create PlannerAgent - TODO 3 implementation.

    This agent has NO TOOLS and only uses LLM reasoning to:
    - Analyze user requests
    - Break down complex tasks into step-by-step execution plans
    - Determine which tools the Executor should use

    Returns tuple: (model, planner_agent)
    Both are smolagents framework objects (TransformersModel, CodeAgent).
    """
    print("\n🎯 Creating Planner Agent (TODO 3)...")
    model = _initialize_model()

    planner = CodeAgent(
        tools=[],  # NO TOOLS - Pure reasoning agent
        model=model,
        additional_authorized_imports=["json"],
        name="PlannerAgent",
        description="""Expert planning agent specialized in task decomposition and workflow design.

Core Responsibilities (TODO 3 Implementation):
- **Task Analysis**: Understands user requests in natural language and identifies analytical goals
- **Workflow Design**: Breaks down complex data science tasks into sequential, executable steps
- **Tool Selection**: Determines which specific tools the Executor needs to use for each step
- **Plan Formatting**: Creates structured execution plans for the Executor
- **Dependency Management**: Ensures steps are ordered correctly with proper data dependencies
- **Error Anticipation**: Predicts potential issues and includes fallback strategies

The Planner NEVER executes tools directly - it only creates plans for the Executor to follow.

When creating plans, think about:
1. What is the user's ultimate goal?
2. What data operations are needed (load, clean, transform, analyze)?
3. In what order should operations happen (dependencies)?
4. What tools exist in the Executor to accomplish each step?
5. What parameters does each tool need?

Output: A clear, sequential plan describing each step and which tool to use.

Use this agent as the first step in the Planner → Executor → Answerer workflow.""",
        max_steps=5,
        planning_interval=3
    )

    print("✅ Planner Agent created successfully!")
    return model, planner


def create_executor_agent():
    """
    Create ExecutorAgent - TODO 3 implementation.

    This agent has ALL TOOLS from smolagents_tools and:
    - Receives execution plans from the Planner
    - Executes each tool step-by-step
    - Manages in-memory DataFrame state using DataFrameStateManager
    - Returns raw execution results to the Answerer

    Returns tuple: (model, executor_agent)
    Both are smolagents framework objects (TransformersModel, CodeAgent).
    """
    print("\n⚙️  Creating Executor Agent with ALL tools (TODO 3)...")
    model = _initialize_model()

    # Collect ALL tools from smolagents_tools (50+ tools)
    all_tools = [
        # Data reading tools
        load_dataset, read_csv_file, read_json_file, get_column_names, get_data_types,
        get_null_counts, get_unique_values, get_numeric_summary, get_first_rows,
        get_column_info, get_data_summary, preview_data, get_dataset_insights,

        # Data manipulation tools
        drop_column, drop_null_rows, fill_numeric_nulls, fill_categorical_nulls,
        encode_categorical, create_new_feature, normalize_column, handle_missing_values,
        create_dummy_variables, modify_column_values, convert_data_types,

        # Data operations tools
        filter_rows_numeric, filter_rows_categorical, select_columns, filter_data,
        perform_math_operations, aggregate_data, string_operations,

        # Statistical analysis tools
        calculate_correlation, perform_ttest, chi_square_test, calculate_group_statistics,

        # Visualization tools
        create_histogram, create_scatter_plot, create_bar_chart, create_correlation_heatmap,

        # Machine Learning tools
        train_model,  # Universal ML tool (TODO 4 - already unified)
        train_logistic_regression, train_decision_tree, train_random_forest_model,
        train_gradient_boosting, train_knn_model, train_svm_model, train_regression_model,
        perform_cross_validation, feature_selection, make_prediction,
        evaluate_model, evaluate_model_universal, predict_with_model_universal,

        # Domain-specific tools
        answer_survival_question, predict_single_passenger_survival,

        # DataFrame state management tools (TODO 2)
        save_dataset, clear_dataframe_cache, get_dataframe_info
    ]

    executor = CodeAgent(
        tools=all_tools,  # ALL TOOLS from smolagents_tools
        model=model,
        additional_authorized_imports=["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "scipy", "json", "joblib"],
        name="ExecutorAgent",
        description="""Expert execution agent specialized in running data science workflows step-by-step.

Core Responsibilities (TODO 3 Implementation):
- **Plan Execution**: Receives structured plans from the Planner and executes them sequentially
- **Tool Invocation**: Calls the appropriate tool for each step with correct parameters
- **State Management**: Manages in-memory DataFrame using DataFrameStateManager (TODO 2)
- **Error Handling**: Catches tool failures and reports them for replanning
- **Result Collection**: Aggregates outputs from all tool executions
- **Progress Tracking**: Reports completion status for each step

The Executor has access to 50+ tools covering:
- Data Loading & Exploration (read_csv_file, get_column_info, get_data_summary, etc.)
- Data Cleaning & Preprocessing with in-memory optimization (handle_missing_values, encode_categorical, etc.)
- Statistical Analysis & Testing (calculate_correlation, perform_ttest, chi_square_test, etc.)
- Data Visualization (create_histogram, create_scatter_plot, create_bar_chart, etc.)
- Machine Learning with unified train_model tool - TODO 4 (train_model, evaluate_model, etc.)
- Domain-Specific Analysis (answer_survival_question, predict_single_passenger_survival)

The Executor NEVER creates plans - it only follows plans from the Planner.
The Executor returns RAW results - the Answerer will format them for users.

Use this agent as the second step in the Planner → Executor → Answerer workflow.""",
        max_steps=20,  # May need many steps for complex workflows
        planning_interval=5
    )

    print(f"✅ Executor Agent created with {len(all_tools)} tools!")
    return model, executor


def create_answerer_agent():
    """
    Create AnswererAgent - TODO 3 implementation.

    This agent has NO TOOLS and only uses LLM reasoning to:
    - Receive raw execution results from the Executor
    - Synthesize results into clear, natural language responses
    - Format data insights for end users

    Returns tuple: (model, answerer_agent)
    Both are smolagents framework objects (TransformersModel, CodeAgent).
    """
    print("\n💬 Creating Answerer Agent (TODO 3)...")
    model = _initialize_model()

    answerer = CodeAgent(
        tools=[],  # NO TOOLS - Pure synthesis agent
        model=model,
        additional_authorized_imports=["json"],
        name="AnswererAgent",
        description="""Expert synthesis agent specialized in translating technical results into user-friendly answers.

Core Responsibilities (TODO 3 Implementation):
- **Result Interpretation**: Analyzes raw tool outputs from the Executor
- **Insight Extraction**: Identifies key findings, patterns, and actionable insights
- **Natural Language Generation**: Formulates clear, concise answers in conversational language
- **Context Integration**: Combines results from multiple steps into a coherent narrative
- **User-Centric Communication**: Adapts technical jargon to user's level of expertise
- **Recommendation Synthesis**: Provides actionable recommendations based on analysis results

The Answerer NEVER executes tools or creates plans - it only synthesizes results.
The Answerer receives raw outputs (numbers, DataFrames, metrics) and produces final answers.

When synthesizing answers:
1. What was the user's original question?
2. What key findings came from the Executor?
3. What insights can be drawn from the results?
4. What actions or recommendations follow from this analysis?
5. How can I explain this clearly to a non-technical user?

Input: Raw results from Executor (text descriptions of tool outputs)
Output: Natural language answer directly addressing the user's original question

Use this agent as the final step in the Planner → Executor → Answerer workflow.""",
        max_steps=3,
        planning_interval=2
    )

    print("✅ Answerer Agent created successfully!")
    return model, answerer


# ============================================================================
# TODO 6: LAZY LOADING IMPLEMENTATION
# ============================================================================
# Load agents on-demand and unload them to free GPU memory
# This allows running the system with minimal GPU footprint

def load_agent_lazy(agent_type: str):
    """
    Load a single agent on-demand - TODO 6 implementation.

    This function:
    - Loads only the requested agent (Planner, Executor, or Answerer)
    - Tracks GPU memory usage before and after loading
    - Returns the agent and its model for use

    Args:
        agent_type: One of "planner", "executor", or "answerer"

    Returns:
        Tuple (model, agent): The TransformersModel and CodeAgent objects

    Usage:
        model, planner = load_agent_lazy("planner")
        # Use the planner...
        unload_agent(model, planner, "planner")
    """
    print(f"\n{'='*60}")
    print(f"🔄 TODO 6: Lazy Loading {agent_type.upper()} agent...")
    print(f"{'='*60}")

    # Track GPU memory before loading
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1e9  # Convert to GB
        print(f"📊 GPU Memory before loading: {mem_before:.2f} GB")
    else:
        mem_before = 0

    # Load the appropriate agent using smolagents framework
    if agent_type == "planner":
        model, agent = create_planner_agent()
    elif agent_type == "executor":
        model, agent = create_executor_agent()
    elif agent_type == "answerer":
        model, agent = create_answerer_agent()
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}. Must be 'planner', 'executor', or 'answerer'")

    # Track GPU memory after loading
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1e9
        mem_delta = mem_after - mem_before
        print(f"📊 GPU Memory after loading: {mem_after:.2f} GB (+{mem_delta:.2f} GB)")

    print(f"✅ {agent_type.upper()} agent loaded successfully!")
    return model, agent


def unload_agent(model, agent, agent_type: str):
    """
    Unload an agent and free GPU memory - TODO 6 implementation.

    This function:
    - Deletes the agent and model objects
    - Calls torch.cuda.empty_cache() to free GPU memory
    - Tracks how much memory was freed

    Args:
        model: The TransformersModel to unload
        agent: The CodeAgent to unload
        agent_type: Name of the agent for logging ("planner", "executor", "answerer")

    Usage:
        model, planner = load_agent_lazy("planner")
        # ... use planner ...
        unload_agent(model, planner, "planner")
    """
    print(f"\n🗑️  TODO 6: Unloading {agent_type.upper()} agent to free GPU memory...")

    # Track GPU memory before unloading
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1e9
        print(f"📊 GPU Memory before unloading: {mem_before:.2f} GB")

    # Delete agent and model objects
    del agent
    del model

    # Free GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1e9
        mem_freed = mem_before - mem_after
        print(f"📊 GPU Memory after unloading: {mem_after:.2f} GB (freed {mem_freed:.2f} GB)")
        print(f"✅ {agent_type.upper()} agent unloaded, GPU memory freed!")
    else:
        print(f"✅ {agent_type.upper()} agent unloaded!")


# ============================================================================
# OLD MANAGER AGENT (DEPRECATED - Replaced by TODO 3)
# ============================================================================
# This function is kept for backward compatibility but is no longer used
# New architecture uses: create_planner_agent(), create_executor_agent(), create_answerer_agent()

def create_manager_agent(model, *agents):
    """Create the manager agent that coordinates all specialized agents."""
    
    print("\n👔 Creating Orchestration Manager Agent...")

    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=list(agents),
        additional_authorized_imports=["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "scipy"],
        name="OrchestrationManagerAgent",
        description="""Master orchestrator agent that coordinates and manages all specialized agents to execute complex data science workflows.

Core Capabilities:
- **Workflow Orchestration**: Decomposes complex data science tasks into step-by-step workflows and assigns them to appropriate specialized agents
- **Agent Coordination**: Manages communication and data flow between DataExplorationAgent, DataPreprocessingAgent, StatisticalInferenceAgent, DataVisualizationAgent, MachineLearningAgent, and QuestionAnsweringAgent
- **Task Decomposition**: Breaks down user requests into atomic subtasks, creating optimal execution plans with clear dependencies
- **Dynamic Planning**: Adapts execution strategy based on intermediate results, adjusting workflow when unexpected patterns or issues arise
- **Resource Management**: Optimizes agent utilization, preventing redundant operations and ensuring efficient task execution
- **Quality Assurance**: Validates outputs from each agent, ensures data consistency across workflow steps, and maintains result quality
- **Error Handling**: Detects failures in agent execution, implements fallback strategies, and provides diagnostic information
- **Result Synthesis**: Aggregates outputs from multiple agents into coherent, comprehensive final answers
- **Context Maintenance**: Preserves analysis context throughout multi-step workflows, ensuring consistency and traceability
- **Decision Making**: Determines optimal agent selection based on task requirements, data characteristics, and desired outcomes

Workflow Patterns:
- **Exploratory Analysis**: DataExplorationAgent → StatisticalInferenceAgent → DataVisualizationAgent
- **Data Preparation**: DataExplorationAgent → DataPreprocessingAgent → DataExplorationAgent (validation)
- **ML Pipeline**: DataExplorationAgent → DataPreprocessingAgent → MachineLearningAgent → DataVisualizationAgent
- **Question Answering**: DataExplorationAgent → QuestionAnsweringAgent
- **Full Analysis**: Coordinates all agents for comprehensive end-to-end data science projects

Use this agent as the primary entry point for complex, multi-step data analysis tasks that require coordination
of multiple specialized capabilities, or when you need an intelligent system to automatically determine the best
workflow strategy for achieving your analytical objectives.""",
        max_steps=25,
        # use_structured_outputs_internally=True,
        planning_interval=5
    )
    
    print("✅ Manager Agent created successfully!")
    print("\n🎯 Multi-Agent System Ready!")
    
    return manager_agent


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def run_analysis(user_prompt: str, file_path: str = None) -> str:
    """
    Run multi-agent data analysis with NEW Planner-Executor-Answerer architecture.

    TODO 3 + TODO 6 Implementation:
    - Uses Planner → Executor → Answerer pattern (TODO 3)
    - Implements lazy loading to minimize GPU memory (TODO 6)
    - Manages in-memory DataFrame state (TODO 2)

    Architecture:
    1. PLANNER creates execution plan
    2. EXECUTOR runs plan with all tools
    3. ANSWERER synthesizes results

    Each agent is loaded on-demand and unloaded after use to free GPU memory.
    """
    if file_path is None:
        file_path = CONFIG["default_dataset"]

    print(f"\n{'='*80}")
    print(f"🚀 Starting Multi-Agent Analysis (TODO 3 + TODO 6)")
    print(f"📝 Prompt: {user_prompt}")
    print(f"📁 File: {file_path}")
    print(f"{'='*80}\n")

    # Load dataset into in-memory DataFrame (TODO 2)
    print("📊 Loading dataset into memory (TODO 2: In-Memory DataFrame)...")
    from smolagents_tools import df_state_manager
    df_state_manager.load_dataframe(file_path)
    print(f"✅ Dataset loaded into memory, ready for analysis\n")

    # Analyze dataset columns for context
    print("📊 Analyzing dataset columns...")
    columns_info = analyze_dataset_columns(file_path)
    column_descriptions = generate_column_descriptions_with_llm(columns_info)
    formatted_columns = format_column_descriptions(column_descriptions)

    # ========================================================================
    # STEP 1: PLANNER (TODO 3 + TODO 6 Lazy Loading)
    # ========================================================================
    planner_prompt = f"""
User question: {user_prompt}
Data file: {file_path}

{formatted_columns}

Your task is to create a detailed execution plan for analyzing this data.

Think step-by-step:
1. What is the user asking for?
2. What operations are needed (explore, clean, transform, analyze, model)?
3. What tools does the Executor have available?
4. In what order should operations happen?

Create a plan that the Executor can follow. Be specific about which tools to use and what parameters they need.
The Executor has 50+ tools including:
- Data loading and exploration tools
- Data cleaning and preprocessing tools
- Statistical analysis tools
- Visualization tools
- Machine learning tools (including train_model for unified ML)

Provide a clear, step-by-step plan.
"""

    try:
        # Load Planner (TODO 6: Lazy loading)
        planner_model, planner = load_agent_lazy("planner")

        print("\n🎯 PLANNER: Creating execution plan...")
        plan = planner.run(planner_prompt)
        print(f"\n✅ Plan created:\n{plan}\n")

        # Unload Planner to free GPU memory (TODO 6)
        unload_agent(planner_model, planner, "planner")

        # ====================================================================
        # STEP 2: EXECUTOR (TODO 3 + TODO 6 Lazy Loading)
        # ====================================================================
        executor_prompt = f"""
You have the following execution plan from the Planner:

{plan}

Now execute this plan step-by-step using the available tools.

IMPORTANT:
- The dataset is already loaded in memory (file_path: {file_path})
- Use the tools to perform each step
- All data manipulation tools work with in-memory DataFrame (TODO 2)
- Report the results of each tool execution
- If a step fails, report the error clearly

Available tools: 50+ tools covering data loading, preprocessing, analysis, visualization, and ML.

Execute the plan now.
"""

        # Load Executor (TODO 6: Lazy loading)
        executor_model, executor = load_agent_lazy("executor")

        print("\n⚙️  EXECUTOR: Executing plan with all tools...")
        results = executor.run(executor_prompt)
        print(f"\n✅ Execution complete. Results:\n{results}\n")

        # Unload Executor to free GPU memory (TODO 6)
        unload_agent(executor_model, executor, "executor")

        # ====================================================================
        # STEP 3: ANSWERER (TODO 3 + TODO 6 Lazy Loading)
        # ====================================================================
        answerer_prompt = f"""
The user asked: {user_prompt}

The Executor performed analysis and returned these results:

{results}

Your task is to synthesize these results into a clear, user-friendly answer.

Think about:
1. What was the user's original question?
2. What key findings came from the analysis?
3. What insights can be drawn?
4. What recommendations or next steps make sense?

Provide a clear, concise answer that directly addresses the user's question.
"""

        # Load Answerer (TODO 6: Lazy loading)
        answerer_model, answerer = load_agent_lazy("answerer")

        print("\n💬 ANSWERER: Synthesizing final answer...")
        final_answer = answerer.run(answerer_prompt)

        # Unload Answerer (TODO 6)
        unload_agent(answerer_model, answerer, "answerer")

        print(f"\n{'='*80}")
        print("✅ Analysis Complete! (TODO 3 + TODO 6 Implementation)")
        print(f"{'='*80}\n")

        return final_answer

    except Exception as e:
        import traceback
        error_msg = f"❌ Error in Planner-Executor-Answerer workflow: {str(e)}\n{traceback.format_exc()}"
        print(f"\n{error_msg}\n")
        return error_msg


def run_analysis_old(user_prompt: str, file_path: str = None) -> str:
    """
    OLD IMPLEMENTATION - DEPRECATED (Replaced by TODO 3).

    This function used the Manager + 6 specialized agents architecture.
    New implementation above uses Planner → Executor → Answerer pattern.

    Kept for backward compatibility but should not be used.
    """
    if file_path is None:
        file_path = CONFIG["default_dataset"]

    print(f"\n{'='*80}")
    print(f"🚀 Starting Multi-Agent Analysis (OLD ARCHITECTURE)")
    print(f"📝 Prompt: {user_prompt}")
    print(f"📁 File: {file_path}")
    print(f"{'='*80}\n")

    # Step 1: Analyze dataset columns
    print("📊 Analyzing dataset columns...")
    columns_info = analyze_dataset_columns(file_path)

    # Step 2: Generate column descriptions with LLM
    column_descriptions = generate_column_descriptions_with_llm(columns_info)

    # Step 3: Format column descriptions for prompt
    formatted_columns = format_column_descriptions(column_descriptions)

    # Create all agents
    model, reader, manipulator, statistical, visualizer, ml, answerer = create_agents()

    # Create manager
    manager = create_manager_agent(
        model, reader, manipulator, statistical, visualizer, ml, answerer
    )

    task = f"""
User question: {user_prompt}
Data file: {file_path}

{formatted_columns}

I want you to create a plan where, at each step, you use an agent.
For every step, explain why you want to use that specific agent and write the prompt you will give to that agent. Provide always the data file path with the prompt.
By the end of all the steps, the user's request must be fully answered.

When you replan, take into account the steps that have already been completed — exclude them from the new plan, and keep only the remaining steps to be done, or modify the steps if something went wrong.

Start by analyzing the user's request and creating a plan.
"""

    print("🎬 Executing analysis...\n")

    try:
        result = manager.run(task)
        print(f"\n{'='*80}")
        print("✅ Analysis Complete!")
        print(f"{'='*80}\n")
        return result
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(f"\n{error_msg}\n")
        return error_msg


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

# if __name__ == "__main__":
#     print("""
# ╔════════════════════════════════════════════════════════════════╗
# ║   🤖 ENHANCED MULTI-AGENT DATA ANALYSIS SYSTEM (smolagents)    ║
# ║                                                                ║
# ║        Comprehensive Analysis for Titanic Dataset              ║
# ╚════════════════════════════════════════════════════════════════╝
#     """)
    
#     # Example prompts for testing
#     example_prompts = [
#         "What's the survival rate for women in first class?",
#         "Create a visualization showing survival by class and gender",
#         "Train a model to predict survival and tell me the most important features",
#         "Handle missing values and prepare the data for modeling",
#         "Perform statistical analysis on age vs survival",
#         "Give me insights about the dataset"
#     ]
    
#     print("\n📝 Example questions you can ask:")
#     for i, prompt in enumerate(example_prompts, 1):
#         print(f"  {i}. {prompt}")
    
#     print("\n📝 Enter your analysis request (or 'example X' for an example):")
#     user_input = input("> ").strip()
    
#     # Check if user wants an example
#     if user_input.lower().startswith('example'):
#         try:
#             idx = int(user_input.split()[1]) - 1
#             user_prompt = example_prompts[idx]
#             print(f"\n✓ Using example: {user_prompt}")
#         except:
#             user_prompt = user_input
#     else:
#         user_prompt = user_input
    
#     # Run analysis
#     result = run_analysis(user_prompt, "data/titanic.csv")
    
#     print("\n" + "="*80)
#     print("📊 ANALYSIS RESULT:")
#     print("="*80)
#     print(result)
#     print("\n" + "="*80)

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║             🤖 MULTI-AGENT DATA ANALYSIS SYSTEM (smolagents)               ║
║                                                                            ║
║  A comprehensive multi-agent system for end-to-end data science workflows ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    print("\n🎯 Interactive Mode - Titanic Dataset Analysis")
    print("="*80)

    # Get file path from user
    print("\n📁 Enter the path to your data file (or press Enter for default 'data/titanic.csv'):")
    file_input = input("File path: ").strip()
    file_path = file_input if file_input else "data/titanic.csv"

    print(f"\n✓ Using file: {file_path}")

    # Display 10 predefined Titanic questions
    print("\n" + "="*80)
    print("📋 PREDEFINED TITANIC DATASET QUESTIONS")
    print("="*80)
    print("\nSelect a question (1-10) or enter 'custom' for your own question:\n")

    for q in TITANIC_QUESTIONS:
        print(f"  {q['numero']}. [{q['livello']:^18}] {q['domanda']}")

    print(f"\n{' '*3}{'CUSTOM':^6}  Enter your own custom question")
    print("="*80)

    # Get user choice
    choice = input("\nYour choice (1-10 or 'custom'): ").strip().lower()

    user_prompt = ""

    # Process choice
    if choice.isdigit() and 1 <= int(choice) <= 10:
        # Use predefined question
        selected_q = TITANIC_QUESTIONS[int(choice) - 1]
        user_prompt = selected_q['domanda']
        print(f"\n✓ Selected Question {choice} [{selected_q['livello']}]:")
        print(f"  '{user_prompt}'")
    elif choice in ['custom', 'c', '']:
        # Custom question
        print("\n📝 Enter your custom analysis request:")
        print("   Type your request below (press Enter twice to finish):\n")

        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            else:
                if lines:  # If we have at least one line and get an empty line, we're done
                    break
                else:  # If first line is empty, ask again
                    continue

        user_prompt = "\n".join(lines)

        if not user_prompt.strip():
            print("\n⚠️  No request provided. Using default example...")
            user_prompt = "Provide a comprehensive overview of the dataset including structure, missing values, and statistical summary"
    else:
        # Invalid choice, default to first question
        print(f"\n⚠️  Invalid choice '{choice}'. Using Question 1 as default...")
        selected_q = TITANIC_QUESTIONS[0]
        user_prompt = selected_q['domanda']
        print(f"  '{user_prompt}'")

    # Run the analysis
    print("\n" + "="*80)
    print("🚀 Starting Analysis")
    print("="*80)

    result = run_analysis(user_prompt, file_path)

    print("\n" + "="*80)
    print("📊 ANALYSIS RESULT:")
    print("="*80)
    print(result)

    print("\n" + "="*80)
    print("✅ Analysis completed!")
    print("="*80)
