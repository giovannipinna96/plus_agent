"""Data manipulation tools."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from langchain.tools import tool


@tool
def create_dummy_variables(file_path: str, column_name: str, prefix: Optional[str] = None) -> str:
    """
    Create dummy variables for a categorical column.
    
    Args:
        file_path: Path to the data file
        column_name: Name of the categorical column
        prefix: Prefix for dummy variable names (optional)
        
    Returns:
        String describing the dummy variables created
    """
    try:
        df = pd.read_csv(file_path)
        
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in dataset"
        
        # Create dummy variables
        dummies = pd.get_dummies(df[column_name], prefix=prefix or column_name)
        
        # Add dummy variables to dataframe
        df_with_dummies = pd.concat([df, dummies], axis=1)
        
        # Save the modified dataframe
        output_path = file_path.replace('.csv', '_with_dummies.csv')
        df_with_dummies.to_csv(output_path, index=False)
        
        return f"Created {len(dummies.columns)} dummy variables for '{column_name}': {list(dummies.columns)}. Saved to: {output_path}"
        
    except Exception as e:
        return f"Error creating dummy variables: {str(e)}"


@tool
def modify_column_values(file_path: str, column_name: str, operation: str, value: Optional[Union[str, float, int]] = None) -> str:
    """
    Modify values in a column using various operations.
    
    Args:
        file_path: Path to the data file
        column_name: Name of the column to modify
        operation: Type of operation (multiply, add, subtract, divide, replace, apply_function)
        value: Value to use in the operation
        
    Returns:
        String describing the modification performed
    """
    try:
        df = pd.read_csv(file_path)
        
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in dataset"
        
        original_column = df[column_name].copy()
        
        if operation == "multiply" and value is not None:
            df[column_name] = df[column_name] * float(value)
        elif operation == "add" and value is not None:
            df[column_name] = df[column_name] + float(value) 
        elif operation == "subtract" and value is not None:
            df[column_name] = df[column_name] - float(value)
        elif operation == "divide" and value is not None:
            df[column_name] = df[column_name] / float(value)
        elif operation == "replace" and value is not None:
            # For replace, value should be in format "old_value,new_value"
            if isinstance(value, str) and "," in value:
                old_val, new_val = value.split(",", 1)
                df[column_name] = df[column_name].replace(old_val.strip(), new_val.strip())
            else:
                return "For replace operation, value should be in format 'old_value,new_value'"
        elif operation == "normalize":
            # Normalize to 0-1 range
            min_val, max_val = df[column_name].min(), df[column_name].max()
            df[column_name] = (df[column_name] - min_val) / (max_val - min_val)
        elif operation == "standardize":
            # Standardize to mean=0, std=1
            mean_val, std_val = df[column_name].mean(), df[column_name].std()
            df[column_name] = (df[column_name] - mean_val) / std_val
        else:
            return f"Unknown operation '{operation}' or missing value parameter"
        
        # Save the modified dataframe
        output_path = file_path.replace('.csv', '_modified.csv')
        df.to_csv(output_path, index=False)
        
        # Calculate changes
        changed_count = (df[column_name] != original_column).sum()
        
        return f"Modified {changed_count} values in column '{column_name}' using operation '{operation}'. Saved to: {output_path}"
        
    except Exception as e:
        return f"Error modifying column values: {str(e)}"


@tool
def handle_missing_values(file_path: str, column_name: str, method: str, fill_value: Optional[Union[str, float, int]] = None) -> str:
    """
    Handle missing values in a column using various methods.
    
    Args:
        file_path: Path to the data file
        column_name: Name of the column with missing values
        method: Method to handle missing values (drop, mean, median, mode, forward_fill, backward_fill, constant)
        fill_value: Value to use for 'constant' method
        
    Returns:
        String describing the missing value handling performed
    """
    try:
        df = pd.read_csv(file_path)
        
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in dataset"
        
        missing_count_before = df[column_name].isnull().sum()
        
        if missing_count_before == 0:
            return f"No missing values found in column '{column_name}'"
        
        if method == "drop":
            df = df.dropna(subset=[column_name])
        elif method == "mean":
            if df[column_name].dtype in ['int64', 'float64']:
                fill_val = df[column_name].mean()
                df[column_name] = df[column_name].fillna(fill_val)
            else:
                return f"Mean method not applicable for non-numeric column '{column_name}'"
        elif method == "median":
            if df[column_name].dtype in ['int64', 'float64']:
                fill_val = df[column_name].median()
                df[column_name] = df[column_name].fillna(fill_val)
            else:
                return f"Median method not applicable for non-numeric column '{column_name}'"
        elif method == "mode":
            mode_val = df[column_name].mode()
            if len(mode_val) > 0:
                df[column_name] = df[column_name].fillna(mode_val[0])
            else:
                return f"No mode found for column '{column_name}'"
        elif method == "forward_fill":
            df[column_name] = df[column_name].fillna(method='ffill')
        elif method == "backward_fill":
            df[column_name] = df[column_name].fillna(method='bfill')
        elif method == "constant" and fill_value is not None:
            df[column_name] = df[column_name].fillna(fill_value)
        else:
            return f"Unknown method '{method}' or missing fill_value for constant method"
        
        missing_count_after = df[column_name].isnull().sum()
        
        # Save the modified dataframe
        output_path = file_path.replace('.csv', '_missing_handled.csv')
        df.to_csv(output_path, index=False)
        
        return f"Handled {missing_count_before - missing_count_after} missing values in column '{column_name}' using method '{method}'. Remaining missing: {missing_count_after}. Saved to: {output_path}"
        
    except Exception as e:
        return f"Error handling missing values: {str(e)}"


@tool
def convert_data_types(file_path: str, column_name: str, target_type: str) -> str:
    """
    Convert data type of a column.
    
    Args:
        file_path: Path to the data file
        column_name: Name of the column to convert
        target_type: Target data type (int, float, string, category, datetime)
        
    Returns:
        String describing the data type conversion performed
    """
    try:
        df = pd.read_csv(file_path)
        
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in dataset"
        
        original_dtype = df[column_name].dtype
        
        if target_type == "int":
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype('Int64')
        elif target_type == "float":
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        elif target_type == "string":
            df[column_name] = df[column_name].astype(str)
        elif target_type == "category":
            df[column_name] = df[column_name].astype('category')
        elif target_type == "datetime":
            df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        else:
            return f"Unknown target type '{target_type}'. Supported types: int, float, string, category, datetime"
        
        new_dtype = df[column_name].dtype
        
        # Save the modified dataframe
        output_path = file_path.replace('.csv', '_type_converted.csv')
        df.to_csv(output_path, index=False)
        
        return f"Converted column '{column_name}' from {original_dtype} to {new_dtype}. Saved to: {output_path}"
        
    except Exception as e:
        return f"Error converting data type: {str(e)}"