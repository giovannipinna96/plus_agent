"""Data operations tools."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from langchain.tools import tool

# Import Titanic-specific tools
from .titanic_specific_tools import (
    calculate_survival_rate_by_group,
    get_statistics_for_profile,
    calculate_survival_probability_by_features,
    get_fare_estimate_by_profile,
    count_passengers_by_criteria
)


@tool
def filter_data(file_path: str, column_name: str, condition: str, value: Union[str, float, int]) -> str:
    """
    Filter data based on a condition.
    
    Args:
        file_path: Path to the data file
        column_name: Name of the column to filter on
        condition: Condition type (equals, not_equals, greater_than, less_than, greater_equal, less_equal, contains)
        value: Value to filter by
        
    Returns:
        String describing the filtering performed
    """
    try:
        df = pd.read_csv(file_path)
        
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in dataset"
        
        original_rows = len(df)
        
        if condition == "equals":
            filtered_df = df[df[column_name] == value]
        elif condition == "not_equals":
            filtered_df = df[df[column_name] != value]
        elif condition == "greater_than":
            filtered_df = df[df[column_name] > float(value)]
        elif condition == "less_than":
            filtered_df = df[df[column_name] < float(value)]
        elif condition == "greater_equal":
            filtered_df = df[df[column_name] >= float(value)]
        elif condition == "less_equal":
            filtered_df = df[df[column_name] <= float(value)]
        elif condition == "contains":
            filtered_df = df[df[column_name].astype(str).str.contains(str(value), na=False)]
        else:
            return f"Unknown condition '{condition}'"
        
        filtered_rows = len(filtered_df)
        
        # Save the filtered dataframe
        output_path = file_path.replace('.csv', '_filtered.csv')
        filtered_df.to_csv(output_path, index=False)
        
        return f"Filtered data: {original_rows} -> {filtered_rows} rows using condition '{column_name} {condition} {value}'. Saved to: {output_path}"
        
    except Exception as e:
        return f"Error filtering data: {str(e)}"


@tool
def perform_math_operations(file_path: str, operation: str, column1: str, column2: Optional[str] = None, value: Optional[float] = None) -> str:
    """
    Perform mathematical operations on columns.
    
    Args:
        file_path: Path to the data file
        operation: Type of operation (add, subtract, multiply, divide, power, square, sqrt, log, abs)
        column1: First column name
        column2: Second column name (for two-column operations)
        value: Numeric value (for column-value operations)
        
    Returns:
        String describing the mathematical operation performed
    """
    try:
        df = pd.read_csv(file_path)
        
        if column1 not in df.columns:
            return f"Column '{column1}' not found in dataset"
        
        if column2 and column2 not in df.columns:
            return f"Column '{column2}' not found in dataset"
        
        # Create new column name for result
        if column2:
            new_col_name = f"{column1}_{operation}_{column2}"
        elif value is not None:
            new_col_name = f"{column1}_{operation}_{value}"
        else:
            new_col_name = f"{column1}_{operation}"
        
        # Perform operations
        if operation == "add":
            if column2:
                df[new_col_name] = df[column1] + df[column2]
            elif value is not None:
                df[new_col_name] = df[column1] + value
        elif operation == "subtract":
            if column2:
                df[new_col_name] = df[column1] - df[column2]
            elif value is not None:
                df[new_col_name] = df[column1] - value
        elif operation == "multiply":
            if column2:
                df[new_col_name] = df[column1] * df[column2]
            elif value is not None:
                df[new_col_name] = df[column1] * value
        elif operation == "divide":
            if column2:
                df[new_col_name] = df[column1] / df[column2]
            elif value is not None:
                df[new_col_name] = df[column1] / value
        elif operation == "power":
            if value is not None:
                df[new_col_name] = df[column1] ** value
            else:
                return "Power operation requires a value parameter"
        elif operation == "square":
            df[new_col_name] = df[column1] ** 2
        elif operation == "sqrt":
            df[new_col_name] = np.sqrt(df[column1])
        elif operation == "log":
            df[new_col_name] = np.log(df[column1])
        elif operation == "abs":
            df[new_col_name] = np.abs(df[column1])
        else:
            return f"Unknown operation '{operation}'"
        
        # Save the modified dataframe
        output_path = file_path.replace('.csv', '_math_ops.csv')
        df.to_csv(output_path, index=False)
        
        return f"Created new column '{new_col_name}' using operation '{operation}'. Saved to: {output_path}"
        
    except Exception as e:
        return f"Error performing math operation: {str(e)}"


@tool
def string_operations(file_path: str, column_name: str, operation: str, parameter: Optional[str] = None) -> str:
    """
    Perform string operations on a text column.
    
    Args:
        file_path: Path to the data file
        column_name: Name of the text column
        operation: Type of operation (upper, lower, title, length, contains_count, split, replace, strip)
        parameter: Additional parameter for some operations
        
    Returns:
        String describing the string operation performed
    """
    try:
        df = pd.read_csv(file_path)
        
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in dataset"
        
        # Convert to string type
        df[column_name] = df[column_name].astype(str)
        
        if operation == "upper":
            new_col_name = f"{column_name}_upper"
            df[new_col_name] = df[column_name].str.upper()
        elif operation == "lower":
            new_col_name = f"{column_name}_lower"
            df[new_col_name] = df[column_name].str.lower()
        elif operation == "title":
            new_col_name = f"{column_name}_title"
            df[new_col_name] = df[column_name].str.title()
        elif operation == "length":
            new_col_name = f"{column_name}_length"
            df[new_col_name] = df[column_name].str.len()
        elif operation == "contains_count" and parameter:
            new_col_name = f"{column_name}_contains_{parameter}"
            df[new_col_name] = df[column_name].str.count(parameter)
        elif operation == "split" and parameter:
            # Split and create new columns
            split_cols = df[column_name].str.split(parameter, expand=True)
            for i, col in enumerate(split_cols.columns):
                df[f"{column_name}_part_{i+1}"] = split_cols[col]
            new_col_name = f"split into {len(split_cols.columns)} columns"
        elif operation == "replace" and parameter:
            # Parameter should be "old,new"
            if "," in parameter:
                old, new = parameter.split(",", 1)
                new_col_name = f"{column_name}_replaced"
                df[new_col_name] = df[column_name].str.replace(old.strip(), new.strip())
            else:
                return "Replace operation requires parameter in format 'old,new'"
        elif operation == "strip":
            new_col_name = f"{column_name}_stripped"
            df[new_col_name] = df[column_name].str.strip()
        else:
            return f"Unknown string operation '{operation}'"
        
        # Save the modified dataframe
        output_path = file_path.replace('.csv', '_string_ops.csv')
        df.to_csv(output_path, index=False)
        
        return f"Applied string operation '{operation}' to column '{column_name}'. Result: {new_col_name}. Saved to: {output_path}"
        
    except Exception as e:
        return f"Error performing string operation: {str(e)}"


@tool
def aggregate_data(file_path: str, group_by_columns: str, agg_column: str, agg_function: str) -> str:
    """
    Aggregate data by grouping columns.
    
    Args:
        file_path: Path to the data file
        group_by_columns: Comma-separated list of columns to group by
        agg_column: Column to aggregate
        agg_function: Aggregation function one of mean, sum, count, min, max, std, median
        
    Returns:
        String describing the aggregation performed
    """
    try:
        df = pd.read_csv(file_path)
        
        # Parse group by columns
        group_cols = [col.strip() for col in group_by_columns.split(',')]
        
        # Check if columns exist
        for col in group_cols + [agg_column]:
            if col not in df.columns:
                return f"Column '{col}' not found in dataset"
        
        # Perform aggregation
        if agg_function == "mean":
            result_df = df.groupby(group_cols)[agg_column].mean().reset_index()
        elif agg_function == "sum":
            result_df = df.groupby(group_cols)[agg_column].sum().reset_index()
        elif agg_function == "count":
            result_df = df.groupby(group_cols)[agg_column].count().reset_index()
        elif agg_function == "min":
            result_df = df.groupby(group_cols)[agg_column].min().reset_index()
        elif agg_function == "max":
            result_df = df.groupby(group_cols)[agg_column].max().reset_index()
        elif agg_function == "std":
            result_df = df.groupby(group_cols)[agg_column].std().reset_index()
        elif agg_function == "median":
            result_df = df.groupby(group_cols)[agg_column].median().reset_index()
        else:
            return f"Unknown aggregation function '{agg_function}'"
        
        # Rename the aggregated column
        result_df = result_df.rename(columns={agg_column: f"{agg_column}_{agg_function}"})
        
        # Save the aggregated dataframe
        output_path = file_path.replace('.csv', '_aggregated.csv')
        result_df.to_csv(output_path, index=False)
        
        return f"Aggregated data by {group_cols} using {agg_function} on '{agg_column}'. Result shape: {result_df.shape}. Saved to: {output_path}"
        
    except Exception as e:
        return f"Error aggregating data: {str(e)}"