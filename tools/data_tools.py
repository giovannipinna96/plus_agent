"""Data reading and analysis tools."""

import pandas as pd
import json
from typing import Dict, Any, List, Optional
from langchain.tools import tool


@tool
def read_csv_file(file_path: str) -> str:
    """
    Read a CSV file and return basic information about the dataset.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        String containing dataset information
    """
    try:
        df = pd.read_csv(file_path)
        
        info = {
            "status": "success",
            "shape": df.shape,
            "columns": list(df.columns),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }
        
        return f"CSV file loaded successfully. Shape: {info['shape']}, Columns: {info['columns']}"
        
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"


@tool
def read_json_file(file_path: str) -> str:
    """
    Read a JSON file and return basic information about the data.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        String containing data information
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
            return f"JSON file loaded successfully. Shape: {df.shape}, Columns: {list(df.columns)}"
        elif isinstance(data, dict):
            return f"JSON object loaded with keys: {list(data.keys())}"
        else:
            return f"JSON data loaded. Type: {type(data)}"
            
    except Exception as e:
        return f"Error reading JSON file: {str(e)}"


@tool
def get_column_info(file_path: str, column_name: Optional[str] = None) -> str:
    """
    Get detailed information about columns in the dataset.
    
    Args:
        file_path: Path to the data file
        column_name: Specific column to analyze (optional)
        
    Returns:
        String containing column information
    """
    try:
        df = pd.read_csv(file_path)
        
        if column_name:
            if column_name not in df.columns:
                return f"Column '{column_name}' not found in dataset"
            
            col = df[column_name]
            info = {
                "column": column_name,
                "data_type": str(col.dtype),
                "non_null_count": col.count(),
                "null_count": col.isnull().sum(),
                "unique_values": col.nunique(),
            }
            
            if col.dtype in ['int64', 'float64']:
                info.update({
                    "mean": col.mean(),
                    "std": col.std(),
                    "min": col.min(),
                    "max": col.max(),
                    "median": col.median()
                })
            elif col.dtype == 'object':
                info["sample_values"] = col.dropna().unique()[:10].tolist()
                
            return f"Column '{column_name}' info: {info}"
        else:
            # Return info for all columns
            column_info = []
            for col in df.columns:
                col_data = df[col]
                info = f"{col}: {col_data.dtype}, Non-null: {col_data.count()}/{len(df)}"
                column_info.append(info)
            
            return f"Dataset columns info:\n" + "\n".join(column_info)
            
    except Exception as e:
        return f"Error getting column info: {str(e)}"


@tool  
def get_data_summary(file_path: str) -> str:
    """
    Get statistical summary of the dataset.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        String containing statistical summary
    """
    try:
        df = pd.read_csv(file_path)
        
        # Get basic info
        info = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }
        
        # Get data types summary
        dtype_counts = df.dtypes.value_counts().to_dict()
        info["data_types"] = {str(k): v for k, v in dtype_counts.items()}
        
        # Get missing values summary
        missing_counts = df.isnull().sum()
        missing_info = missing_counts[missing_counts > 0].to_dict()
        info["missing_values"] = missing_info
        
        # Get numeric summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            info["numeric_summary"] = df[numeric_cols].describe().to_string()
        
        return f"Data Summary: {info}"
        
    except Exception as e:
        return f"Error getting data summary: {str(e)}"


@tool
def preview_data(file_path: str, num_rows: int = 5) -> str:
    """
    Preview the first few rows of the dataset.
    
    Args:
        file_path: Path to the data file
        num_rows: Number of rows to preview (default: 5)
        
    Returns:
        String containing preview data
    """
    try:
        df = pd.read_csv(file_path)
        
        preview = df.head(num_rows).to_string()
        
        return f"Data Preview (first {num_rows} rows):\n{preview}"
        
    except Exception as e:
        return f"Error previewing data: {str(e)}"