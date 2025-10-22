"""ReAct-style tools using @tool decorator with JSON string input."""

from langchain.tools import tool
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib


# ========== DATA TOOLS ==========

@tool("read_csv_file", return_direct=True)
def read_csv_file(params: str) -> str:
    """
    Read a CSV file and return basic information about the dataset.
    Expects JSON string with keys: file_path
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]

        df = pd.read_csv(file_path)

        info = {
            "status": "success",
            "shape": df.shape,
            "columns": list(df.columns),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }

        return f"CSV file loaded successfully. Shape: {info['shape']}, Columns: {info['columns']}"

    except Exception as e:
        return f"Error in read_csv_file: {str(e)}"


@tool("read_json_file", return_direct=True)
def read_json_file(params: str) -> str:
    """
    Read a JSON file and return basic information about the data.
    Expects JSON string with keys: file_path
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]

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
        return f"Error in read_json_file: {str(e)}"


@tool("get_column_info", return_direct=True)
def get_column_info(params: str) -> str:
    """
    Get detailed information about columns in the dataset.
    Expects JSON string with keys: file_path, column_name (optional)
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        column_name = args.get("column_name")

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
        return f"Error in get_column_info: {str(e)}"


@tool("get_data_summary", return_direct=True)
def get_data_summary(params: str) -> str:
    """
    Get statistical summary of the dataset.
    Expects JSON string with keys: file_path
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]

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
        return f"Error in get_data_summary: {str(e)}"


@tool("preview_data", return_direct=True)
def preview_data(params: str) -> str:
    """
    Preview the first few rows of the dataset.
    Expects JSON string with keys: file_path, num_rows (optional, default=5)
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        num_rows = args.get("num_rows", 5)

        df = pd.read_csv(file_path)
        preview = df.head(num_rows).to_string()

        return f"Data Preview (first {num_rows} rows):\n{preview}"

    except Exception as e:
        return f"Error in preview_data: {str(e)}"


# ========== MANIPULATION TOOLS ==========

@tool("create_dummy_variables", return_direct=True)
def create_dummy_variables(params: str) -> str:
    """
    Create dummy variables for a categorical column.
    Expects JSON string with keys: file_path, column_name, prefix (optional)
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        column_name = args["column_name"]
        prefix = args.get("prefix")

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
        return f"Error in create_dummy_variables: {str(e)}"


@tool("modify_column_values", return_direct=True)
def modify_column_values(params: str) -> str:
    """
    Modify values in a column using various operations.
    Expects JSON string with keys: file_path, column_name, operation, value (optional)
    Operations: multiply, add, subtract, divide, replace, normalize, standardize
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        column_name = args["column_name"]
        operation = args["operation"]
        value = args.get("value")

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
        return f"Error in modify_column_values: {str(e)}"


@tool("handle_missing_values", return_direct=True)
def handle_missing_values(params: str) -> str:
    """
    Handle missing values in a column using various methods.
    Expects JSON string with keys: file_path, column_name, method, fill_value (optional)
    Methods: drop, mean, median, mode, forward_fill, backward_fill, constant
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        column_name = args["column_name"]
        method = args["method"]
        fill_value = args.get("fill_value")

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
        return f"Error in handle_missing_values: {str(e)}"


@tool("convert_data_types", return_direct=True)
def convert_data_types(params: str) -> str:
    """
    Convert data type of a column.
    Expects JSON string with keys: file_path, column_name, target_type
    Target types: int, float, string, category, datetime
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        column_name = args["column_name"]
        target_type = args["target_type"]

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
        return f"Error in convert_data_types: {str(e)}"


# ========== OPERATIONS TOOLS ==========

@tool("filter_data", return_direct=True)
def filter_data(params: str) -> str:
    """
    Filter data based on a condition.
    Expects JSON string with keys: file_path, column_name, condition, value
    Conditions: equals, not_equals, greater_than, less_than, greater_equal, less_equal, contains
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        column_name = args["column_name"]
        condition = args["condition"]
        value = args["value"]

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
        return f"Error in filter_data: {str(e)}"


@tool("perform_math_operations", return_direct=True)
def perform_math_operations(params: str) -> str:
    """
    Perform mathematical operations on columns.
    Expects JSON string with keys: file_path, operation, column1, column2 (optional), value (optional)
    Operations: add, subtract, multiply, divide, power, square, sqrt, log, abs
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        operation = args["operation"]
        column1 = args["column1"]
        column2 = args.get("column2")
        value = args.get("value")

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
        return f"Error in perform_math_operations: {str(e)}"


@tool("string_operations", return_direct=True)
def string_operations(params: str) -> str:
    """
    Perform string operations on a text column.
    Expects JSON string with keys: file_path, column_name, operation, parameter (optional)
    Operations: upper, lower, title, length, contains_count, split, replace, strip
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        column_name = args["column_name"]
        operation = args["operation"]
        parameter = args.get("parameter")

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
        return f"Error in string_operations: {str(e)}"


@tool("aggregate_data", return_direct=True)
def aggregate_data(params: str) -> str:
    """
    Aggregate data by grouping columns.
    Expects JSON string with keys: file_path, group_by_columns, agg_column, agg_function
    Functions: mean, sum, count, min, max, std, median
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        group_by_columns = args["group_by_columns"]
        agg_column = args["agg_column"]
        agg_function = args["agg_function"]

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
        return f"Error in aggregate_data: {str(e)}"


# ========== ML TOOLS ==========

@tool("train_regression_model", return_direct=True)
def train_regression_model(params: str) -> str:
    """
    Train a regression model.
    Expects JSON string with keys: file_path, target_column, feature_columns, model_type (optional, default='linear')
    Model types: linear, random_forest
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        target_column = args["target_column"]
        feature_columns = args["feature_columns"]
        model_type = args.get("model_type", "linear")

        df = pd.read_csv(file_path)

        # Parse feature columns
        features = [col.strip() for col in feature_columns.split(',')]

        # Check if columns exist
        for col in features + [target_column]:
            if col not in df.columns:
                return f"Column '{col}' not found in dataset"

        # Prepare data
        X = df[features]
        y = df[target_column]

        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        # Handle categorical variables in features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            return f"Unknown regression model type '{model_type}'"

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        # Save model
        model_path = file_path.replace('.csv', f'_{model_type}_regression_model.joblib')
        joblib.dump(model, model_path)

        results = {
            "model_type": f"{model_type} regression",
            "features": features,
            "target": target_column,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "cv_r2_mean": round(cv_scores.mean(), 4),
            "cv_r2_std": round(cv_scores.std(), 4),
            "model_saved": model_path
        }

        return f"Regression model trained: {results}"

    except Exception as e:
        return f"Error in train_regression_model: {str(e)}"


@tool("train_svm_model", return_direct=True)
def train_svm_model(params: str) -> str:
    """
    Train a Support Vector Machine model.
    Expects JSON string with keys: file_path, target_column, feature_columns, task_type (optional, default='classification')
    Task types: classification, regression
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        target_column = args["target_column"]
        feature_columns = args["feature_columns"]
        task_type = args.get("task_type", "classification")

        df = pd.read_csv(file_path)

        # Parse feature columns
        features = [col.strip() for col in feature_columns.split(',')]

        # Check if columns exist
        for col in features + [target_column]:
            if col not in df.columns:
                return f"Column '{col}' not found in dataset"

        # Prepare data
        X = df[features]
        y = df[target_column]

        # Handle missing values
        X = X.fillna(X.mean())
        if task_type == "classification":
            y = y.fillna(y.mode()[0])
        else:
            y = y.fillna(y.mean())

        # Handle categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        if task_type == "classification" and y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        if task_type == "classification":
            model = SVC(kernel='rbf', random_state=42)
            scoring_metric = 'accuracy'
        else:
            model = SVR(kernel='rbf')
            scoring_metric = 'r2'

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            metrics = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics = {
                "mse": round(mse, 4),
                "r2_score": round(r2, 4)
            }

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring_metric)

        # Save model
        model_path = file_path.replace('.csv', f'_svm_{task_type}_model.joblib')
        joblib.dump(model, model_path)

        results = {
            "model_type": f"SVM {task_type}",
            "features": features,
            "target": target_column,
            "train_size": len(X_train),
            "test_size": len(X_test),
            **metrics,
            "cv_score_mean": round(cv_scores.mean(), 4),
            "cv_score_std": round(cv_scores.std(), 4),
            "model_saved": model_path
        }

        return f"SVM model trained: {results}"

    except Exception as e:
        return f"Error in train_svm_model: {str(e)}"


@tool("train_random_forest_model", return_direct=True)
def train_random_forest_model(params: str) -> str:
    """
    Train a Random Forest model.
    Expects JSON string with keys: file_path, target_column, feature_columns, task_type (optional, default='classification')
    Task types: classification, regression
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        target_column = args["target_column"]
        feature_columns = args["feature_columns"]
        task_type = args.get("task_type", "classification")

        df = pd.read_csv(file_path)

        # Parse feature columns
        features = [col.strip() for col in feature_columns.split(',')]

        # Check if columns exist
        for col in features + [target_column]:
            if col not in df.columns:
                return f"Column '{col}' not found in dataset"

        # Prepare data
        X = df[features]
        y = df[target_column]

        # Handle missing values
        X = X.fillna(X.mean())
        if task_type == "classification":
            y = y.fillna(y.mode()[0])
        else:
            y = y.fillna(y.mean())

        # Handle categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        if task_type == "classification" and y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            scoring_metric = 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            scoring_metric = 'r2'

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            metrics = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics = {
                "mse": round(mse, 4),
                "r2_score": round(r2, 4)
            }

        # Feature importance
        feature_importance = dict(zip(features, [round(imp, 4) for imp in model.feature_importances_]))

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring_metric)

        # Save model
        model_path = file_path.replace('.csv', f'_rf_{task_type}_model.joblib')
        joblib.dump(model, model_path)

        results = {
            "model_type": f"Random Forest {task_type}",
            "features": features,
            "target": target_column,
            "train_size": len(X_train),
            "test_size": len(X_test),
            **metrics,
            "cv_score_mean": round(cv_scores.mean(), 4),
            "cv_score_std": round(cv_scores.std(), 4),
            "feature_importance": feature_importance,
            "model_saved": model_path
        }

        return f"Random Forest model trained: {results}"

    except Exception as e:
        return f"Error in train_random_forest_model: {str(e)}"


@tool("train_knn_model", return_direct=True)
def train_knn_model(params: str) -> str:
    """
    Train a K-Nearest Neighbors model.
    Expects JSON string with keys: file_path, target_column, feature_columns, task_type (optional, default='classification'), n_neighbors (optional, default=5)
    Task types: classification, regression
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        target_column = args["target_column"]
        feature_columns = args["feature_columns"]
        task_type = args.get("task_type", "classification")
        n_neighbors = args.get("n_neighbors", 5)

        df = pd.read_csv(file_path)

        # Parse feature columns
        features = [col.strip() for col in feature_columns.split(',')]

        # Check if columns exist
        for col in features + [target_column]:
            if col not in df.columns:
                return f"Column '{col}' not found in dataset"

        # Prepare data
        X = df[features]
        y = df[target_column]

        # Handle missing values
        X = X.fillna(X.mean())
        if task_type == "classification":
            y = y.fillna(y.mode()[0])
        else:
            y = y.fillna(y.mean())

        # Handle categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        if task_type == "classification" and y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        if task_type == "classification":
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            scoring_metric = 'accuracy'
        else:
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
            scoring_metric = 'r2'

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            metrics = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics = {
                "mse": round(mse, 4),
                "r2_score": round(r2, 4)
            }

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring_metric)

        # Save model
        model_path = file_path.replace('.csv', f'_knn_{task_type}_model.joblib')
        joblib.dump(model, model_path)

        results = {
            "model_type": f"KNN {task_type}",
            "n_neighbors": n_neighbors,
            "features": features,
            "target": target_column,
            "train_size": len(X_train),
            "test_size": len(X_test),
            **metrics,
            "cv_score_mean": round(cv_scores.mean(), 4),
            "cv_score_std": round(cv_scores.std(), 4),
            "model_saved": model_path
        }

        return f"KNN model trained: {results}"

    except Exception as e:
        return f"Error in train_knn_model: {str(e)}"


@tool("evaluate_model", return_direct=True)
def evaluate_model(params: str) -> str:
    """
    Evaluate a trained model on new test data.
    Expects JSON string with keys: model_path, test_data_path, target_column, feature_columns
    """
    try:
        args = json.loads(params)
        model_path = args["model_path"]
        test_data_path = args["test_data_path"]
        target_column = args["target_column"]
        feature_columns = args["feature_columns"]

        # Load the model
        model = joblib.load(model_path)

        # Load test data
        df = pd.read_csv(test_data_path)

        # Parse feature columns
        features = [col.strip() for col in feature_columns.split(',')]

        # Check if columns exist
        for col in features + [target_column]:
            if col not in df.columns:
                return f"Column '{col}' not found in test dataset"

        # Prepare test data
        X = df[features]
        y = df[target_column]

        # Handle missing values
        X = X.fillna(X.mean())

        # Handle categorical variables (same as training)
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # Make predictions
        y_pred = model.predict(X)

        # Determine if this is classification or regression based on model type
        model_name = str(type(model).__name__)
        is_classification = any(clf in model_name for clf in ['Classifier', 'SVC'])

        # Calculate appropriate metrics
        if is_classification:
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

            results = {
                "model_type": model_name,
                "task": "classification",
                "test_samples": len(X),
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        else:
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mse)

            results = {
                "model_type": model_name,
                "task": "regression",
                "test_samples": len(X),
                "mse": round(mse, 4),
                "rmse": round(rmse, 4),
                "r2_score": round(r2, 4)
            }

        return f"Model evaluation results: {results}"

    except Exception as e:
        return f"Error in evaluate_model: {str(e)}"


# ========== TITANIC SPECIFIC TOOLS ==========

@tool("calculate_survival_rate_by_group", return_direct=True)
def calculate_survival_rate_by_group(params: str) -> str:
    """
    Calculate survival rates by a specific grouping column.
    Expects JSON string with keys: file_path, group_column
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        group_column = args["group_column"]

        df = pd.read_csv(file_path)

        if group_column not in df.columns:
            return f"Column '{group_column}' not found in dataset"

        if 'survived' not in df.columns:
            return "Column 'survived' not found in dataset"

        # Calculate survival rates
        survival_stats = df.groupby(group_column).agg({
            'survived': ['count', 'sum', 'mean']
        }).round(4)

        # Flatten column names
        survival_stats.columns = ['total_passengers', 'survivors', 'survival_rate']
        survival_stats = survival_stats.reset_index()

        # Convert to readable format
        results = {}
        for _, row in survival_stats.iterrows():
            group_value = row[group_column]
            results[f"{group_column}_{group_value}"] = {
                "total_passengers": int(row['total_passengers']),
                "survivors": int(row['survivors']),
                "survival_rate": f"{row['survival_rate']:.1%}"
            }

        return f"Survival rates by {group_column}: {results}"

    except Exception as e:
        return f"Error in calculate_survival_rate_by_group: {str(e)}"


@tool("get_statistics_for_profile", return_direct=True)
def get_statistics_for_profile(params: str) -> str:
    """
    Get statistics for a specific passenger profile.
    Expects JSON string with keys: file_path, filters, target_column
    Filters format: "column=value,column=value"
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        filters = args["filters"]
        target_column = args["target_column"]

        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            return f"Column '{target_column}' not found in dataset"

        # Apply filters
        filtered_df = df.copy()
        filter_conditions = []

        for filter_str in filters.split(','):
            if '=' in filter_str:
                col, val = filter_str.strip().split('=')
                col, val = col.strip(), val.strip()

                if col not in df.columns:
                    return f"Filter column '{col}' not found in dataset"

                # Convert value to appropriate type
                if df[col].dtype in ['int64', 'float64']:
                    try:
                        val = float(val)
                    except:
                        pass
                elif val.lower() in ['true', 'false']:
                    val = val.lower() == 'true'

                filtered_df = filtered_df[filtered_df[col] == val]
                filter_conditions.append(f"{col}={val}")

        if len(filtered_df) == 0:
            return f"No passengers found matching filters: {filter_conditions}"

        # Calculate statistics
        if df[target_column].dtype in ['int64', 'float64']:
            stats = {
                "count": len(filtered_df),
                "mean": filtered_df[target_column].mean(),
                "median": filtered_df[target_column].median(),
                "std": filtered_df[target_column].std(),
                "min": filtered_df[target_column].min(),
                "max": filtered_df[target_column].max()
            }
        else:
            # For categorical columns
            value_counts = filtered_df[target_column].value_counts()
            stats = {
                "count": len(filtered_df),
                "value_counts": value_counts.to_dict(),
                "most_common": value_counts.index[0] if len(value_counts) > 0 else None
            }

        return f"Statistics for profile {filter_conditions} on {target_column}: {stats}"

    except Exception as e:
        return f"Error in get_statistics_for_profile: {str(e)}"


@tool("calculate_survival_probability_by_features", return_direct=True)
def calculate_survival_probability_by_features(params: str) -> str:
    """
    Calculate survival probability for specific passenger characteristics.
    Expects JSON string with keys: file_path, sex, pclass, age_range (optional, default='all')
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        sex = args["sex"]
        pclass = args["pclass"]
        age_range = args.get("age_range", "all")

        df = pd.read_csv(file_path)

        required_cols = ['survived', 'sex', 'pclass', 'age']
        for col in required_cols:
            if col not in df.columns:
                return f"Required column '{col}' not found in dataset"

        # Apply filters
        filtered_df = df.copy()

        # Filter by sex and class
        filtered_df = filtered_df[
            (filtered_df['sex'] == sex.lower()) &
            (filtered_df['pclass'] == pclass)
        ]

        # Apply age filter if specified
        if age_range != "all":
            try:
                if '-' in age_range:
                    min_age, max_age = map(int, age_range.split('-'))
                    filtered_df = filtered_df[
                        (filtered_df['age'] >= min_age) &
                        (filtered_df['age'] <= max_age)
                    ]
            except:
                return f"Invalid age_range format: {age_range}. Use format like '18-30' or 'all'"

        if len(filtered_df) == 0:
            return f"No passengers found matching criteria: sex={sex}, class={pclass}, age_range={age_range}"

        # Calculate survival statistics
        total_passengers = len(filtered_df)
        survivors = filtered_df['survived'].sum()
        survival_rate = survivors / total_passengers if total_passengers > 0 else 0

        # Additional insights
        age_stats = filtered_df['age'].describe() if not filtered_df['age'].isna().all() else None
        fare_stats = filtered_df['fare'].describe() if 'fare' in filtered_df.columns else None

        results = {
            "profile": f"{sex.title()} in {pclass}{'st' if pclass == 1 else 'nd' if pclass == 2 else 'rd'} class",
            "age_range": age_range,
            "sample_size": total_passengers,
            "survivors": int(survivors),
            "survival_probability": f"{survival_rate:.1%}",
            "survival_rate_decimal": round(survival_rate, 3)
        }

        if age_stats is not None:
            results["age_stats"] = {
                "mean": round(age_stats['mean'], 1),
                "median": round(age_stats['50%'], 1)
            }

        if fare_stats is not None:
            results["fare_stats"] = {
                "mean": round(fare_stats['mean'], 2),
                "median": round(fare_stats['50%'], 2)
            }

        return f"Survival analysis results: {results}"

    except Exception as e:
        return f"Error in calculate_survival_probability_by_features: {str(e)}"


@tool("get_fare_estimate_by_profile", return_direct=True)
def get_fare_estimate_by_profile(params: str) -> str:
    """
    Estimate fare price based on passenger profile.
    Expects JSON string with keys: file_path, sex, pclass, age
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        sex = args["sex"]
        pclass = args["pclass"]
        age = args["age"]

        df = pd.read_csv(file_path)

        required_cols = ['sex', 'pclass', 'age', 'fare']
        for col in required_cols:
            if col not in df.columns:
                return f"Required column '{col}' not found in dataset"

        # Filter by similar profiles
        similar_passengers = df[
            (df['sex'] == sex.lower()) &
            (df['pclass'] == pclass) &
            (df['age'].notna()) &
            (df['fare'].notna())
        ]

        if len(similar_passengers) == 0:
            return f"No passengers found with similar profile: sex={sex}, class={pclass}"

        # Find passengers with similar age (±5 years)
        age_similar = similar_passengers[
            (similar_passengers['age'] >= age - 5) &
            (similar_passengers['age'] <= age + 5)
        ]

        if len(age_similar) == 0:
            # If no age-similar passengers, use all passengers with same sex and class
            age_similar = similar_passengers

        # Calculate fare statistics
        fare_stats = age_similar['fare'].describe()

        results = {
            "profile": f"{age}-year-old {sex.lower()} in {pclass}{'st' if pclass == 1 else 'nd' if pclass == 2 else 'rd'} class",
            "sample_size": len(age_similar),
            "estimated_fare": round(fare_stats['mean'], 2),
            "fare_range": {
                "min": round(fare_stats['min'], 2),
                "max": round(fare_stats['max'], 2),
                "median": round(fare_stats['50%'], 2),
                "q25": round(fare_stats['25%'], 2),
                "q75": round(fare_stats['75%'], 2)
            },
            "most_likely_fare": round(fare_stats['50%'], 2)  # median is often more representative
        }

        return f"Fare estimate results: {results}"

    except Exception as e:
        return f"Error in get_fare_estimate_by_profile: {str(e)}"


@tool("count_passengers_by_criteria", return_direct=True)
def count_passengers_by_criteria(params: str) -> str:
    """
    Count passengers matching specific criteria.
    Expects JSON string with keys: file_path, criteria
    Criteria format: "column1=value1,column2=value2" or complex filters
    """
    try:
        args = json.loads(params)
        file_path = args["file_path"]
        criteria = args["criteria"]

        df = pd.read_csv(file_path)

        # Parse criteria
        filtered_df = df.copy()
        applied_filters = []

        if criteria.strip():
            for criterion in criteria.split(','):
                if '=' in criterion:
                    col, val = criterion.strip().split('=', 1)
                    col, val = col.strip(), val.strip()

                    if col not in df.columns:
                        return f"Column '{col}' not found in dataset"

                    # Handle different comparison operators
                    if val.startswith('<='):
                        val = float(val[2:])
                        filtered_df = filtered_df[filtered_df[col] <= val]
                        applied_filters.append(f"{col} <= {val}")
                    elif val.startswith('>='):
                        val = float(val[2:])
                        filtered_df = filtered_df[filtered_df[col] >= val]
                        applied_filters.append(f"{col} >= {val}")
                    elif val.startswith('<'):
                        val = float(val[1:])
                        filtered_df = filtered_df[filtered_df[col] < val]
                        applied_filters.append(f"{col} < {val}")
                    elif val.startswith('>'):
                        val = float(val[1:])
                        filtered_df = filtered_df[filtered_df[col] > val]
                        applied_filters.append(f"{col} > {val}")
                    else:
                        # Exact match
                        if df[col].dtype in ['int64', 'float64']:
                            try:
                                val = float(val)
                            except:
                                pass
                        elif val.lower() in ['true', 'false']:
                            val = val.lower() == 'true'

                        filtered_df = filtered_df[filtered_df[col] == val]
                        applied_filters.append(f"{col} = {val}")

        count = len(filtered_df)
        total = len(df)
        percentage = (count / total * 100) if total > 0 else 0

        results = {
            "total_passengers": total,
            "matching_criteria": count,
            "percentage": f"{percentage:.1f}%",
            "applied_filters": applied_filters
        }

        return f"Passenger count results: {results}"

    except Exception as e:
        return f"Error in count_passengers_by_criteria: {str(e)}"


# ========== ALL TOOLS LIST ==========

ALL_REACT_TOOLS = [
    # Data tools
    read_csv_file,
    read_json_file,
    get_column_info,
    get_data_summary,
    preview_data,
    # Manipulation tools
    create_dummy_variables,
    modify_column_values,
    handle_missing_values,
    convert_data_types,
    # Operations tools
    filter_data,
    perform_math_operations,
    string_operations,
    aggregate_data,
    # ML tools
    train_regression_model,
    train_svm_model,
    train_random_forest_model,
    train_knn_model,
    evaluate_model,
    # Titanic specific tools
    calculate_survival_rate_by_group,
    get_statistics_for_profile,
    calculate_survival_probability_by_features,
    get_fare_estimate_by_profile,
    count_passengers_by_criteria,
]
