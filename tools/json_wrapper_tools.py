"""JSON wrapper tools that accept JSON string input for LangChain ReAct agents."""

import json
from typing import Optional, Union
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

# Import original tools from manipulation_tools
from .manipulation_tools import (
    handle_missing_values as _handle_missing_values,
    create_dummy_variables as _create_dummy_variables,
    modify_column_values as _modify_column_values,
    convert_data_types as _convert_data_types
)

# Import original tools from operations_tools
from .operations_tools import (
    filter_data as _filter_data,
    perform_math_operations as _perform_math_operations,
    string_operations as _string_operations,
    aggregate_data as _aggregate_data
)

# Import original tools from ml_tools
from .ml_tools import (
    train_regression_model as _train_regression_model,
    train_svm_model as _train_svm_model,
    train_random_forest_model as _train_random_forest_model,
    train_knn_model as _train_knn_model,
    evaluate_model as _evaluate_model
)


# Define input schemas for tools
# Keep JSONInput for backward compatibility with non-refactored tools
class JSONInput(BaseModel):
    json_string: str = Field(description="JSON string containing tool parameters")

# Define individual input schemas for each tool
class HandleMissingValuesInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    column_name: str = Field(description="Name of the column with missing values")
    method: str = Field(description="Method to handle missing values (drop, mean, median, mode, forward_fill, backward_fill, constant)")
    fill_value: Optional[Union[str, float, int]] = Field(default=None, description="Value to use for 'constant' method")

class CreateDummyVariablesInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    column_name: str = Field(description="Name of the categorical column")
    prefix: Optional[str] = Field(default=None, description="Prefix for dummy variable names")

class ModifyColumnValuesInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    column_name: str = Field(description="Name of the column to modify")
    operation: str = Field(description="Type of operation (multiply, add, subtract, divide, replace, normalize, standardize)")
    value: Optional[Union[str, float, int]] = Field(default=None, description="Value to use in the operation")

class ConvertDataTypesInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    column_name: str = Field(description="Name of the column to convert")
    target_type: str = Field(description="Target data type (int, float, string, category, datetime)")


# Helper function to clean JSON string
def _clean_json_string(json_string: str) -> str:
    """Remove surrounding quotes from JSON string."""
    return json_string.strip().strip("'").strip('"')


# ========== DATA MANIPULATION TOOLS WRAPPERS ==========

def _handle_missing_values_wrapper(file_path: str, column_name: str, method: str, fill_value: Optional[Union[str, float, int]] = None) -> str:
    """Wrapper that calls original tool with direct parameters."""
    try:
        return _handle_missing_values(
            file_path=file_path,
            column_name=column_name,
            method=method,
            fill_value=fill_value
        )
    except Exception as e:
        return f"Error in handle_missing_values: {str(e)}"


def _create_dummy_variables_wrapper(file_path: str, column_name: str, prefix: Optional[str] = None) -> str:
    """Wrapper that calls original tool with direct parameters."""
    try:
        return _create_dummy_variables(
            file_path=file_path,
            column_name=column_name,
            prefix=prefix
        )
    except Exception as e:
        return f"Error in create_dummy_variables: {str(e)}"


def _modify_column_values_wrapper(file_path: str, column_name: str, operation: str, value: Optional[Union[str, float, int]] = None) -> str:
    """Wrapper that calls original tool with direct parameters."""
    try:
        return _modify_column_values(
            file_path=file_path,
            column_name=column_name,
            operation=operation,
            value=value
        )
    except Exception as e:
        return f"Error in modify_column_values: {str(e)}"


def _convert_data_types_wrapper(file_path: str, column_name: str, target_type: str) -> str:
    """Wrapper that calls original tool with direct parameters."""
    try:
        return _convert_data_types(
            file_path=file_path,
            column_name=column_name,
            target_type=target_type
        )
    except Exception as e:
        return f"Error in convert_data_types: {str(e)}"


# ========== FIXED WRAPPERS FOR REACT AGENTS ==========
# These wrappers accept json_string parameter from JSONInput schema

def _handle_missing_values_wrapper_fixed(json_string: str) -> str:
    """Wrapper that accepts json_string from ReAct agent."""
    try:
        json_string = _clean_json_string(json_string)
        if not json_string or json_string.isspace():
            return "Error: Received empty JSON string."
        params = json.loads(json_string)
        return _handle_missing_values(
            file_path=params['file_path'],
            column_name=params['column_name'],
            method=params['method'],
            fill_value=params.get('fill_value')
        )
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {str(e)}. Received: {repr(json_string[:200])}"
    except KeyError as e:
        return f"Missing required key: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def _create_dummy_variables_wrapper_fixed(json_string: str) -> str:
    """Wrapper that accepts json_string from ReAct agent."""
    try:
        json_string = _clean_json_string(json_string)
        params = json.loads(json_string)
        return _create_dummy_variables(
            file_path=params['file_path'],
            column_name=params['column_name'],
            prefix=params.get('prefix')
        )
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {str(e)}"
    except KeyError as e:
        return f"Missing required key: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def _modify_column_values_wrapper_fixed(json_string: str) -> str:
    """Wrapper that accepts json_string from ReAct agent."""
    try:
        json_string = _clean_json_string(json_string)
        params = json.loads(json_string)
        return _modify_column_values(
            file_path=params['file_path'],
            column_name=params['column_name'],
            operation=params['operation'],
            value=params.get('value')
        )
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {str(e)}"
    except KeyError as e:
        return f"Missing required key: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def _convert_data_types_wrapper_fixed(json_string: str) -> str:
    """Wrapper that accepts json_string from ReAct agent."""
    try:
        json_string = _clean_json_string(json_string)
        params = json.loads(json_string)
        return _convert_data_types(
            file_path=params['file_path'],
            column_name=params['column_name'],
            target_type=params['target_type']
        )
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {str(e)}"
    except KeyError as e:
        return f"Missing required key: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


# ========== DATA OPERATIONS TOOLS WRAPPERS ==========

def _filter_data_wrapper(json_string: str) -> str:
    """Wrapper that parses JSON string and calls original tool."""
    try:
        json_string = _clean_json_string(json_string)
        params = json.loads(json_string)
        return _filter_data(
            file_path=params['file_path'],
            column_name=params['column_name'],
            condition=params['condition'],
            value=params['value']
        )
    except json.JSONDecodeError as e:
        return f"Error parsing JSON input: {str(e)}"
    except KeyError as e:
        return f"Missing required key in JSON: {str(e)}"
    except Exception as e:
        return f"Error in filter_data: {str(e)}"


def _perform_math_operations_wrapper(json_string: str) -> str:
    """Wrapper that parses JSON string and calls original tool."""
    try:
        json_string = _clean_json_string(json_string)
        params = json.loads(json_string)
        return _perform_math_operations(
            file_path=params['file_path'],
            operation=params['operation'],
            column1=params['column1'],
            column2=params.get('column2'),
            value=params.get('value')
        )
    except json.JSONDecodeError as e:
        return f"Error parsing JSON input: {str(e)}"
    except KeyError as e:
        return f"Missing required key in JSON: {str(e)}"
    except Exception as e:
        return f"Error in perform_math_operations: {str(e)}"


def _string_operations_wrapper(json_string: str) -> str:
    """Wrapper that parses JSON string and calls original tool."""
    try:
        json_string = _clean_json_string(json_string)
        params = json.loads(json_string)
        return _string_operations(
            file_path=params['file_path'],
            column_name=params['column_name'],
            operation=params['operation'],
            parameter=params.get('parameter')
        )
    except json.JSONDecodeError as e:
        return f"Error parsing JSON input: {str(e)}"
    except KeyError as e:
        return f"Missing required key in JSON: {str(e)}"
    except Exception as e:
        return f"Error in string_operations: {str(e)}"


def _aggregate_data_wrapper(json_string: str) -> str:
    """Wrapper that parses JSON string and calls original tool."""
    try:
        json_string = _clean_json_string(json_string)
        params = json.loads(json_string)
        return _aggregate_data(
            file_path=params['file_path'],
            group_by_columns=params['group_by_columns'],
            agg_column=params['agg_column'],
            agg_function=params['agg_function']
        )
    except json.JSONDecodeError as e:
        return f"Error parsing JSON input: {str(e)}"
    except KeyError as e:
        return f"Missing required key in JSON: {str(e)}"
    except Exception as e:
        return f"Error in aggregate_data: {str(e)}"


# ========== MACHINE LEARNING TOOLS WRAPPERS ==========

def _train_regression_model_wrapper(json_string: str) -> str:
    """Wrapper that parses JSON string and calls original tool."""
    try:
        json_string = _clean_json_string(json_string)
        params = json.loads(json_string)
        return _train_regression_model(
            file_path=params['file_path'],
            target_column=params['target_column'],
            feature_columns=params['feature_columns'],
            model_type=params.get('model_type', 'linear')
        )
    except json.JSONDecodeError as e:
        return f"Error parsing JSON input: {str(e)}"
    except KeyError as e:
        return f"Missing required key in JSON: {str(e)}"
    except Exception as e:
        return f"Error in train_regression_model: {str(e)}"


def _train_svm_model_wrapper(json_string: str) -> str:
    """Wrapper that parses JSON string and calls original tool."""
    try:
        json_string = _clean_json_string(json_string)
        params = json.loads(json_string)
        return _train_svm_model(
            file_path=params['file_path'],
            target_column=params['target_column'],
            feature_columns=params['feature_columns'],
            task_type=params.get('task_type', 'classification')
        )
    except json.JSONDecodeError as e:
        return f"Error parsing JSON input: {str(e)}"
    except KeyError as e:
        return f"Missing required key in JSON: {str(e)}"
    except Exception as e:
        return f"Error in train_svm_model: {str(e)}"


def _train_random_forest_model_wrapper(json_string: str) -> str:
    """Wrapper that parses JSON string and calls original tool."""
    try:
        json_string = _clean_json_string(json_string)
        params = json.loads(json_string)
        return _train_random_forest_model(
            file_path=params['file_path'],
            target_column=params['target_column'],
            feature_columns=params['feature_columns'],
            task_type=params.get('task_type', 'classification')
        )
    except json.JSONDecodeError as e:
        return f"Error parsing JSON input: {str(e)}"
    except KeyError as e:
        return f"Missing required key in JSON: {str(e)}"
    except Exception as e:
        return f"Error in train_random_forest_model: {str(e)}"


def _train_knn_model_wrapper(json_string: str) -> str:
    """Wrapper that parses JSON string and calls original tool."""
    try:
        json_string = _clean_json_string(json_string)
        params = json.loads(json_string)
        return _train_knn_model(
            file_path=params['file_path'],
            target_column=params['target_column'],
            feature_columns=params['feature_columns'],
            task_type=params.get('task_type', 'classification'),
            n_neighbors=params.get('n_neighbors', 5)
        )
    except json.JSONDecodeError as e:
        return f"Error parsing JSON input: {str(e)}"
    except KeyError as e:
        return f"Missing required key in JSON: {str(e)}"
    except Exception as e:
        return f"Error in train_knn_model: {str(e)}"


def _evaluate_model_wrapper(json_string: str) -> str:
    """Wrapper that parses JSON string and calls original tool."""
    try:
        json_string = _clean_json_string(json_string)
        params = json.loads(json_string)
        return _evaluate_model(
            model_path=params['model_path'],
            test_data_path=params['test_data_path'],
            target_column=params['target_column'],
            feature_columns=params['feature_columns']
        )
    except json.JSONDecodeError as e:
        return f"Error parsing JSON input: {str(e)}"
    except KeyError as e:
        return f"Missing required key in JSON: {str(e)}"
    except Exception as e:
        return f"Error in evaluate_model: {str(e)}"


# ========== CREATE STRUCTURED TOOL INSTANCES ==========

# Data Manipulation Tools
# Using @tool decorator WITHOUT args_schema for raw string input
from langchain.tools import tool

@tool
def handle_missing_values(tool_input: str) -> str:
    """Handle missing values in a column using various methods (drop, mean, median, mode, forward_fill, backward_fill, constant).
    Input: JSON string with format: {\"file_path\": \"...\", \"column_name\": \"...\", \"method\": \"...\", \"fill_value\": \"...\" (optional)}"""
    return _handle_missing_values_wrapper_fixed(tool_input)

@tool
def create_dummy_variables(tool_input: str) -> str:
    """Create one-hot encoded dummy variables for a categorical column.
    Input: JSON string with format: {\"file_path\": \"...\", \"column_name\": \"...\", \"prefix\": \"...\" (optional)}"""
    return _create_dummy_variables_wrapper_fixed(tool_input)

@tool
def modify_column_values(tool_input: str) -> str:
    """Modify values in a column using various operations (multiply, add, subtract, divide, replace, normalize, standardize).
    Input: JSON string with format: {\"file_path\": \"...\", \"column_name\": \"...\", \"operation\": \"...\", \"value\": \"...\" (optional)}"""
    return _modify_column_values_wrapper_fixed(tool_input)

@tool
def convert_data_types(tool_input: str) -> str:
    """Convert data type of a column (int, float, string, category, datetime).
    Input: JSON string with format: {\"file_path\": \"...\", \"column_name\": \"...\", \"target_type\": \"...\"}"""
    return _convert_data_types_wrapper_fixed(tool_input)

# Data Operations Tools
filter_data = StructuredTool.from_function(
    func=_filter_data_wrapper,
    name="filter_data",
    description="Filter data based on a condition. Input must be JSON string with keys: file_path, column_name, condition, value. Example: {\"file_path\": \"data.csv\", \"column_name\": \"Age\", \"condition\": \"greater_than\", \"value\": 30}",
    args_schema=JSONInput
)

perform_math_operations = StructuredTool.from_function(
    func=_perform_math_operations_wrapper,
    name="perform_math_operations",
    description="Perform mathematical operations on columns. Input must be JSON string with keys: file_path, operation, column1, column2 (optional), value (optional). Example: {\"file_path\": \"data.csv\", \"operation\": \"add\", \"column1\": \"Age\", \"value\": 10}",
    args_schema=JSONInput
)

string_operations = StructuredTool.from_function(
    func=_string_operations_wrapper,
    name="string_operations",
    description="Perform string operations on a text column. Input must be JSON string with keys: file_path, column_name, operation, parameter (optional). Example: {\"file_path\": \"data.csv\", \"column_name\": \"Name\", \"operation\": \"upper\"}",
    args_schema=JSONInput
)

aggregate_data = StructuredTool.from_function(
    func=_aggregate_data_wrapper,
    name="aggregate_data",
    description="Aggregate data by grouping columns. Input must be JSON string with keys: file_path, group_by_columns, agg_column, agg_function. Example: {\"file_path\": \"data.csv\", \"group_by_columns\": \"pclass\", \"agg_column\": \"fare\", \"agg_function\": \"mean\"}",
    args_schema=JSONInput
)

# Machine Learning Tools
train_regression_model = StructuredTool.from_function(
    func=_train_regression_model_wrapper,
    name="train_regression_model",
    description="Train a regression model. Input must be JSON string with keys: file_path, target_column, feature_columns, model_type (optional). Example: {\"file_path\": \"data.csv\", \"target_column\": \"fare\", \"feature_columns\": \"age,pclass\", \"model_type\": \"linear\"}",
    args_schema=JSONInput
)

train_svm_model = StructuredTool.from_function(
    func=_train_svm_model_wrapper,
    name="train_svm_model",
    description="Train a Support Vector Machine model. Input must be JSON string with keys: file_path, target_column, feature_columns, task_type (optional). Example: {\"file_path\": \"data.csv\", \"target_column\": \"survived\", \"feature_columns\": \"age,pclass,fare\", \"task_type\": \"classification\"}",
    args_schema=JSONInput
)

train_random_forest_model = StructuredTool.from_function(
    func=_train_random_forest_model_wrapper,
    name="train_random_forest_model",
    description="Train a Random Forest model. Input must be JSON string with keys: file_path, target_column, feature_columns, task_type (optional). Example: {\"file_path\": \"data.csv\", \"target_column\": \"survived\", \"feature_columns\": \"pclass,age,fare,sex\", \"task_type\": \"classification\"}",
    args_schema=JSONInput
)

train_knn_model = StructuredTool.from_function(
    func=_train_knn_model_wrapper,
    name="train_knn_model",
    description="Train a K-Nearest Neighbors model. Input must be JSON string with keys: file_path, target_column, feature_columns, task_type (optional), n_neighbors (optional). Example: {\"file_path\": \"data.csv\", \"target_column\": \"survived\", \"feature_columns\": \"age,pclass,fare\", \"task_type\": \"classification\", \"n_neighbors\": 5}",
    args_schema=JSONInput
)

evaluate_model = StructuredTool.from_function(
    func=_evaluate_model_wrapper,
    name="evaluate_model",
    description="Evaluate a trained model on new test data. Input must be JSON string with keys: model_path, test_data_path, target_column, feature_columns. Example: {\"model_path\": \"model.joblib\", \"test_data_path\": \"test.csv\", \"target_column\": \"survived\", \"feature_columns\": \"age,pclass,fare\"}",
    args_schema=JSONInput
)
