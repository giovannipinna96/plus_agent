"""Flexible wrapper tools that handle both JSON string and dict input."""

from langchain.tools import StructuredTool
import json
from typing import List, Union, Dict, Any

# Import all tools from tool_pydantic
from .tool_pydantic import (
    # Data tools
    read_csv_file as _read_csv_file,
    read_json_file as _read_json_file,
    get_column_info as _get_column_info,
    get_data_summary as _get_data_summary,
    preview_data as _preview_data,
    # Manipulation tools
    create_dummy_variables as _create_dummy_variables,
    modify_column_values as _modify_column_values,
    handle_missing_values as _handle_missing_values,
    convert_data_types as _convert_data_types,
    # ML tools
    train_regression_model as _train_regression_model,
    train_svm_model as _train_svm_model,
    train_random_forest_model as _train_random_forest_model,
    train_knn_model as _train_knn_model,
    evaluate_model as _evaluate_model,
    # Titanic specific tools
    calculate_survival_rate_by_group as _calculate_survival_rate_by_group,
    get_statistics_for_profile as _get_statistics_for_profile,
    calculate_survival_probability_by_features as _calculate_survival_probability_by_features,
    get_fare_estimate_by_profile as _get_fare_estimate_by_profile,
    count_passengers_by_criteria as _count_passengers_by_criteria,
    # Operations tools
    filter_data as _filter_data,
    perform_math_operations as _perform_math_operations,
    string_operations as _string_operations,
    aggregate_data as _aggregate_data,
)


def make_flexible_tool(tool: StructuredTool) -> StructuredTool:
    """
    Create a flexible wrapper that accepts both JSON string and dict input.

    This wrapper automatically detects the input type:
    - If input is a JSON string, it parses it to a dict
    - If input is already a dict, it uses it directly
    - Then passes the dict parameters to the original tool function

    Args:
        tool: The StructuredTool to wrap

    Returns:
        A new StructuredTool with flexible input handling
    """
    def wrapper(params: Union[str, Dict[str, Any]]) -> str:
        """
        Flexible wrapper that handles both string and dict input.

        Args:
            params: Either a JSON string or a dict containing tool parameters

        Returns:
            Result from the original tool or an error message
        """
        try:
            # Check if params is a string (JSON)
            if isinstance(params, str):
                # Parse JSON string to dictionary
                args = json.loads(params)
            # Check if params is already a dict
            elif isinstance(params, dict):
                # Use dict directly
                args = params
            else:
                return f"Error: Expected string or dict, got {type(params).__name__}"

            # Call original tool function with parsed/direct arguments
            return tool.func(**args)

        except json.JSONDecodeError as e:
            return f"Error parsing JSON string: {str(e)}. Please provide valid JSON format."
        except TypeError as e:
            return f"Error in function arguments: {str(e)}. Check parameter names and types."
        except Exception as e:
            return f"Error executing tool '{tool.name}': {str(e)}"

    # Create new StructuredTool with wrapper function
    return StructuredTool.from_function(
        func=wrapper,
        name=tool.name,
        description=tool.description
    )


def make_flexible_enhanced_tool(tool: StructuredTool) -> StructuredTool:
    """
    Create an enhanced flexible wrapper with additional validation and cleaning.

    This wrapper provides:
    - Accepts both JSON string and dict input
    - Automatic input cleaning (removes quotes, whitespace)
    - Enhanced error messages
    - Input validation

    Args:
        tool: The StructuredTool to wrap

    Returns:
        A new StructuredTool with enhanced flexible input handling
    """
    def enhanced_wrapper(params: Union[str, Dict[str, Any]]) -> str:
        """
        Enhanced flexible wrapper with validation and cleaning.

        Args:
            params: Either a JSON string or a dict containing tool parameters

        Returns:
            Result from the original tool or a detailed error message
        """
        try:
            # Handle string input
            if isinstance(params, str):
                # Validate input is not empty
                if not params or params.strip() == "":
                    return f"Error: Empty string provided to '{tool.name}'. Expected JSON string or dict."

                # Clean input string
                params = params.strip()

                # Remove surrounding quotes if present (common LLM mistake)
                if params.startswith('"') and params.endswith('"'):
                    params = params[1:-1]
                if params.startswith("'") and params.endswith("'"):
                    params = params[1:-1]

                # Parse JSON string
                args = json.loads(params)

                # Validate args is a dictionary
                if not isinstance(args, dict):
                    return f"Error: Expected JSON object (dictionary), got {type(args).__name__}"

            # Handle dict input
            elif isinstance(params, dict):
                # Use dict directly
                args = params

            # Handle invalid input type
            else:
                return (
                    f"Error: Invalid input type for '{tool.name}'\n"
                    f"Expected: JSON string or dict\n"
                    f"Received: {type(params).__name__}"
                )

            # Call original tool function
            result = tool.func(**args)

            return result

        except json.JSONDecodeError as e:
            return (
                f"JSON parsing error in '{tool.name}': {str(e)}\n"
                f"Input received (first 200 chars): {str(params)[:200]}...\n"
                f"Expected format: Valid JSON string or dict"
            )
        except TypeError as e:
            return (
                f"Parameter error in '{tool.name}': {str(e)}\n"
                f"Check that all required parameters are provided with correct names\n"
                f"Input type: {type(params).__name__}"
            )
        except KeyError as e:
            return (
                f"Missing required parameter in '{tool.name}': {str(e)}\n"
                f"Input received: {str(params)[:200]}..."
            )
        except Exception as e:
            return (
                f"Execution error in '{tool.name}': {str(e)}\n"
                f"Tool: {tool.name}\n"
                f"Error type: {type(e).__name__}\n"
                f"Input type: {type(params).__name__}"
            )

    return StructuredTool.from_function(
        func=enhanced_wrapper,
        name=f"{tool.name}_flex",
        description=f"{tool.description} [Accepts JSON string or dict]"
    )


# ========== CREATE FLEXIBLE WRAPPED TOOLS ==========

# Data Tools
read_csv_file_flex = make_flexible_tool(_read_csv_file)
read_json_file_flex = make_flexible_tool(_read_json_file)
get_column_info_flex = make_flexible_tool(_get_column_info)
get_data_summary_flex = make_flexible_tool(_get_data_summary)
preview_data_flex = make_flexible_tool(_preview_data)

# Manipulation Tools
create_dummy_variables_flex = make_flexible_tool(_create_dummy_variables)
modify_column_values_flex = make_flexible_tool(_modify_column_values)
handle_missing_values_flex = make_flexible_tool(_handle_missing_values)
convert_data_types_flex = make_flexible_tool(_convert_data_types)

# ML Tools
train_regression_model_flex = make_flexible_tool(_train_regression_model)
train_svm_model_flex = make_flexible_tool(_train_svm_model)
train_random_forest_model_flex = make_flexible_tool(_train_random_forest_model)
train_knn_model_flex = make_flexible_tool(_train_knn_model)
evaluate_model_flex = make_flexible_tool(_evaluate_model)

# Titanic Specific Tools
calculate_survival_rate_by_group_flex = make_flexible_tool(_calculate_survival_rate_by_group)
get_statistics_for_profile_flex = make_flexible_tool(_get_statistics_for_profile)
calculate_survival_probability_by_features_flex = make_flexible_tool(_calculate_survival_probability_by_features)
get_fare_estimate_by_profile_flex = make_flexible_tool(_get_fare_estimate_by_profile)
count_passengers_by_criteria_flex = make_flexible_tool(_count_passengers_by_criteria)

# Operations Tools
filter_data_flex = make_flexible_tool(_filter_data)
perform_math_operations_flex = make_flexible_tool(_perform_math_operations)
string_operations_flex = make_flexible_tool(_string_operations)
aggregate_data_flex = make_flexible_tool(_aggregate_data)


# ========== CREATE ENHANCED FLEXIBLE WRAPPED TOOLS ==========

# Data Tools (Enhanced)
read_csv_file_flex_enhanced = make_flexible_enhanced_tool(_read_csv_file)
read_json_file_flex_enhanced = make_flexible_enhanced_tool(_read_json_file)
get_column_info_flex_enhanced = make_flexible_enhanced_tool(_get_column_info)
get_data_summary_flex_enhanced = make_flexible_enhanced_tool(_get_data_summary)
preview_data_flex_enhanced = make_flexible_enhanced_tool(_preview_data)

# Manipulation Tools (Enhanced)
create_dummy_variables_flex_enhanced = make_flexible_enhanced_tool(_create_dummy_variables)
modify_column_values_flex_enhanced = make_flexible_enhanced_tool(_modify_column_values)
handle_missing_values_flex_enhanced = make_flexible_enhanced_tool(_handle_missing_values)
convert_data_types_flex_enhanced = make_flexible_enhanced_tool(_convert_data_types)

# ML Tools (Enhanced)
train_regression_model_flex_enhanced = make_flexible_enhanced_tool(_train_regression_model)
train_svm_model_flex_enhanced = make_flexible_enhanced_tool(_train_svm_model)
train_random_forest_model_flex_enhanced = make_flexible_enhanced_tool(_train_random_forest_model)
train_knn_model_flex_enhanced = make_flexible_enhanced_tool(_train_knn_model)
evaluate_model_flex_enhanced = make_flexible_enhanced_tool(_evaluate_model)

# Titanic Specific Tools (Enhanced)
calculate_survival_rate_by_group_flex_enhanced = make_flexible_enhanced_tool(_calculate_survival_rate_by_group)
get_statistics_for_profile_flex_enhanced = make_flexible_enhanced_tool(_get_statistics_for_profile)
calculate_survival_probability_by_features_flex_enhanced = make_flexible_enhanced_tool(_calculate_survival_probability_by_features)
get_fare_estimate_by_profile_flex_enhanced = make_flexible_enhanced_tool(_get_fare_estimate_by_profile)
count_passengers_by_criteria_flex_enhanced = make_flexible_enhanced_tool(_count_passengers_by_criteria)

# Operations Tools (Enhanced)
filter_data_flex_enhanced = make_flexible_enhanced_tool(_filter_data)
perform_math_operations_flex_enhanced = make_flexible_enhanced_tool(_perform_math_operations)
string_operations_flex_enhanced = make_flexible_enhanced_tool(_string_operations)
aggregate_data_flex_enhanced = make_flexible_enhanced_tool(_aggregate_data)


# ========== TOOL COLLECTIONS ==========

ALL_FLEXIBLE_TOOLS: List[StructuredTool] = [
    # Data tools
    read_csv_file_flex,
    read_json_file_flex,
    get_column_info_flex,
    get_data_summary_flex,
    preview_data_flex,
    # Manipulation tools
    create_dummy_variables_flex,
    modify_column_values_flex,
    handle_missing_values_flex,
    convert_data_types_flex,
    # ML tools
    train_regression_model_flex,
    train_svm_model_flex,
    train_random_forest_model_flex,
    train_knn_model_flex,
    evaluate_model_flex,
    # Titanic specific tools
    calculate_survival_rate_by_group_flex,
    get_statistics_for_profile_flex,
    calculate_survival_probability_by_features_flex,
    get_fare_estimate_by_profile_flex,
    count_passengers_by_criteria_flex,
    # Operations tools
    filter_data_flex,
    perform_math_operations_flex,
    string_operations_flex,
    aggregate_data_flex,
]

ALL_FLEXIBLE_ENHANCED_TOOLS: List[StructuredTool] = [
    # Data tools
    read_csv_file_flex_enhanced,
    read_json_file_flex_enhanced,
    get_column_info_flex_enhanced,
    get_data_summary_flex_enhanced,
    preview_data_flex_enhanced,
    # Manipulation tools
    create_dummy_variables_flex_enhanced,
    modify_column_values_flex_enhanced,
    handle_missing_values_flex_enhanced,
    convert_data_types_flex_enhanced,
    # ML tools
    train_regression_model_flex_enhanced,
    train_svm_model_flex_enhanced,
    train_random_forest_model_flex_enhanced,
    train_knn_model_flex_enhanced,
    evaluate_model_flex_enhanced,
    # Titanic specific tools
    calculate_survival_rate_by_group_flex_enhanced,
    get_statistics_for_profile_flex_enhanced,
    calculate_survival_probability_by_features_flex_enhanced,
    get_fare_estimate_by_profile_flex_enhanced,
    count_passengers_by_criteria_flex_enhanced,
    # Operations tools
    filter_data_flex_enhanced,
    perform_math_operations_flex_enhanced,
    string_operations_flex_enhanced,
    aggregate_data_flex_enhanced,
]


# ========== UTILITY FUNCTIONS ==========

def get_flexible_tool_by_name(name: str) -> StructuredTool:
    """
    Get a flexible-wrapped tool by its name.

    Args:
        name: The name of the tool

    Returns:
        The flexible-wrapped StructuredTool

    Raises:
        ValueError: If tool name not found
    """
    for tool in ALL_FLEXIBLE_TOOLS:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool '{name}' not found in flexible tools")


def get_flexible_enhanced_tool_by_name(name: str) -> StructuredTool:
    """
    Get an enhanced flexible-wrapped tool by its base name.

    Args:
        name: The base name of the tool (without '_flex' suffix)

    Returns:
        The enhanced flexible-wrapped StructuredTool

    Raises:
        ValueError: If tool name not found
    """
    flex_name = f"{name}_flex"
    for tool in ALL_FLEXIBLE_ENHANCED_TOOLS:
        if tool.name == flex_name:
            return tool
    raise ValueError(f"Enhanced flexible tool '{flex_name}' not found")


def list_flexible_tools() -> List[str]:
    """
    List all available flexible-wrapped tool names.

    Returns:
        List of tool names
    """
    return [tool.name for tool in ALL_FLEXIBLE_TOOLS]


def list_flexible_enhanced_tools() -> List[str]:
    """
    List all available enhanced flexible-wrapped tool names.

    Returns:
        List of tool names
    """
    return [tool.name for tool in ALL_FLEXIBLE_ENHANCED_TOOLS]


# ========== EXAMPLE USAGE ==========

"""
Example usage of flexible tools:

# Usage with JSON string
from plus_agent.tools.tool_flexible_wrapper import handle_missing_values_flex

result1 = handle_missing_values_flex('{"file_path": "data.csv", "column_name": "Age", "method": "mean"}')
print(result1)

# Usage with dict
result2 = handle_missing_values_flex({
    "file_path": "data.csv",
    "column_name": "Age",
    "method": "mean"
})
print(result2)

# Both will produce the same result!

# Using the enhanced version for better error messages
from plus_agent.tools.tool_flexible_wrapper import handle_missing_values_flex_enhanced

result3 = handle_missing_values_flex_enhanced({
    "file_path": "data.csv",
    "column_name": "Age",
    "method": "mean"
})
print(result3)
"""
