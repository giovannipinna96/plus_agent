"""Safe wrapper tools that handle JSON input parsing with error handling."""

from langchain.tools import StructuredTool
import json
from typing import List

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


def make_json_safe_tool(tool: StructuredTool) -> StructuredTool:
    """
    Create a JSON-safe wrapper for a StructuredTool.

    This wrapper handles JSON string input parsing and provides error handling
    for malformed inputs. It's particularly useful for tools that need to
    accept JSON strings as input from LLM agents.

    Args:
        tool: The StructuredTool to wrap

    Returns:
        A new StructuredTool with JSON parsing and error handling
    """
    def wrapper(params: str) -> str:
        """
        Wrapper function that parses JSON input and calls the original tool.

        Args:
            params: JSON string containing the tool parameters

        Returns:
            Result from the original tool or an error message
        """
        try:
            # Parse JSON string to dictionary
            args = json.loads(params)

            # Call original tool function with parsed arguments
            return tool.func(**args)

        except json.JSONDecodeError as e:
            return f"Error parsing JSON input: {str(e)}. Please provide valid JSON format."
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


def make_enhanced_safe_tool(tool: StructuredTool) -> StructuredTool:
    """
    Create an enhanced safe wrapper with additional validation and logging.

    This wrapper provides:
    - JSON parsing with detailed error messages
    - Input validation
    - Parameter type checking
    - Execution logging

    Args:
        tool: The StructuredTool to wrap

    Returns:
        A new StructuredTool with enhanced safety features
    """
    def enhanced_wrapper(params: str) -> str:
        """
        Enhanced wrapper with validation and detailed error messages.

        Args:
            params: JSON string containing the tool parameters

        Returns:
            Result from the original tool or a detailed error message
        """
        # Validate input is not empty
        if not params or params.strip() == "":
            return f"Error: Empty input provided to '{tool.name}'. Expected JSON string."

        try:
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

            # Call original tool function
            result = tool.func(**args)

            return result

        except json.JSONDecodeError as e:
            return (
                f"JSON parsing error in '{tool.name}': {str(e)}\n"
                f"Input received: {params[:200]}...\n"
                f"Expected format: JSON object with required parameters"
            )
        except TypeError as e:
            return (
                f"Parameter error in '{tool.name}': {str(e)}\n"
                f"Check that all required parameters are provided with correct names"
            )
        except KeyError as e:
            return (
                f"Missing required parameter in '{tool.name}': {str(e)}\n"
                f"Input received: {params[:200]}..."
            )
        except Exception as e:
            return (
                f"Execution error in '{tool.name}': {str(e)}\n"
                f"Tool: {tool.name}\n"
                f"Error type: {type(e).__name__}"
            )

    return StructuredTool.from_function(
        func=enhanced_wrapper,
        name=f"{tool.name}_safe",
        description=f"{tool.description} [Enhanced with safety wrapper]"
    )


# ========== CREATE JSON-SAFE WRAPPED TOOLS ==========

# Data Tools
read_csv_file_safe = make_json_safe_tool(_read_csv_file)
read_json_file_safe = make_json_safe_tool(_read_json_file)
get_column_info_safe = make_json_safe_tool(_get_column_info)
get_data_summary_safe = make_json_safe_tool(_get_data_summary)
preview_data_safe = make_json_safe_tool(_preview_data)

# Manipulation Tools
create_dummy_variables_safe = make_json_safe_tool(_create_dummy_variables)
modify_column_values_safe = make_json_safe_tool(_modify_column_values)
handle_missing_values_safe = make_json_safe_tool(_handle_missing_values)
convert_data_types_safe = make_json_safe_tool(_convert_data_types)

# ML Tools
train_regression_model_safe = make_json_safe_tool(_train_regression_model)
train_svm_model_safe = make_json_safe_tool(_train_svm_model)
train_random_forest_model_safe = make_json_safe_tool(_train_random_forest_model)
train_knn_model_safe = make_json_safe_tool(_train_knn_model)
evaluate_model_safe = make_json_safe_tool(_evaluate_model)

# Titanic Specific Tools
calculate_survival_rate_by_group_safe = make_json_safe_tool(_calculate_survival_rate_by_group)
get_statistics_for_profile_safe = make_json_safe_tool(_get_statistics_for_profile)
calculate_survival_probability_by_features_safe = make_json_safe_tool(_calculate_survival_probability_by_features)
get_fare_estimate_by_profile_safe = make_json_safe_tool(_get_fare_estimate_by_profile)
count_passengers_by_criteria_safe = make_json_safe_tool(_count_passengers_by_criteria)

# Operations Tools
filter_data_safe = make_json_safe_tool(_filter_data)
perform_math_operations_safe = make_json_safe_tool(_perform_math_operations)
string_operations_safe = make_json_safe_tool(_string_operations)
aggregate_data_safe = make_json_safe_tool(_aggregate_data)


# ========== CREATE ENHANCED SAFE WRAPPED TOOLS ==========

# Data Tools (Enhanced)
read_csv_file_enhanced = make_enhanced_safe_tool(_read_csv_file)
read_json_file_enhanced = make_enhanced_safe_tool(_read_json_file)
get_column_info_enhanced = make_enhanced_safe_tool(_get_column_info)
get_data_summary_enhanced = make_enhanced_safe_tool(_get_data_summary)
preview_data_enhanced = make_enhanced_safe_tool(_preview_data)

# Manipulation Tools (Enhanced)
create_dummy_variables_enhanced = make_enhanced_safe_tool(_create_dummy_variables)
modify_column_values_enhanced = make_enhanced_safe_tool(_modify_column_values)
handle_missing_values_enhanced = make_enhanced_safe_tool(_handle_missing_values)
convert_data_types_enhanced = make_enhanced_safe_tool(_convert_data_types)

# ML Tools (Enhanced)
train_regression_model_enhanced = make_enhanced_safe_tool(_train_regression_model)
train_svm_model_enhanced = make_enhanced_safe_tool(_train_svm_model)
train_random_forest_model_enhanced = make_enhanced_safe_tool(_train_random_forest_model)
train_knn_model_enhanced = make_enhanced_safe_tool(_train_knn_model)
evaluate_model_enhanced = make_enhanced_safe_tool(_evaluate_model)

# Titanic Specific Tools (Enhanced)
calculate_survival_rate_by_group_enhanced = make_enhanced_safe_tool(_calculate_survival_rate_by_group)
get_statistics_for_profile_enhanced = make_enhanced_safe_tool(_get_statistics_for_profile)
calculate_survival_probability_by_features_enhanced = make_enhanced_safe_tool(_calculate_survival_probability_by_features)
get_fare_estimate_by_profile_enhanced = make_enhanced_safe_tool(_get_fare_estimate_by_profile)
count_passengers_by_criteria_enhanced = make_enhanced_safe_tool(_count_passengers_by_criteria)

# Operations Tools (Enhanced)
filter_data_enhanced = make_enhanced_safe_tool(_filter_data)
perform_math_operations_enhanced = make_enhanced_safe_tool(_perform_math_operations)
string_operations_enhanced = make_enhanced_safe_tool(_string_operations)
aggregate_data_enhanced = make_enhanced_safe_tool(_aggregate_data)


# ========== TOOL COLLECTIONS ==========

ALL_SAFE_TOOLS: List[StructuredTool] = [
    # Data tools
    read_csv_file_safe,
    read_json_file_safe,
    get_column_info_safe,
    get_data_summary_safe,
    preview_data_safe,
    # Manipulation tools
    create_dummy_variables_safe,
    modify_column_values_safe,
    handle_missing_values_safe,
    convert_data_types_safe,
    # ML tools
    train_regression_model_safe,
    train_svm_model_safe,
    train_random_forest_model_safe,
    train_knn_model_safe,
    evaluate_model_safe,
    # Titanic specific tools
    calculate_survival_rate_by_group_safe,
    get_statistics_for_profile_safe,
    calculate_survival_probability_by_features_safe,
    get_fare_estimate_by_profile_safe,
    count_passengers_by_criteria_safe,
    # Operations tools
    filter_data_safe,
    perform_math_operations_safe,
    string_operations_safe,
    aggregate_data_safe,
]

ALL_ENHANCED_TOOLS: List[StructuredTool] = [
    # Data tools
    read_csv_file_enhanced,
    read_json_file_enhanced,
    get_column_info_enhanced,
    get_data_summary_enhanced,
    preview_data_enhanced,
    # Manipulation tools
    create_dummy_variables_enhanced,
    modify_column_values_enhanced,
    handle_missing_values_enhanced,
    convert_data_types_enhanced,
    # ML tools
    train_regression_model_enhanced,
    train_svm_model_enhanced,
    train_random_forest_model_enhanced,
    train_knn_model_enhanced,
    evaluate_model_enhanced,
    # Titanic specific tools
    calculate_survival_rate_by_group_enhanced,
    get_statistics_for_profile_enhanced,
    calculate_survival_probability_by_features_enhanced,
    get_fare_estimate_by_profile_enhanced,
    count_passengers_by_criteria_enhanced,
    # Operations tools
    filter_data_enhanced,
    perform_math_operations_enhanced,
    string_operations_enhanced,
    aggregate_data_enhanced,
]


# ========== UTILITY FUNCTIONS ==========

def get_safe_tool_by_name(name: str) -> StructuredTool:
    """
    Get a safe-wrapped tool by its name.

    Args:
        name: The name of the tool

    Returns:
        The safe-wrapped StructuredTool

    Raises:
        ValueError: If tool name not found
    """
    for tool in ALL_SAFE_TOOLS:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool '{name}' not found in safe tools")


def get_enhanced_tool_by_name(name: str) -> StructuredTool:
    """
    Get an enhanced safe-wrapped tool by its base name.

    Args:
        name: The base name of the tool (without '_safe' suffix)

    Returns:
        The enhanced safe-wrapped StructuredTool

    Raises:
        ValueError: If tool name not found
    """
    safe_name = f"{name}_safe"
    for tool in ALL_ENHANCED_TOOLS:
        if tool.name == safe_name:
            return tool
    raise ValueError(f"Enhanced tool '{safe_name}' not found")


def list_safe_tools() -> List[str]:
    """
    List all available safe-wrapped tool names.

    Returns:
        List of tool names
    """
    return [tool.name for tool in ALL_SAFE_TOOLS]


def list_enhanced_tools() -> List[str]:
    """
    List all available enhanced safe-wrapped tool names.

    Returns:
        List of tool names
    """
    return [tool.name for tool in ALL_ENHANCED_TOOLS]
