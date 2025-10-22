"""Pydantic-based tool definitions using StructuredTool."""

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional, Union

# Import original functions from tool modules
from .data_tools import (
    read_csv_file as _read_csv_file,
    read_json_file as _read_json_file,
    get_column_info as _get_column_info,
    get_data_summary as _get_data_summary,
    preview_data as _preview_data
)

from .manipulation_tools import (
    create_dummy_variables as _create_dummy_variables,
    modify_column_values as _modify_column_values,
    handle_missing_values as _handle_missing_values,
    convert_data_types as _convert_data_types
)

from .ml_tools import (
    train_regression_model as _train_regression_model,
    train_svm_model as _train_svm_model,
    train_random_forest_model as _train_random_forest_model,
    train_knn_model as _train_knn_model,
    evaluate_model as _evaluate_model
)

from .titanic_specific_tools import (
    calculate_survival_rate_by_group as _calculate_survival_rate_by_group,
    get_statistics_for_profile as _get_statistics_for_profile,
    calculate_survival_probability_by_features as _calculate_survival_probability_by_features,
    get_fare_estimate_by_profile as _get_fare_estimate_by_profile,
    count_passengers_by_criteria as _count_passengers_by_criteria
)

from .operations_tools import (
    filter_data as _filter_data,
    perform_math_operations as _perform_math_operations,
    string_operations as _string_operations,
    aggregate_data as _aggregate_data
)


# ========== DATA TOOLS INPUT SCHEMAS ==========

class ReadCsvFileInput(BaseModel):
    file_path: str = Field(description="Path to the CSV file")


class ReadJsonFileInput(BaseModel):
    file_path: str = Field(description="Path to the JSON file")


class GetColumnInfoInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    column_name: Optional[str] = Field(default=None, description="Specific column to analyze (optional)")


class GetDataSummaryInput(BaseModel):
    file_path: str = Field(description="Path to the data file")


class PreviewDataInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    num_rows: int = Field(default=5, description="Number of rows to preview")


# ========== MANIPULATION TOOLS INPUT SCHEMAS ==========

class CreateDummyVariablesInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    column_name: str = Field(description="Name of the categorical column")
    prefix: Optional[str] = Field(default=None, description="Prefix for dummy variable names")


class ModifyColumnValuesInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    column_name: str = Field(description="Name of the column to modify")
    operation: str = Field(description="Type of operation (multiply, add, subtract, divide, replace, normalize, standardize)")
    value: Optional[Union[str, float, int]] = Field(default=None, description="Value to use in the operation")


class HandleMissingValuesInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    column_name: str = Field(description="Name of the column with missing values")
    method: str = Field(description="Method to handle missing values (drop, mean, median, mode, forward_fill, backward_fill, constant)")
    fill_value: Optional[Union[str, float, int]] = Field(default=None, description="Value to use for 'constant' method")


class ConvertDataTypesInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    column_name: str = Field(description="Name of the column to convert")
    target_type: str = Field(description="Target data type (int, float, string, category, datetime)")


# ========== ML TOOLS INPUT SCHEMAS ==========

class TrainRegressionModelInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    target_column: str = Field(description="Name of the target variable column")
    feature_columns: str = Field(description="Comma-separated list of feature columns")
    model_type: str = Field(default="linear", description="Type of regression (linear, random_forest)")


class TrainSvmModelInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    target_column: str = Field(description="Name of the target variable column")
    feature_columns: str = Field(description="Comma-separated list of feature columns")
    task_type: str = Field(default="classification", description="Type of task (classification, regression)")


class TrainRandomForestModelInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    target_column: str = Field(description="Name of the target variable column")
    feature_columns: str = Field(description="Comma-separated list of feature columns")
    task_type: str = Field(default="classification", description="Type of task (classification, regression)")


class TrainKnnModelInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    target_column: str = Field(description="Name of the target variable column")
    feature_columns: str = Field(description="Comma-separated list of feature columns")
    task_type: str = Field(default="classification", description="Type of task (classification, regression)")
    n_neighbors: int = Field(default=5, description="Number of neighbors to use")


class EvaluateModelInput(BaseModel):
    model_path: str = Field(description="Path to the saved model file")
    test_data_path: str = Field(description="Path to the test data file")
    target_column: str = Field(description="Name of the target variable column")
    feature_columns: str = Field(description="Comma-separated list of feature columns")


# ========== TITANIC SPECIFIC TOOLS INPUT SCHEMAS ==========

class CalculateSurvivalRateByGroupInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    group_column: str = Field(description="Column to group by (e.g., 'pclass', 'sex', 'embarked')")


class GetStatisticsForProfileInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    filters: str = Field(description="Comma-separated filters in format 'column=value,column=value'")
    target_column: str = Field(description="Column to get statistics for")


class CalculateSurvivalProbabilityByFeaturesInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    sex: str = Field(description="Gender (male/female)")
    pclass: int = Field(description="Passenger class (1, 2, or 3)")
    age_range: str = Field(default="all", description="Age range filter (e.g., '18-30', '30-50', 'all')")


class GetFareEstimateByProfileInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    sex: str = Field(description="Gender (male/female)")
    pclass: int = Field(description="Passenger class (1, 2, or 3)")
    age: float = Field(description="Age of the passenger")


class CountPassengersByCriteriaInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    criteria: str = Field(description="Criteria in format 'column1=value1,column2=value2' or complex filters")


# ========== OPERATIONS TOOLS INPUT SCHEMAS ==========

class FilterDataInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    column_name: str = Field(description="Name of the column to filter on")
    condition: str = Field(description="Condition type (equals, not_equals, greater_than, less_than, greater_equal, less_equal, contains)")
    value: Union[str, float, int] = Field(description="Value to filter by")


class PerformMathOperationsInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    operation: str = Field(description="Type of operation (add, subtract, multiply, divide, power, square, sqrt, log, abs)")
    column1: str = Field(description="First column name")
    column2: Optional[str] = Field(default=None, description="Second column name (for two-column operations)")
    value: Optional[float] = Field(default=None, description="Numeric value (for column-value operations)")


class StringOperationsInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    column_name: str = Field(description="Name of the text column")
    operation: str = Field(description="Type of operation (upper, lower, title, length, contains_count, split, replace, strip)")
    parameter: Optional[str] = Field(default=None, description="Additional parameter for some operations")


class AggregateDataInput(BaseModel):
    file_path: str = Field(description="Path to the data file")
    group_by_columns: str = Field(description="Comma-separated list of columns to group by")
    agg_column: str = Field(description="Column to aggregate")
    agg_function: str = Field(description="Aggregation function (mean, sum, count, min, max, std, median)")


# ========== DATA TOOLS ==========

read_csv_file = StructuredTool.from_function(
    func=_read_csv_file,
    args_schema=ReadCsvFileInput,
    description="Read a CSV file and return basic information about the dataset"
)

read_json_file = StructuredTool.from_function(
    func=_read_json_file,
    args_schema=ReadJsonFileInput,
    description="Read a JSON file and return basic information about the data"
)

get_column_info = StructuredTool.from_function(
    func=_get_column_info,
    args_schema=GetColumnInfoInput,
    description="Get detailed information about columns in the dataset"
)

get_data_summary = StructuredTool.from_function(
    func=_get_data_summary,
    args_schema=GetDataSummaryInput,
    description="Get statistical summary of the dataset"
)

preview_data = StructuredTool.from_function(
    func=_preview_data,
    args_schema=PreviewDataInput,
    description="Preview the first few rows of the dataset"
)


# ========== MANIPULATION TOOLS ==========

create_dummy_variables = StructuredTool.from_function(
    func=_create_dummy_variables,
    args_schema=CreateDummyVariablesInput,
    description="Create dummy variables for a categorical column"
)

modify_column_values = StructuredTool.from_function(
    func=_modify_column_values,
    args_schema=ModifyColumnValuesInput,
    description="Modify values in a column using various operations"
)

handle_missing_values = StructuredTool.from_function(
    func=_handle_missing_values,
    args_schema=HandleMissingValuesInput,
    description="Handle missing values in a column using various methods"
)

convert_data_types = StructuredTool.from_function(
    func=_convert_data_types,
    args_schema=ConvertDataTypesInput,
    description="Convert data type of a column"
)


# ========== ML TOOLS ==========

train_regression_model = StructuredTool.from_function(
    func=_train_regression_model,
    args_schema=TrainRegressionModelInput,
    description="Train a regression model"
)

train_svm_model = StructuredTool.from_function(
    func=_train_svm_model,
    args_schema=TrainSvmModelInput,
    description="Train a Support Vector Machine model"
)

train_random_forest_model = StructuredTool.from_function(
    func=_train_random_forest_model,
    args_schema=TrainRandomForestModelInput,
    description="Train a Random Forest model"
)

train_knn_model = StructuredTool.from_function(
    func=_train_knn_model,
    args_schema=TrainKnnModelInput,
    description="Train a K-Nearest Neighbors model"
)

evaluate_model = StructuredTool.from_function(
    func=_evaluate_model,
    args_schema=EvaluateModelInput,
    description="Evaluate a trained model on new test data"
)


# ========== TITANIC SPECIFIC TOOLS ==========

calculate_survival_rate_by_group = StructuredTool.from_function(
    func=_calculate_survival_rate_by_group,
    args_schema=CalculateSurvivalRateByGroupInput,
    description="Calculate survival rates by a specific grouping column"
)

get_statistics_for_profile = StructuredTool.from_function(
    func=_get_statistics_for_profile,
    args_schema=GetStatisticsForProfileInput,
    description="Get statistics for a specific passenger profile"
)

calculate_survival_probability_by_features = StructuredTool.from_function(
    func=_calculate_survival_probability_by_features,
    args_schema=CalculateSurvivalProbabilityByFeaturesInput,
    description="Calculate survival probability for specific passenger characteristics"
)

get_fare_estimate_by_profile = StructuredTool.from_function(
    func=_get_fare_estimate_by_profile,
    args_schema=GetFareEstimateByProfileInput,
    description="Estimate fare price based on passenger profile"
)

count_passengers_by_criteria = StructuredTool.from_function(
    func=_count_passengers_by_criteria,
    args_schema=CountPassengersByCriteriaInput,
    description="Count passengers matching specific criteria"
)


# ========== OPERATIONS TOOLS ==========

filter_data = StructuredTool.from_function(
    func=_filter_data,
    args_schema=FilterDataInput,
    description="Filter data based on a condition"
)

perform_math_operations = StructuredTool.from_function(
    func=_perform_math_operations,
    args_schema=PerformMathOperationsInput,
    description="Perform mathematical operations on columns"
)

string_operations = StructuredTool.from_function(
    func=_string_operations,
    args_schema=StringOperationsInput,
    description="Perform string operations on a text column"
)

aggregate_data = StructuredTool.from_function(
    func=_aggregate_data,
    args_schema=AggregateDataInput,
    description="Aggregate data by grouping columns"
)


# ========== ALL TOOLS LIST ==========

ALL_TOOLS = [
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
    # Operations tools
    filter_data,
    perform_math_operations,
    string_operations,
    aggregate_data,
]
