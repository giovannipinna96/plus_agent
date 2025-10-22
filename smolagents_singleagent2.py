#!/usr/bin/env python3
"""
Multi-Agent Data Analysis System using smolagents

A comprehensive multi-agent system for data analysis powered by smolagents and Hugging Face models.
This system orchestrates specialized AI agents to perform end-to-end data science workflows.

Architecture:
- Manager Agent: Orchestrates the overall workflow
- Data Reader Agent: Analyzes datasets and provides structure information
- Data Manipulation Agent: Handles data preprocessing and transformations
- Data Operations Agent: Performs mathematical operations and aggregations
- ML Prediction Agent: Trains and evaluates machine learning models

Author: Multi-Agent System
Framework: smolagents (Hugging Face)
"""

import os
import json
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
from smolagents import CodeAgent, InferenceClientModel, tool, TransformersModel, QuantizedTransformersModelHF
from smolagents import Model, ChatMessage, MessageRole, TokenUsage
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

# Import Titanic questions for interactive menu
from titanic_questions import TITANIC_QUESTIONS
import json

# Load environment variables
load_dotenv()

# Load detailed Titanic questions
with open('titanic_questions_detailed.json', 'r', encoding='utf-8') as f:
    TITANIC_QUESTIONS_DETAILED = json.load(f)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "model_id": os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct"),  # Agents use Llama-3.1-8B
    # "model_id": os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),  # Agents use Llama-3.1-8B
    #"model_id": os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct"),
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




# class QuantizedTransformersModel(Model):
#     def __init__(self, model_id: str, quant_config: BitsAndBytesConfig, **kwargs):
#         super().__init__(**kwargs)
#         self.model_id = model_id
#         self.quant_config = quant_config
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             quantization_config=quant_config,
#             device_map="auto"
#         )

#     def generate(self, messages, **kwargs):
#         prompt = "\n".join(m["content"] for m in messages)
#         inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=kwargs.get("max_new_tokens", 512),
#             temperature=kwargs.get("temperature", 0.7),
#             do_sample=True
#         )
#         decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return decoded

#     def __call__(self, *args, **kwargs):
#         return self.generate(*args, **kwargs)

# class QuantizedTransformersModelHF(Model):
#     def __init__(
#         self,
#         model_id: str | None = None,
#         device_map: str | None = None,
#         torch_dtype: str | None = None,
#         trust_remote_code: bool = False,
#         model_kwargs: dict[str, Any] | None = None,
#         max_new_tokens: int = 4096,
#         max_tokens: int | None = None,
#         quant_config: BitsAndBytesConfig | None = None,
#         **kwargs,
#     ):
#         try:
#             import torch
#             from transformers import (
#                 AutoModelForCausalLM,
#                 AutoModelForImageTextToText,
#                 AutoProcessor,
#                 AutoTokenizer,
#                 TextIteratorStreamer,
#             )
#         except ModuleNotFoundError:
#             raise ModuleNotFoundError(
#                 "Please install 'transformers' extra to use 'TransformersModel': `pip install 'smolagents[transformers]'`"
#             )

#         if not model_id:
#             warnings.warn(
#                 "The 'model_id' parameter will be required in version 2.0.0. "
#                 "Please update your code to pass this parameter to avoid future errors. "
#                 "For now, it defaults to 'HuggingFaceTB/SmolLM2-1.7B-Instruct'.",
#                 FutureWarning,
#             )
#             model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

#         max_new_tokens = max_tokens if max_tokens is not None else max_new_tokens

#         if device_map is None:
#             device_map = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info(f"Using device: {device_map}")
#         self._is_vlm = False
#         self.model_kwargs = model_kwargs or {}
#         try:
#             self.model = AutoModelForImageTextToText.from_pretrained(
#                 model_id,
#                 device_map=device_map,
#                 torch_dtype=torch_dtype,
#                 trust_remote_code=trust_remote_code,
#                 **self.model_kwargs,
#             )
#             self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
#             self._is_vlm = True
#             self.streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore

#         except ValueError as e:
#             if "Unrecognized configuration class" in str(e):
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     model_id,
#                     device_map=device_map,
#                     torch_dtype=torch_dtype,
#                     trust_remote_code=trust_remote_code,
#                     quantization_config=quant_config,
#                     **self.model_kwargs,
#                 )
#                 self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
#                 self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore
#             else:
#                 raise e
#         except Exception as e:
#             raise ValueError(f"Failed to load tokenizer and model for {model_id=}: {e}") from e
#         super().__init__(
#             flatten_messages_as_text=not self._is_vlm, model_id=model_id, max_new_tokens=max_new_tokens, **kwargs
#         )

#     def make_stopping_criteria(self, stop_sequences: list[str], tokenizer) -> "StoppingCriteriaList":
#         from transformers import StoppingCriteria, StoppingCriteriaList

#         class StopOnStrings(StoppingCriteria):
#             def __init__(self, stop_strings: list[str], tokenizer):
#                 self.stop_strings = stop_strings
#                 self.tokenizer = tokenizer
#                 self.stream = ""

#             def reset(self):
#                 self.stream = ""

#             def __call__(self, input_ids, scores, **kwargs):
#                 generated = self.tokenizer.decode(input_ids[0][-1], skip_special_tokens=True)
#                 self.stream += generated
#                 if any([self.stream.endswith(stop_string) for stop_string in self.stop_strings]):
#                     return True
#                 return False

#         return StoppingCriteriaList([StopOnStrings(stop_sequences, tokenizer)])

#     def _prepare_completion_args(
#         self,
#         messages: list[ChatMessage | dict],
#         stop_sequences: list[str] | None = None,
#         tools_to_call_from: list[Tool] | None = None,
#         **kwargs,
#     ) -> dict[str, Any]:
#         completion_kwargs = self._prepare_completion_kwargs(
#             messages=messages,
#             stop_sequences=stop_sequences,
#             tools_to_call_from=tools_to_call_from,
#             tool_choice=None,
#             **kwargs,
#         )

#         messages = completion_kwargs.pop("messages")
#         stop_sequences = completion_kwargs.pop("stop", None)
#         tools = completion_kwargs.pop("tools", None)

#         max_new_tokens = (
#             kwargs.get("max_new_tokens")
#             or kwargs.get("max_tokens")
#             or self.kwargs.get("max_new_tokens")
#             or self.kwargs.get("max_tokens")
#             or 1024
#         )
#         prompt_tensor = (self.processor if hasattr(self, "processor") else self.tokenizer).apply_chat_template(
#             messages,
#             tools=tools,
#             return_tensors="pt",
#             add_generation_prompt=True,
#             tokenize=True,
#             return_dict=True,
#         )
#         prompt_tensor = prompt_tensor.to(self.model.device)  # type: ignore
#         if hasattr(prompt_tensor, "input_ids"):
#             prompt_tensor = prompt_tensor["input_ids"]

#         model_tokenizer = self.processor.tokenizer if hasattr(self, "processor") else self.tokenizer
#         stopping_criteria = (
#             self.make_stopping_criteria(stop_sequences, tokenizer=model_tokenizer) if stop_sequences else None
#         )
#         completion_kwargs["max_new_tokens"] = max_new_tokens
#         return dict(
#             inputs=prompt_tensor,
#             use_cache=True,
#             stopping_criteria=stopping_criteria,
#             **completion_kwargs,
#         )

#     def generate(
#         self,
#         messages: list[ChatMessage | dict],
#         stop_sequences: list[str] | None = None,
#         response_format: dict[str, str] | None = None,
#         tools_to_call_from: list[Tool] | None = None,
#         **kwargs,
#     ) -> ChatMessage:
#         if response_format is not None:
#             raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
#         generation_kwargs = self._prepare_completion_args(
#             messages=messages,
#             stop_sequences=stop_sequences,
#             tools_to_call_from=tools_to_call_from,
#             **kwargs,
#         )
#         count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore
#         out = self.model.generate(
#             **generation_kwargs,
#         )
#         generated_tokens = out[0, count_prompt_tokens:]
#         if hasattr(self, "processor"):
#             output_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
#         else:
#             output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

#         if stop_sequences is not None:
#             output_text = remove_content_after_stop_sequences(output_text, stop_sequences)
#         return ChatMessage(
#             role=MessageRole.ASSISTANT,
#             content=output_text,
#             raw={
#                 "out": output_text,
#                 "completion_kwargs": {key: value for key, value in generation_kwargs.items() if key != "inputs"},
#             },
#             token_usage=TokenUsage(
#                 input_tokens=count_prompt_tokens,
#                 output_tokens=len(generated_tokens),
#             ),
#         )

#     def generate_stream(
#         self,
#         messages: list[ChatMessage | dict],
#         stop_sequences: list[str] | None = None,
#         response_format: dict[str, str] | None = None,
#         tools_to_call_from: list[Tool] | None = None,
#         **kwargs,
#     ) -> Generator[ChatMessageStreamDelta]:
#         if response_format is not None:
#             raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
#         generation_kwargs = self._prepare_completion_args(
#             messages=messages,
#             stop_sequences=stop_sequences,
#             response_format=response_format,
#             tools_to_call_from=tools_to_call_from,
#             **kwargs,
#         )

#         # Get prompt token count once
#         count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore

#         # Start generation in a separate thread
#         thread = Thread(target=self.model.generate, kwargs={"streamer": self.streamer, **generation_kwargs})
#         thread.start()

#         # Process streaming output
#         is_first_token = True
#         count_generated_tokens = 0
#         for new_text in self.streamer:
#             count_generated_tokens += 1
#             # Only include input tokens in the first yielded token
#             input_tokens = count_prompt_tokens if is_first_token else 0
#             is_first_token = False
#             yield ChatMessageStreamDelta(
#                 content=new_text,
#                 tool_calls=None,
#                 token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=1),
#             )
#             count_prompt_tokens = 0
#         thread.join()

#         # Update final output token count
#         self._last_output_token_count = count_generated_tokens


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
# DATA READING TOOLS
# ============================================================================

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


# ============================================================================
# DATA MANIPULATION TOOLS
# ============================================================================

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
        operation: Type of operation (multiply, add, subtract, divide, replace, normalize, standardize)
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


# ============================================================================
# DATA OPERATIONS TOOLS
# ============================================================================

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
def aggregate_data(file_path: str, group_by_columns: str, agg_column: str, agg_function: str) -> str:
    """
    Aggregate data by grouping columns.

    Args:
        file_path: Path to the data file
        group_by_columns: Comma-separated list of columns to group by
        agg_column: Column to aggregate
        agg_function: Aggregation function (mean, sum, count, min, max, std, median)

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

        return f"Aggregated data by {group_cols} using {agg_function} on '{agg_column}'. Result shape: {result_df.shape}. Result:\n{result_df.to_string()}\nSaved to: {output_path}"

    except Exception as e:
        return f"Error aggregating data: {str(e)}"


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


# ============================================================================
# MACHINE LEARNING TOOLS
# ============================================================================

@tool
def train_regression_model(file_path: str, target_column: str, feature_columns: str, model_type: str = "linear") -> str:
    """
    Train a regression model.

    Args:
        file_path: Path to the data file
        target_column: Name of the target variable column
        feature_columns: Comma-separated list of feature columns
        model_type: Type of regression (linear, random_forest)

    Returns:
        String describing the model training results
    """
    try:
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
        return f"Error training regression model: {str(e)}"


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
#             metrics,
#             "cv_score_mean": round(cv_scores.mean(), 4),
#             "cv_score_std": round(cv_scores.std(), 4),
#             "model_saved": model_path
#         }

#         return f"SVM model trained: {results}"

#     except Exception as e:
#         return f"Error training SVM model: {str(e)}"


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
#             metrics,
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
#             metrics,
#             "cv_score_mean": round(cv_scores.mean(), 4),
#             "cv_score_std": round(cv_scores.std(), 4),
#             "model_saved": model_path
#         }

#         return f"KNN model trained: {results}"

#     except Exception as e:
#         return f"Error training KNN model: {str(e)}"


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
    
# ============================================================================
# DATA READING TOOLS - Simplified and Specific
# ============================================================================

@tool
def load_dataset(file_path: str) -> str:
    """
    Load a dataset and return basic shape information.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        String with dataset shape (rows, columns)
    """
    try:
        df = pd.read_csv(file_path)
        return f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns"
    except Exception as e:
        return f"Error loading dataset: {str(e)}"


@tool
def get_column_names(file_path: str) -> str:
    """
    Get list of all column names in the dataset.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Comma-separated list of column names
    """
    try:
        df = pd.read_csv(file_path)
        return f"Columns: {', '.join(df.columns.tolist())}"
    except Exception as e:
        return f"Error getting columns: {str(e)}"


@tool
def get_data_types(file_path: str) -> str:
    """
    Get data types of all columns.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        String with column names and their data types
    """
    try:
        df = pd.read_csv(file_path)
        dtypes = df.dtypes.to_dict()
        result = []
        for col, dtype in dtypes.items():
            result.append(f"{col}: {dtype}")
        return "Data types:\n" + "\n".join(result)
    except Exception as e:
        return f"Error getting data types: {str(e)}"


@tool
def get_null_counts(file_path: str) -> str:
    """
    Count missing values in each column.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        String with missing value counts per column
    """
    try:
        df = pd.read_csv(file_path)
        null_counts = df.isnull().sum()
        result = []
        for col, count in null_counts.items():
            if count > 0:
                pct = (count / len(df)) * 100
                result.append(f"{col}: {count} ({pct:.1f}%)")
        
        if result:
            return "Missing values:\n" + "\n".join(result)
        else:
            return "No missing values found"
    except Exception as e:
        return f"Error counting nulls: {str(e)}"


@tool
def get_unique_values(file_path: str, column_name: str) -> str:
    """
    Get unique values in a specific column.
    
    Args:
        file_path: Path to the CSV file
        column_name: Name of the column
        
    Returns:
        String with unique values and their count
    """
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found"
        
        unique_vals = df[column_name].dropna().unique()
        n_unique = len(unique_vals)
        
        if n_unique <= 20:
            return f"Column '{column_name}' has {n_unique} unique values: {', '.join(map(str, unique_vals))}"
        else:
            sample = unique_vals[:10]
            return f"Column '{column_name}' has {n_unique} unique values. First 10: {', '.join(map(str, sample))}"
    except Exception as e:
        return f"Error getting unique values: {str(e)}"


@tool
def get_numeric_summary(file_path: str, column_name: str) -> str:
    """
    Get statistical summary for a numeric column.
    
    Args:
        file_path: Path to the CSV file
        column_name: Name of the numeric column
        
    Returns:
        String with mean, median, std, min, max, quartiles
    """
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found"
        
        col = df[column_name].dropna()
        
        stats = {
            "count": len(col),
            "mean": col.mean(),
            "median": col.median(),
            "std": col.std(),
            "min": col.min(),
            "25%": col.quantile(0.25),
            "50%": col.quantile(0.50),
            "75%": col.quantile(0.75),
            "max": col.max()
        }
        
        result = f"Statistics for '{column_name}':\n"
        for key, val in stats.items():
            result += f"  {key}: {val:.2f}\n"
        
        return result
    except Exception as e:
        return f"Error getting summary: {str(e)}"


@tool
def get_first_rows(file_path: str, n_rows: int = 5) -> str:
    """
    Get first N rows of the dataset.
    
    Args:
        file_path: Path to the CSV file
        n_rows: Number of rows to return
        
    Returns:
        String representation of first N rows
    """
    try:
        df = pd.read_csv(file_path)
        preview = df.head(n_rows).to_string()
        return f"First {n_rows} rows:\n{preview}"
    except Exception as e:
        return f"Error getting rows: {str(e)}"


# ============================================================================
# DATA MANIPULATION TOOLS - More Specific
# ============================================================================

@tool
def drop_column(file_path: str, column_name: str) -> str:
    """
    Drop a specific column from the dataset.
    
    Args:
        file_path: Path to the CSV file
        column_name: Name of column to drop
        
    Returns:
        String confirming column drop and new file path
    """
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found"
        
        df = df.drop(columns=[column_name])
        output_path = file_path.replace('.csv', '_dropped.csv')
        df.to_csv(output_path, index=False)
        
        return f"Dropped column '{column_name}'. New shape: {df.shape}. Saved to: {output_path}"
    except Exception as e:
        return f"Error dropping column: {str(e)}"


@tool
def drop_null_rows(file_path: str, column_name: Optional[str] = None) -> str:
    """
    Drop rows with missing values.
    
    Args:
        file_path: Path to the CSV file
        column_name: Specific column to check for nulls (optional, if None drops any row with null)
        
    Returns:
        String confirming rows dropped
    """
    try:
        df = pd.read_csv(file_path)
        original_len = len(df)
        
        if column_name:
            if column_name not in df.columns:
                return f"Column '{column_name}' not found"
            df = df.dropna(subset=[column_name])
        else:
            df = df.dropna()
        
        dropped = original_len - len(df)
        output_path = file_path.replace('.csv', '_no_nulls.csv')
        df.to_csv(output_path, index=False)
        
        return f"Dropped {dropped} rows with nulls. New shape: {df.shape}. Saved to: {output_path}"
    except Exception as e:
        return f"Error dropping null rows: {str(e)}"


@tool
def fill_numeric_nulls(file_path: str, column_name: str, method: str = "mean") -> str:
    """
    Fill missing values in a numeric column.
    
    Args:
        file_path: Path to the CSV file
        column_name: Name of the numeric column
        method: Fill method - 'mean', 'median', 'zero', or numeric value
        
    Returns:
        String confirming fill operation
    """
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found"
        
        null_count = df[column_name].isnull().sum()
        if null_count == 0:
            return f"No missing values in '{column_name}'"
        
        if method == "mean":
            fill_value = df[column_name].mean()
        elif method == "median":
            fill_value = df[column_name].median()
        elif method == "zero":
            fill_value = 0
        else:
            try:
                fill_value = float(method)
            except:
                return f"Invalid method: {method}"
        
        df[column_name] = df[column_name].fillna(fill_value)
        
        output_path = file_path.replace('.csv', '_filled.csv')
        df.to_csv(output_path, index=False)
        
        return f"Filled {null_count} nulls in '{column_name}' with {method} ({fill_value:.2f}). Saved to: {output_path}"
    except Exception as e:
        return f"Error filling nulls: {str(e)}"


@tool
def fill_categorical_nulls(file_path: str, column_name: str, method: str = "mode") -> str:
    """
    Fill missing values in a categorical column.
    
    Args:
        file_path: Path to the CSV file
        column_name: Name of the categorical column
        method: Fill method - 'mode', 'unknown', or specific value
        
    Returns:
        String confirming fill operation
    """
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found"
        
        null_count = df[column_name].isnull().sum()
        if null_count == 0:
            return f"No missing values in '{column_name}'"
        
        if method == "mode":
            fill_value = df[column_name].mode()[0] if len(df[column_name].mode()) > 0 else "unknown"
        elif method == "unknown":
            fill_value = "unknown"
        else:
            fill_value = method
        
        df[column_name] = df[column_name].fillna(fill_value)
        
        output_path = file_path.replace('.csv', '_filled.csv')
        df.to_csv(output_path, index=False)
        
        return f"Filled {null_count} nulls in '{column_name}' with '{fill_value}'. Saved to: {output_path}"
    except Exception as e:
        return f"Error filling categorical nulls: {str(e)}"


@tool
def encode_categorical(file_path: str, column_name: str, encoding_type: str = "onehot") -> str:
    """
    Encode a categorical column.
    
    Args:
        file_path: Path to the CSV file
        column_name: Name of categorical column
        encoding_type: 'onehot' or 'label'
        
    Returns:
        String confirming encoding operation
    """
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found"
        
        if encoding_type == "onehot":
            dummies = pd.get_dummies(df[column_name], prefix=column_name, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[column_name])
            msg = f"One-hot encoded '{column_name}' into {len(dummies.columns)} columns"
        elif encoding_type == "label":
            le = LabelEncoder()
            df[f"{column_name}_encoded"] = le.fit_transform(df[column_name].fillna('missing'))
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            msg = f"Label encoded '{column_name}'. Mapping: {mapping}"
        else:
            return f"Unknown encoding type: {encoding_type}"
        
        output_path = file_path.replace('.csv', '_encoded.csv')
        df.to_csv(output_path, index=False)
        
        return f"{msg}. Saved to: {output_path}"
    except Exception as e:
        return f"Error encoding: {str(e)}"


@tool
def create_new_feature(file_path: str, new_column: str, column1: str, operation: str, column2_or_value: Union[str, float]) -> str:
    """
    Create a new feature from existing columns.
    
    Args:
        file_path: Path to the CSV file
        new_column: Name for the new feature
        column1: First column name
        operation: Operation to perform (+, -, *, /, >, <, ==)
        column2_or_value: Second column name or numeric value
        
    Returns:
        String confirming feature creation
    """
    try:
        df = pd.read_csv(file_path)
        
        if column1 not in df.columns:
            return f"Column '{column1}' not found"
        
        # Check if column2_or_value is a column name or a value
        if isinstance(column2_or_value, str) and column2_or_value in df.columns:
            operand2 = df[column2_or_value]
        else:
            try:
                operand2 = float(column2_or_value)
            except:
                operand2 = column2_or_value
        
        operand1 = df[column1]
        
        if operation == '+':
            df[new_column] = operand1 + operand2
        elif operation == '-':
            df[new_column] = operand1 - operand2
        elif operation == '*':
            df[new_column] = operand1 * operand2
        elif operation == '/':
            df[new_column] = operand1 / operand2
        elif operation == '>':
            df[new_column] = (operand1 > operand2).astype(int)
        elif operation == '<':
            df[new_column] = (operand1 < operand2).astype(int)
        elif operation == '==':
            df[new_column] = (operand1 == operand2).astype(int)
        else:
            return f"Unknown operation: {operation}"
        
        output_path = file_path.replace('.csv', '_new_feature.csv')
        df.to_csv(output_path, index=False)
        
        return f"Created new feature '{new_column}' using {column1} {operation} {column2_or_value}. Saved to: {output_path}"
    except Exception as e:
        return f"Error creating feature: {str(e)}"


@tool
def normalize_column(file_path: str, column_name: str, method: str = "minmax") -> str:
    """
    Normalize a numeric column.
    
    Args:
        file_path: Path to the CSV file
        column_name: Name of column to normalize
        method: 'minmax' (0-1) or 'zscore' (standardization)
        
    Returns:
        String confirming normalization
    """
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found"
        
        col = df[column_name].values.reshape(-1, 1)
        
        if method == "minmax":
            scaler = MinMaxScaler()
            df[f"{column_name}_normalized"] = scaler.fit_transform(col)
            msg = f"Min-max normalized '{column_name}' to range [0, 1]"
        elif method == "zscore":
            scaler = StandardScaler()
            df[f"{column_name}_standardized"] = scaler.fit_transform(col)
            msg = f"Z-score standardized '{column_name}' (mean=0, std=1)"
        else:
            return f"Unknown method: {method}"
        
        output_path = file_path.replace('.csv', '_normalized.csv')
        df.to_csv(output_path, index=False)
        
        return f"{msg}. Saved to: {output_path}"
    except Exception as e:
        return f"Error normalizing: {str(e)}"


# ============================================================================
# DATA FILTERING AND SELECTION TOOLS
# ============================================================================

@tool
def filter_rows_numeric(file_path: str, column_name: str, operator: str, value: float) -> str:
    """
    Filter rows based on numeric condition.
    
    Args:
        file_path: Path to the CSV file
        column_name: Column to filter on
        operator: Comparison operator (>, <, >=, <=, ==, !=)
        value: Numeric value to compare
        
    Returns:
        String with filtering results
    """
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found"
        
        original_len = len(df)
        
        if operator == '>':
            df_filtered = df[df[column_name] > value]
        elif operator == '<':
            df_filtered = df[df[column_name] < value]
        elif operator == '>=':
            df_filtered = df[df[column_name] >= value]
        elif operator == '<=':
            df_filtered = df[df[column_name] <= value]
        elif operator == '==':
            df_filtered = df[df[column_name] == value]
        elif operator == '!=':
            df_filtered = df[df[column_name] != value]
        else:
            return f"Unknown operator: {operator}"
        
        output_path = file_path.replace('.csv', '_filtered.csv')
        df_filtered.to_csv(output_path, index=False)
        
        return f"Filtered {original_len} -> {len(df_filtered)} rows where {column_name} {operator} {value}. Saved to: {output_path}"
    except Exception as e:
        return f"Error filtering: {str(e)}"


@tool
def filter_rows_categorical(file_path: str, column_name: str, values: str, include: bool = True) -> str:
    """
    Filter rows based on categorical values.
    
    Args:
        file_path: Path to the CSV file
        column_name: Column to filter on
        values: Comma-separated list of values
        include: If True, keep rows with these values. If False, exclude them
        
    Returns:
        String with filtering results
    """
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found"
        
        value_list = [v.strip() for v in values.split(',')]
        original_len = len(df)
        
        if include:
            df_filtered = df[df[column_name].isin(value_list)]
            action = "included"
        else:
            df_filtered = df[~df[column_name].isin(value_list)]
            action = "excluded"
        
        output_path = file_path.replace('.csv', '_filtered.csv')
        df_filtered.to_csv(output_path, index=False)
        
        return f"Filtered {original_len} -> {len(df_filtered)} rows. {action}: {value_list}. Saved to: {output_path}"
    except Exception as e:
        return f"Error filtering: {str(e)}"


@tool
def select_columns(file_path: str, columns: str) -> str:
    """
    Select specific columns from the dataset.
    
    Args:
        file_path: Path to the CSV file
        columns: Comma-separated list of column names to keep
        
    Returns:
        String confirming column selection
    """
    try:
        df = pd.read_csv(file_path)
        col_list = [c.strip() for c in columns.split(',')]
        
        missing = [c for c in col_list if c not in df.columns]
        if missing:
            return f"Columns not found: {missing}"
        
        df_selected = df[col_list]
        
        output_path = file_path.replace('.csv', '_selected.csv')
        df_selected.to_csv(output_path, index=False)
        
        return f"Selected {len(col_list)} columns. New shape: {df_selected.shape}. Saved to: {output_path}"
    except Exception as e:
        return f"Error selecting columns: {str(e)}"


# ============================================================================
# STATISTICAL ANALYSIS TOOLS
# ============================================================================

@tool
def calculate_correlation(file_path: str, column1: str, column2: str, method: str = "pearson") -> str:
    """
    Calculate correlation between two numeric columns.
    
    Args:
        file_path: Path to the CSV file
        column1: First column name
        column2: Second column name
        method: 'pearson' or 'spearman'
        
    Returns:
        String with correlation coefficient and p-value
    """
    try:
        df = pd.read_csv(file_path)
        
        for col in [column1, column2]:
            if col not in df.columns:
                return f"Column '{col}' not found"
        
        # Remove rows with missing values
        data = df[[column1, column2]].dropna()
        
        if method == "pearson":
            corr, p_value = pearsonr(data[column1], data[column2])
        elif method == "spearman":
            corr, p_value = spearmanr(data[column1], data[column2])
        else:
            return f"Unknown method: {method}"
        
        interpretation = ""
        abs_corr = abs(corr)
        if abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        
        direction = "positive" if corr > 0 else "negative"
        
        return f"{method.capitalize()} correlation between '{column1}' and '{column2}': {corr:.4f} (p-value: {p_value:.4f}). This is a {strength} {direction} correlation."
    except Exception as e:
        return f"Error calculating correlation: {str(e)}"


@tool
def perform_ttest(file_path: str, column_name: str, group_column: str) -> str:
    """
    Perform t-test between two groups.
    
    Args:
        file_path: Path to the CSV file
        column_name: Numeric column to test
        group_column: Column with two groups
        
    Returns:
        String with t-statistic and p-value
    """
    try:
        df = pd.read_csv(file_path)
        
        if column_name not in df.columns or group_column not in df.columns:
            return "Column not found"
        
        groups = df[group_column].unique()
        if len(groups) != 2:
            return f"T-test requires exactly 2 groups. Found {len(groups)}: {groups}"
        
        group1 = df[df[group_column] == groups[0]][column_name].dropna()
        group2 = df[df[group_column] == groups[1]][column_name].dropna()
        
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        mean1, mean2 = group1.mean(), group2.mean()
        
        sig = "statistically significant" if p_value < 0.05 else "not statistically significant"
        
        return f"T-test for '{column_name}' between groups:\n  {groups[0]}: mean={mean1:.2f}, n={len(group1)}\n  {groups[1]}: mean={mean2:.2f}, n={len(group2)}\n  t-statistic: {t_stat:.4f}\n  p-value: {p_value:.4f}\n  Result: {sig} (α=0.05)"
    except Exception as e:
        return f"Error performing t-test: {str(e)}"


@tool
def chi_square_test(file_path: str, column1: str, column2: str) -> str:
    """
    Perform chi-square test between two categorical variables.
    
    Args:
        file_path: Path to the CSV file
        column1: First categorical column
        column2: Second categorical column
        
    Returns:
        String with chi-square statistic and p-value
    """
    try:
        df = pd.read_csv(file_path)
        
        if column1 not in df.columns or column2 not in df.columns:
            return "Column not found"
        
        # Create contingency table
        contingency = pd.crosstab(df[column1], df[column2])
        
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        sig = "dependent" if p_value < 0.05 else "independent"
        
        return f"Chi-square test between '{column1}' and '{column2}':\n  Chi-square statistic: {chi2:.4f}\n  p-value: {p_value:.4f}\n  Degrees of freedom: {dof}\n  Result: Variables are {sig} (α=0.05)"
    except Exception as e:
        return f"Error performing chi-square test: {str(e)}"


@tool
def calculate_group_statistics(file_path: str, value_column: str, group_column: str) -> str:
    """
    Calculate statistics for each group.
    
    Args:
        file_path: Path to the CSV file
        value_column: Numeric column to analyze
        group_column: Column to group by
        
    Returns:
        String with statistics for each group
    """
    try:
        df = pd.read_csv(file_path)
        
        if value_column not in df.columns or group_column not in df.columns:
            return "Column not found"
        
        grouped = df.groupby(group_column)[value_column].agg([
            'count', 'mean', 'std', 'min', 'max'
        ])
        
        result = f"Group statistics for '{value_column}' by '{group_column}':\n"
        result += grouped.to_string()
        
        return result
    except Exception as e:
        return f"Error calculating group statistics: {str(e)}"


# ============================================================================
# DATA VISUALIZATION TOOLS
# ============================================================================

@tool
def create_histogram(file_path: str, column_name: str, bins: int = 20) -> str:
    """
    Create histogram for a numeric column.
    
    Args:
        file_path: Path to the CSV file
        column_name: Column to plot
        bins: Number of bins
        
    Returns:
        String confirming plot creation
    """
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found"
        
        plt.figure(figsize=(10, 6))
        plt.hist(df[column_name].dropna(), bins=bins, edgecolor='black', alpha=0.7)
        plt.title(f'Histogram of {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        output_path = file_path.replace('.csv', f'_histogram_{column_name}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Calculate statistics
        data = df[column_name].dropna()
        stats_info = f"Mean: {data.mean():.2f}, Median: {data.median():.2f}, Std: {data.std():.2f}"
        
        return f"Histogram created for '{column_name}'. {stats_info}. Saved to: {output_path}"
    except Exception as e:
        return f"Error creating histogram: {str(e)}"


@tool
def create_scatter_plot(file_path: str, x_column: str, y_column: str, color_column: Optional[str] = None) -> str:
    """
    Create scatter plot between two numeric columns.
    
    Args:
        file_path: Path to the CSV file
        x_column: Column for x-axis
        y_column: Column for y-axis
        color_column: Optional column for color coding
        
    Returns:
        String confirming plot creation
    """
    try:
        df = pd.read_csv(file_path)
        
        for col in [x_column, y_column]:
            if col not in df.columns:
                return f"Column '{col}' not found"
        
        plt.figure(figsize=(10, 6))
        
        if color_column and color_column in df.columns:
            for category in df[color_column].unique():
                mask = df[color_column] == category
                plt.scatter(df[mask][x_column], df[mask][y_column], label=category, alpha=0.6)
            plt.legend()
        else:
            plt.scatter(df[x_column], df[y_column], alpha=0.6)
        
        plt.title(f'Scatter Plot: {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid(True, alpha=0.3)
        
        output_path = file_path.replace('.csv', f'_scatter_{x_column}_{y_column}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Calculate correlation
        corr = df[[x_column, y_column]].corr().iloc[0, 1]
        
        return f"Scatter plot created. Correlation: {corr:.3f}. Saved to: {output_path}"
    except Exception as e:
        return f"Error creating scatter plot: {str(e)}"


@tool
def create_bar_chart(file_path: str, column_name: str, top_n: int = 10) -> str:
    """
    Create bar chart for categorical column.
    
    Args:
        file_path: Path to the CSV file
        column_name: Categorical column to plot
        top_n: Number of top categories to show
        
    Returns:
        String confirming plot creation
    """
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found"
        
        value_counts = df[column_name].value_counts().head(top_n)
        
        plt.figure(figsize=(10, 6))
        value_counts.plot(kind='bar', edgecolor='black', alpha=0.7)
        plt.title(f'Bar Chart: Top {top_n} values in {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        output_path = file_path.replace('.csv', f'_bar_{column_name}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return f"Bar chart created for '{column_name}'. Top value: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences). Saved to: {output_path}"
    except Exception as e:
        return f"Error creating bar chart: {str(e)}"


@tool
def create_correlation_heatmap(file_path: str, columns: Optional[str] = None) -> str:
    """
    Create correlation heatmap for numeric columns.
    
    Args:
        file_path: Path to the CSV file
        columns: Optional comma-separated list of columns (if None, uses all numeric)
        
    Returns:
        String confirming plot creation
    """
    try:
        df = pd.read_csv(file_path)
        
        if columns:
            col_list = [c.strip() for c in columns.split(',')]
            df_numeric = df[col_list]
        else:
            df_numeric = df.select_dtypes(include=[np.number])
        
        if df_numeric.shape[1] < 2:
            return "Need at least 2 numeric columns for correlation heatmap"
        
        plt.figure(figsize=(12, 8))
        corr_matrix = df_numeric.corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap')
        
        output_path = file_path.replace('.csv', '_correlation_heatmap.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Find strongest correlations
        upper_tri = np.triu(np.ones_like(corr_matrix), k=1)
        upper_tri_corr = corr_matrix.where(upper_tri.astype(bool))
        strongest = upper_tri_corr.abs().unstack().nlargest(3)
        
        return f"Correlation heatmap created for {corr_matrix.shape[0]} variables. Saved to: {output_path}"
    except Exception as e:
        return f"Error creating heatmap: {str(e)}"


# ============================================================================
# ENHANCED MACHINE LEARNING TOOLS
# ============================================================================

@tool
def train_random_forest_model(file_path: str, target_column: str, feature_columns: str, task_type: str = "classification") -> str:
    """
    Train a Random Forest model.

    Args:
        file_path: Path to the data file
        target_column: Name of the target variable column
        feature_columns: Comma-separated list of feature columns
        task_type: Type of task (classification, regression)

    Returns:
        String describing the model training results including feature importance
    """
    try:
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
        return f"Error training Random Forest model: {str(e)}"


@tool
def train_knn_model(file_path: str, target_column: str, feature_columns: str, task_type: str = "classification", n_neighbors: int = 5) -> str:
    """
    Train a K-Nearest Neighbors model.

    Args:
        file_path: Path to the data file
        target_column: Name of the target variable column
        feature_columns: Comma-separated list of feature columns
        task_type: Type of task (classification, regression)
        n_neighbors: Number of neighbors to use

    Returns:
        String describing the model training results
    """
    try:
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
        return f"Error training KNN model: {str(e)}"


@tool
def train_svm_model(file_path: str, target_column: str, feature_columns: str, task_type: str = "classification") -> str:
    """
    Train a Support Vector Machine model.

    Args:
        file_path: Path to the data file
        target_column: Name of the target variable column
        feature_columns: Comma-separated list of feature columns
        task_type: Type of task (classification, regression)

    Returns:
        String describing the model training results
    """
    try:
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
        return f"Error training SVM model: {str(e)}"


@tool
def evaluate_model(model_path: str, test_data_path: str, target_column: str, feature_columns: str) -> str:
    """
    Evaluate a trained model on new test data.

    Args:
        model_path: Path to the saved model file
        test_data_path: Path to the test data file
        target_column: Name of the target variable column
        feature_columns: Comma-separated list of feature columns

    Returns:
        String describing the model evaluation results
    """
    try:
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
        return f"Error evaluating model: {str(e)}"



@tool
def train_logistic_regression(file_path: str, target_column: str, feature_columns: str) -> str:
    """
    Train logistic regression for binary classification.
    
    Args:
        file_path: Path to the CSV file
        target_column: Target variable column
        feature_columns: Comma-separated list of feature columns
        
    Returns:
        String with model performance metrics
    """
    try:
        df = pd.read_csv(file_path)
        features = [col.strip() for col in feature_columns.split(',')]
        
        # Prepare data
        X = df[features]
        y = df[target_column]
        
        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('missing'))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Encode target if categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # Check if binary classification for AUC
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y_test, y_proba)
            auc_str = f", AUC: {auc:.3f}"
        else:
            auc_str = ""
        
        # Feature importance (coefficients)
        importance = dict(zip(features, model.coef_[0]))
        top_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        # Save model
        model_path = file_path.replace('.csv', '_logistic_model.joblib')
        joblib.dump(model, model_path)
        
        return f"Logistic Regression trained:\n  Accuracy: {acc:.3f}\n  Precision: {prec:.3f}\n  Recall: {rec:.3f}\n  F1: {f1:.3f}{auc_str}\n  Top features: {top_features}\n  Model saved: {model_path}"
    except Exception as e:
        return f"Error training logistic regression: {str(e)}"


@tool
def train_decision_tree(file_path: str, target_column: str, feature_columns: str, max_depth: int = 5) -> str:
    """
    Train decision tree classifier.
    
    Args:
        file_path: Path to the CSV file
        target_column: Target variable column
        feature_columns: Comma-separated list of feature columns
        max_depth: Maximum depth of the tree
        
    Returns:
        String with model performance and feature importance
    """
    try:
        df = pd.read_csv(file_path)
        features = [col.strip() for col in feature_columns.split(',')]
        
        # Prepare data
        X = df[features]
        y = df[target_column]
        
        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('missing'))
        
        X = X.fillna(X.mean())
        
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        
        # Feature importance
        importance = dict(zip(features, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Save model
        model_path = file_path.replace('.csv', '_decision_tree.joblib')
        joblib.dump(model, model_path)
        
        return f"Decision Tree trained:\n  Accuracy: {acc:.3f}\n  Max depth: {max_depth}\n  Top features: {top_features}\n  Model saved: {model_path}"
    except Exception as e:
        return f"Error training decision tree: {str(e)}"


@tool
def train_gradient_boosting(file_path: str, target_column: str, feature_columns: str, n_estimators: int = 100) -> str:
    """
    Train gradient boosting classifier.
    
    Args:
        file_path: Path to the CSV file
        target_column: Target variable column
        feature_columns: Comma-separated list of feature columns
        n_estimators: Number of boosting stages
        
    Returns:
        String with model performance
    """
    try:
        df = pd.read_csv(file_path)
        features = [col.strip() for col in feature_columns.split(',')]
        
        # Prepare data
        X = df[features]
        y = df[target_column]
        
        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('missing'))
        
        X = X.fillna(X.mean())
        
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        
        # Feature importance
        importance = dict(zip(features, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Save model
        model_path = file_path.replace('.csv', '_gradient_boosting.joblib')
        joblib.dump(model, model_path)
        
        return f"Gradient Boosting trained:\n  Accuracy: {acc:.3f}\n  N estimators: {n_estimators}\n  Top features: {top_features}\n  Model saved: {model_path}"
    except Exception as e:
        return f"Error training gradient boosting: {str(e)}"


@tool
def make_prediction(model_path: str, data_path: str, feature_columns: str) -> str:
    """
    Make predictions using a trained model.
    
    Args:
        model_path: Path to saved model
        data_path: Path to data for prediction
        feature_columns: Comma-separated list of feature columns
        
    Returns:
        String with predictions
    """
    try:
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        features = [col.strip() for col in feature_columns.split(',')]
        
        X = df[features]
        
        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('missing'))
        
        X = X.fillna(X.mean())
        
        # Make predictions
        predictions = model.predict(X)
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        
        # If probabilistic model, add probabilities
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
            if probas.shape[1] == 2:
                df['probability'] = probas[:, 1]
        
        output_path = data_path.replace('.csv', '_predictions.csv')
        df.to_csv(output_path, index=False)
        
        # Summary
        unique, counts = np.unique(predictions, return_counts=True)
        pred_summary = dict(zip(unique, counts))
        
        return f"Predictions made for {len(predictions)} samples:\n  Distribution: {pred_summary}\n  Results saved to: {output_path}"
    except Exception as e:
        return f"Error making predictions: {str(e)}"


@tool
def perform_cross_validation(file_path: str, target_column: str, feature_columns: str, model_type: str = "random_forest", cv_folds: int = 5) -> str:
    """
    Perform cross-validation for model evaluation.
    
    Args:
        file_path: Path to the CSV file
        target_column: Target variable column
        feature_columns: Comma-separated list of feature columns
        model_type: Type of model (random_forest, logistic, svm)
        cv_folds: Number of cross-validation folds
        
    Returns:
        String with cross-validation scores
    """
    try:
        df = pd.read_csv(file_path)
        features = [col.strip() for col in feature_columns.split(',')]
        
        # Prepare data
        X = df[features]
        y = df[target_column]
        
        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('missing'))
        
        X = X.fillna(X.mean())
        
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        
        # Select model
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "logistic":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "svm":
            model = SVC(random_state=42)
        else:
            return f"Unknown model type: {model_type}"
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        
        return f"Cross-validation results ({cv_folds} folds):\n  Model: {model_type}\n  Scores: {cv_scores.round(3)}\n  Mean: {cv_scores.mean():.3f}\n  Std: {cv_scores.std():.3f}\n  Min: {cv_scores.min():.3f}\n  Max: {cv_scores.max():.3f}"
    except Exception as e:
        return f"Error in cross-validation: {str(e)}"


@tool
def feature_selection(file_path: str, target_column: str, feature_columns: str, n_features: int = 5) -> str:
    """
    Select top features based on importance.
    
    Args:
        file_path: Path to the CSV file
        target_column: Target variable column
        feature_columns: Comma-separated list of feature columns
        n_features: Number of top features to select
        
    Returns:
        String with selected features and their scores
    """
    try:
        df = pd.read_csv(file_path)
        features = [col.strip() for col in feature_columns.split(',')]
        
        # Prepare data
        X = df[features]
        y = df[target_column]
        
        # Handle categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('missing'))
        
        X = X.fillna(X.mean())
        
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(n_features, len(features)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = [f for f, s in zip(features, selected_mask) if s]
        
        # Get scores
        feature_scores = dict(zip(features, selector.scores_))
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
        
        # Save selected features dataset
        df_selected = pd.concat([df[selected_features], df[target_column]], axis=1)
        output_path = file_path.replace('.csv', '_selected_features.csv')
        df_selected.to_csv(output_path, index=False)
        
        return f"Feature selection complete:\n  Selected {len(selected_features)} features: {selected_features}\n  Top scores: {top_features}\n  Saved to: {output_path}"
    except Exception as e:
        return f"Error in feature selection: {str(e)}"


# ============================================================================
# ANSWER GENERATION TOOLS
# ============================================================================

@tool
def answer_survival_question(file_path: str, passenger_info: str) -> str:
    """
    Answer survival questions about Titanic passengers.
    
    Args:
        file_path: Path to the CSV file
        passenger_info: Description of passenger (e.g., "male, age 30, first class")
        
    Returns:
        String with survival prediction and analysis
    """
    try:
        df = pd.read_csv(file_path)
        
        # Parse passenger info
        info_lower = passenger_info.lower()
        
        # Extract features
        conditions = []
        
        if 'male' in info_lower or 'man' in info_lower:
            conditions.append(df['Sex'] == 'male')
        elif 'female' in info_lower or 'woman' in info_lower:
            conditions.append(df['Sex'] == 'female')
        
        if 'first class' in info_lower or 'class 1' in info_lower:
            conditions.append(df['Pclass'] == 1)
        elif 'second class' in info_lower or 'class 2' in info_lower:
            conditions.append(df['Pclass'] == 2)
        elif 'third class' in info_lower or 'class 3' in info_lower:
            conditions.append(df['Pclass'] == 3)
        
        # Age extraction
        import re
        age_match = re.search(r'age (\d+)', info_lower)
        if age_match:
            age = int(age_match.group(1))
            conditions.append((df['Age'] >= age - 5) & (df['Age'] <= age + 5))
        
        # Filter data
        if conditions:
            mask = conditions[0]
            for cond in conditions[1:]:
                mask = mask & cond
            filtered = df[mask]
        else:
            filtered = df
        
        if len(filtered) == 0:
            return "No passengers found matching the description"
        
        # Calculate survival rate
        survival_rate = filtered['Survived'].mean()
        total_matching = len(filtered)
        survived = filtered['Survived'].sum()
        
        # Additional context
        overall_survival = df['Survived'].mean()
        
        return f"Survival analysis for '{passenger_info}':\n  Matching passengers: {total_matching}\n  Survived: {survived}\n  Survival rate: {survival_rate:.1%}\n  Overall survival rate: {overall_survival:.1%}\n  Relative survival: {survival_rate/overall_survival:.2f}x the average"
    except Exception as e:
        return f"Error analyzing survival: {str(e)}"


@tool
def predict_single_passenger_survival(
    file_path: str,
    passenger_age: float,
    passenger_sex: str,
    passenger_class: int,
    siblings_spouses: int = 0,
    parents_children: int = 0,
    fare: float = None
) -> str:
    """
    Predicts survival probability for a single Titanic passenger based on their characteristics.

    This specialized tool trains a Random Forest machine learning model on the complete Titanic
    dataset and predicts survival probability for a hypothetical passenger. It provides detailed
    breakdown with confidence levels, feature importance, and contextual comparison to historical rates.

    Args:
        file_path: Path to the Titanic CSV file
        passenger_age: Age in years (0.1 to 120)
        passenger_sex: Gender ('male' or 'female', case-insensitive)
        passenger_class: Ticket class (1=First, 2=Second, 3=Third)
        siblings_spouses: Number of siblings/spouses aboard (default: 0)
        parents_children: Number of parents/children aboard (default: 0)
        fare: Ticket fare in pounds (optional, uses class median if None)

    Returns:
        Comprehensive prediction report with probability, context, and interpretation
    """
    try:
        # Input validation
        passenger_sex = passenger_sex.lower().strip()
        if passenger_sex not in ['male', 'female']:
            return f"✗ Error: Invalid sex '{passenger_sex}'. Must be 'male' or 'female'"

        if passenger_class not in [1, 2, 3]:
            return f"✗ Error: Invalid class {passenger_class}. Must be 1, 2, or 3"

        if not (0.1 <= passenger_age <= 120):
            return f"✗ Error: Invalid age {passenger_age}. Must be between 0.1 and 120"

        if siblings_spouses < 0 or parents_children < 0:
            return "✗ Error: Family counts cannot be negative"

        # Load dataset
        df = pd.read_csv(file_path)

        # Handle column names (lowercase/uppercase)
        col_map = {col.lower(): col for col in df.columns}

        # Check required columns
        required = ['survived', 'pclass', 'sex', 'age']
        if not all(req in col_map for req in required):
            return f"✗ Error: Missing required columns"

        # Prepare features
        feature_names_orig = [col_map['pclass'], col_map['sex'], col_map['age']]
        if 'sibsp' in col_map:
            feature_names_orig.append(col_map['sibsp'])
        if 'parch' in col_map:
            feature_names_orig.append(col_map['parch'])
        if 'fare' in col_map:
            feature_names_orig.append(col_map['fare'])

        # Create training dataset
        df_train = df[[col_map['survived']] + feature_names_orig].copy()
        df_train.columns = ['survived', 'pclass', 'sex', 'age'] + \
                          (['sibsp'] if 'sibsp' in col_map else []) + \
                          (['parch'] if 'parch' in col_map else []) + \
                          (['fare'] if 'fare' in col_map else [])

        # Handle missing values
        for col in df_train.columns:
            if col in ['age', 'fare']:
                df_train[col] = df_train[col].fillna(df_train[col].median())
            elif col == 'sex':
                df_train[col] = df_train[col].fillna('male')

        # Encode sex
        df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})

        # Separate features and target
        feature_cols = [c for c in df_train.columns if c != 'survived']
        X = df_train[feature_cols]
        y = df_train['survived']

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)

        # Calculate metrics
        train_acc = model.score(X, y)
        cv_scores = cross_val_score(model, X, y, cv=5)

        # Prepare passenger data
        if fare is None and 'fare' in feature_cols:
            class_fares = df_train[df_train['pclass'] == passenger_class]['fare']
            fare = class_fares.median() if len(class_fares) > 0 else 0
            fare_estimated = True
        else:
            fare_estimated = False

        passenger_data = {
            'pclass': passenger_class,
            'sex': 1 if passenger_sex == 'male' else 0,
            'age': passenger_age
        }

        if 'sibsp' in feature_cols:
            passenger_data['sibsp'] = siblings_spouses
        if 'parch' in feature_cols:
            passenger_data['parch'] = parents_children
        if 'fare' in feature_cols:
            passenger_data['fare'] = fare if fare is not None else 0

        passenger_df = pd.DataFrame([passenger_data])[feature_cols]

        # Make prediction
        survival_prob = model.predict_proba(passenger_df)[0][1]
        prediction = model.predict(passenger_df)[0]

        # Get feature importance
        importances = list(zip(feature_cols, model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)

        # Calculate historical rates
        overall_survival = y.mean()
        sex_survival = df_train[df_train['sex'] == passenger_data['sex']]['survived'].mean()
        class_sex_survival = df_train[
            (df_train['pclass'] == passenger_class) &
            (df_train['sex'] == passenger_data['sex'])
        ]['survived'].mean()

        # Find similar passengers
        similar = df_train[
            (df_train['pclass'] == passenger_class) &
            (df_train['sex'] == passenger_data['sex']) &
            (df_train['age'].between(passenger_age - 5, passenger_age + 5))
        ]

        # Build result
        class_names = {
            1: "1st class (First/Upper)",
            2: "2nd class (Second/Middle)",
            3: "3rd class (Third/Working)"
        }

        result = [
            "",
            "🚢 Survival Prediction for Titanic Passenger",
            "=" * 60,
            "",
            "👤 Passenger Profile:",
            f"   • Age: {passenger_age} years old",
            f"   • Sex: {passenger_sex.capitalize()}",
            f"   • Class: {class_names[passenger_class]}",
            f"   • Family: {siblings_spouses} siblings/spouses, {parents_children} parents/children",
        ]

        if 'fare' in feature_cols:
            fare_str = f"£{fare:.2f}"
            if fare_estimated:
                fare_str += " (estimated)"
            result.append(f"   • Fare: {fare_str}")

        result.extend([
            "",
            "🎯 Prediction Results:",
            f"   • Survival Probability: {survival_prob*100:.1f}%",
            f"   • Prediction: {'✓ SURVIVED' if prediction == 1 else '✗ DID NOT SURVIVE'}",
        ])

        # Confidence
        if survival_prob > 0.75:
            conf_desc = "Very High"
            conf_emoji = "🟢"
        elif survival_prob > 0.60:
            conf_desc = "High"
            conf_emoji = "🟢"
        elif survival_prob > 0.45:
            conf_desc = "Uncertain"
            conf_emoji = "🟡"
        else:
            conf_desc = "Low"
            conf_emoji = "🔴"

        result.append(f"   • Confidence: {conf_emoji} {conf_desc}")
        result.append("")

        # Feature importance
        result.append("📊 Feature Importance:")
        feature_names_pretty = {
            'sex': 'Sex',
            'pclass': 'Class',
            'age': 'Age',
            'fare': 'Fare',
            'sibsp': 'Siblings/Spouses',
            'parch': 'Parents/Children'
        }

        for idx, (feat, imp) in enumerate(importances[:5], 1):
            result.append(f"   {idx}. {feature_names_pretty.get(feat, feat)}: {imp*100:.1f}%")

        result.append("")

        # Historical context
        result.extend([
            "📜 Historical Context:",
            f"   • Overall survival: {overall_survival*100:.1f}%",
            f"   • {passenger_sex.capitalize()} survival: {sex_survival*100:.1f}%",
            f"   • {class_names[passenger_class].split('(')[0].strip()} {passenger_sex}: {class_sex_survival*100:.1f}%",
        ])

        if class_sex_survival > 0:
            ratio = survival_prob / class_sex_survival
            if ratio > 1.2:
                result.append(f"   → Your profile: {ratio:.1f}x BETTER than average")
            elif ratio < 0.8:
                result.append(f"   → Your profile: {1/ratio:.1f}x WORSE than average")

        result.append("")

        # Similar passengers
        if len(similar) > 0:
            similar_survived = similar['survived'].sum()
            result.extend([
                f"👥 Similar Passengers (Age {max(0, passenger_age-5):.0f}-{passenger_age+5:.0f}):",
                f"   • Found: {len(similar)} passengers",
                f"   • Survived: {int(similar_survived)} ({similar_survived/len(similar)*100:.1f}%)",
                ""
            ])

        # Model performance
        result.extend([
            "🔬 Model Performance:",
            f"   • Training accuracy: {train_acc*100:.1f}%",
            f"   • Cross-validation: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%",
            "",
            "=" * 60
        ])

        return "\n".join(result)

    except FileNotFoundError:
        return f"✗ Error: File not found at '{file_path}'"
    except Exception as e:
        return f"✗ Error: {type(e).__name__}: {str(e)}"


@tool
def get_dataset_insights(file_path: str) -> str:
    """
    Generate comprehensive insights about the dataset.

    Args:
        file_path: Path to the CSV file

    Returns:
        String with key insights about the data
    """
    try:
        df = pd.read_csv(file_path)
        
        insights = []
        
        # Basic info
        insights.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns")
        
        # Missing data
        missing = df.isnull().sum()
        if missing.sum() > 0:
            worst_missing = missing.nlargest(3)
            insights.append(f"Missing data in: {worst_missing.to_dict()}")
        
        # For Titanic specific insights
        if 'Survived' in df.columns:
            survival_rate = df['Survived'].mean()
            insights.append(f"Overall survival rate: {survival_rate:.1%}")
            
            if 'Sex' in df.columns:
                female_survival = df[df['Sex'] == 'female']['Survived'].mean()
                male_survival = df[df['Sex'] == 'male']['Survived'].mean()
                insights.append(f"Female survival: {female_survival:.1%}, Male survival: {male_survival:.1%}")
            
            if 'Pclass' in df.columns:
                class_survival = df.groupby('Pclass')['Survived'].mean()
                insights.append(f"Survival by class: {class_survival.to_dict()}")
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        insights.append(f"Numeric columns: {list(numeric_cols)}")
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        insights.append(f"Categorical columns: {list(cat_cols)}")
        
        return "Dataset Insights:\n" + "\n".join(f"• {insight}" for insight in insights)
    except Exception as e:
        return f"Error generating insights: {str(e)}"




# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

def create_agents():
    """Create and return all specialized agents."""

    print("🔧 Initializing Model...")
    # Initialize the LLM model
    # model = TransformersModel(
    #     model_id=CONFIG["model_id"],
    #     temperature=CONFIG["temperature"],
    #     max_new_tokens=CONFIG["max_new_tokens"],
    #     trust_remote_code=True,
    #     token=CONFIG["huggingface_token"]
    # )
    # model = InferenceClientModel(
    #     model_id=CONFIG["model_id"],
    #     token=CONFIG["huggingface_token"]
    # )
    print(f"✅ Model initialized: {CONFIG['model_id']}")

    print("\n🤖 Creating Specialized Agents...")

    # Data Reader Agent
    print("   📊 Creating Data Reader Agent...")
    # data_reader_agent = CodeAgent(
    #     tools=[
    #         read_csv_file,
    #         read_json_file,
    #         get_column_info,
    #         get_data_summary,
    #         preview_data
    #     ],
    #     model=model,
    #     name="data_reader",
    #     description="Analyzes datasets, reads CSV/JSON files, provides column information, statistical summaries, and data previews. Use this agent for data exploration and understanding data structure.",
    #     max_steps=10,
    #     verbosity_level=1
    # )

    # Data Manipulation Agent
    print("   🔧 Creating Data Manipulation Agent...")
    # data_manipulation_agent = CodeAgent(
    #     tools=[
    #         handle_missing_values,
    #         create_dummy_variables,
    #         modify_column_values,
    #         convert_data_types
    #     ],
    #     model=model,
    #     name="data_manipulation",
    #     description="Handles data preprocessing, missing value imputation, categorical encoding, data type conversions, and column transformations. Use this agent for data cleaning and preparation.",
    #     max_steps=10,
    #     verbosity_level=1
    # )

    # Data Operations Agent
    print("   ⚡ Creating Data Operations Agent...")
    # data_operations_agent = CodeAgent(
    #     tools=[
    #         filter_data,
    #         perform_math_operations,
    #         aggregate_data,
    #         string_operations
    #     ],
    #     model=model,
    #     name="data_operations",
    #     description="Performs mathematical operations, data filtering, aggregations, grouping, and string manipulations. Use this agent for data analysis and feature engineering.",
    #     max_steps=10,
    #     verbosity_level=1
    # )

    # ML Prediction Agent
    # print("   🎯 Creating ML Prediction Agent...")
    # ml_prediction_agent = CodeAgent(
    #     tools=[
    #         train_regression_model,
    #         train_svm_model,
    #         train_random_forest_model,
    #         train_knn_model,
    #         evaluate_model
    #     ],
    #     model=model,
    #     name="ml_prediction",
    #     description="Trains and evaluates machine learning models including Linear Regression, SVM, Random Forest, and KNN. Provides model performance metrics, feature importance, and cross-validation results. Use this agent for predictive modeling and model evaluation.",
    #     max_steps=10,
    #     verbosity_level=1
    # )

    print("✅ All specialized agents created successfully!")

    # return model, data_reader_agent, data_manipulation_agent, data_operations_agent, ml_prediction_agent


# ============================================================================
# MANAGER AGENT (ORCHESTRATOR)
# ============================================================================

def create_manager_agent():
    """Create the manager agent that coordinates all specialized agents."""

    # print("\n👔 Creating Manager Agent (Orchestrator)...")

    # Wrap specialized agents in ManagedAgent
    # managed_data_reader = ManagedAgent(
    #     agent=data_reader_agent,
    #     name="data_reader",
    #     description="Analyzes datasets, reads CSV/JSON files, provides column information, statistical summaries, and data previews. Use for data exploration and understanding data structure."
    # )

    # managed_data_manipulation = ManagedAgent(
    #     agent=data_manipulation_agent,
    #     name="data_manipulation",
    #     description="Handles data preprocessing, missing value imputation, categorical encoding, data type conversions, and column transformations. Use for data cleaning and preparation."
    # )

    # managed_data_operations = ManagedAgent(
    #     agent=data_operations_agent,
    #     name="data_operations",
    #     description="Performs mathematical operations, data filtering, aggregations, grouping, and string manipulations. Use for data analysis and feature engineering."
    # )

    # managed_ml_prediction = ManagedAgent(
    #     agent=ml_prediction_agent,
    #     name="ml_prediction",
    #     description="Trains and evaluates machine learning models including Linear Regression, SVM, Random Forest, and KNN. Provides model performance metrics, feature importance, and cross-validation results. Use for predictive modeling."
    # )
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    
    # VRAM usata in MB
    used = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2

    print(f"GPU VRAM allocata: {used:.2f} MB")
    print(f"GPU VRAM riservata: {reserved:.2f} MB")
    print(f"GPU VRAM totale: {total:.2f} MB")
    print(f"GPU VRAM libera: {total - reserved:.2f} MB")
    
    # model = TransformersModel(
    #     model_id=CONFIG["model_id"],
    #     temperature=CONFIG["temperature"],
    #     max_new_tokens=CONFIG["max_new_tokens"],
    #     trust_remote_code=True,
    #     token=CONFIG["huggingface_token"],
    #     # model_kwargs={"quantization_config":bnb_config}
    #     # quantization_config=bnb_config
    #     model_kwargs={"dtype":"auto"}
    # )

    model = QuantizedTransformersModelHF(CONFIG["model_id"], quant_config=quant_config)
    
    # Create the manager agent
    manager_agent = CodeAgent(
        tools=[
            # read_csv_file,
            # read_json_file,
            # get_column_info,
            # get_data_summary,
            # preview_data,
            # handle_missing_values,
            # create_dummy_variables,
            # modify_column_values,
            # convert_data_types,
            # filter_data,
            # perform_math_operations,
            # aggregate_data,
            # string_operations,
            # train_regression_model,
            # train_svm_model,
            # train_random_forest_model,
            # train_knn_model,
            # evaluate_model
            
            load_dataset,
            get_column_names,
            get_data_types,
            get_null_counts,
            get_unique_values,
            get_numeric_summary,
            get_first_rows,
            get_dataset_insights,
            drop_column,
            drop_null_rows,
            fill_numeric_nulls,
            fill_categorical_nulls,
            encode_categorical,
            create_new_feature,
            normalize_column,
            filter_rows_numeric,
            filter_rows_categorical,
            select_columns,
            calculate_correlation,
            perform_ttest,
            chi_square_test,
            calculate_group_statistics,
            create_histogram,
            create_scatter_plot,
            create_bar_chart,
            create_correlation_heatmap,
            train_logistic_regression,
            train_decision_tree,
            train_random_forest_model,
            train_gradient_boosting,
            train_knn_model,
            train_svm_model,
            perform_cross_validation,
            feature_selection,
            make_prediction,
            evaluate_model,
            answer_survival_question,
            predict_single_passenger_survival,
            ],
        model=model,
        # managed_agents=[
        #     data_reader_agent,
        #     data_manipulation_agent,
        #     data_operations_agent,
        #     ml_prediction_agent
        # ],
        additional_authorized_imports=["pandas", "numpy", "sklearn", "joblib", "csv", "json"],
        max_steps=20,
        verbosity_level=2,
        planning_interval=5
    )
    
    # VRAM usata in MB
    used = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2

    print(f"GPU VRAM allocata: {used:.2f} MB")
    print(f"GPU VRAM riservata: {reserved:.2f} MB")
    print(f"GPU VRAM totale: {total:.2f} MB")
    print(f"GPU VRAM libera: {total - reserved:.2f} MB")

    print("✅ Manager Agent created successfully!")
    print("\n🎯 Multi-Agent System Ready!")

    return manager_agent


# ============================================================================
# ORCHESTRATION FUNCTION
# ============================================================================

def run_analysis(user_prompt: str, file_path: str = None, question_number: int = None) -> str:
    """
    Run multi-agent data analysis based on user prompt.

    Args:
        user_prompt: User's analysis request in natural language
        file_path: Path to the data file (optional, uses default if not provided)
        question_number: Question number (1-10) to use detailed context (optional)

    Returns:
        String containing the analysis results
    """
    # Use default dataset if no file path provided
    if file_path is None:
        file_path = CONFIG["default_dataset"]

    print(f"\n{'='*80}")
    print(f"🚀 Starting Multi-Agent Analysis")
    print(f"{'='*80}")
    print(f"📝 User Prompt: {user_prompt}")
    print(f"📁 Data File: {file_path}")
    if question_number:
        print(f"❓ Question Number: {question_number} (using detailed context)")
    print(f"{'='*80}\n")

    # Step 1: Analyze dataset columns
    print("📊 Analyzing dataset columns...")
    columns_info = analyze_dataset_columns(file_path)

    # Step 2: Generate column descriptions with LLM
    column_descriptions = generate_column_descriptions_with_llm(columns_info)

    # Step 3: Format column descriptions for prompt
    formatted_columns = format_column_descriptions(column_descriptions)

    # Create agents
    # model, data_reader, data_manipulation, data_operations, ml_prediction = create_agents()

    # Create manager agent
    manager = create_manager_agent()

    # Format the task for the manager with column descriptions
    task = f"""Task: {user_prompt}
Data file: {file_path}

{formatted_columns}

## IMPORTANT INSTRUCTIONS
You must perform only the actions required to complete the given task.
Use exclusively the tools that have been explicitly provided to you.
Do not invent or assume the existence of any other tools, functions, or agents.
If the task cannot be completed with the available tools, state this clearly instead of attempting to improvise.
Always choose the simplest possible plan to achieve the objective. Avoid unnecessary steps or complexity.
If the user provides additional information or instructions in a future request, you may refine or extend your plan accordingly.

## Your goal
Analyze the task and complete it using only the available tools, following a logical and minimal sequence of steps.
Provide a clear and complete final answer containing all relevant results.

"""
# IMPORTANT INSTRUCTIONS:
# - You MUST ONLY use the tools and agents that have been explicitly provided to you
# - DO NOT invent, create, or imagine new tools, functions, or agents
# - DO NOT attempt to use tools or methods that are not in the provided list
# - If you cannot complete a task with the available tools, state this clearly

# You have access to ONLY these specialized agents:
# - data_reader: For reading and exploring the dataset (tools: read_csv_file, read_json_file, get_column_info, get_data_summary, preview_data)
# - data_manipulation: For data cleaning and preprocessing (tools: handle_missing_values, create_dummy_variables, modify_column_values, convert_data_types)
# - data_operations: For mathematical operations and aggregations (tools: filter_data, perform_math_operations, aggregate_data, string_operations)
# - ml_prediction: For training and evaluating machine learning models (tools: train_regression_model, train_svm_model, train_random_forest_model, train_knn_model, evaluate_model)

# Analyze the task and delegate work ONLY to the appropriate agents listed above in the correct order.
# Use ONLY the tools available to each agent.
# Provide a comprehensive final answer with all results.


    print("🎬 Executing analysis...\n")

    try:
        # Run the manager agent
        result = manager.run(task)

        print(f"\n{'='*80}")
        print("✅ Analysis Complete!")
        print(f"{'='*80}\n")

        return result

    except Exception as e:
        error_msg = f"❌ Error during analysis: {str(e)}"
        print(f"\n{error_msg}\n")
        return error_msg


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

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
