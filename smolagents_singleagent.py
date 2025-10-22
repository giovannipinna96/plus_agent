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
from smolagents import CodeAgent, InferenceClientModel, tool, TransformersModel

# Import Titanic questions for interactive menu
from titanic_questions import TITANIC_QUESTIONS

# Import all tools from centralized file (TODO #1: Centralization)
from smolagents_tools import *

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "model_id": os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),  # Agents use Llama-3.1-8B
    #"model_id": os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct"),
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


# ==============================================================================
# ALL TOOLS IMPORTED FROM smolagents_tools.py
# (TODO #1: Centralization - Removed ~3200 lines of duplicated tool definitions)
# ==============================================================================

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
    
    model = TransformersModel(
        model_id=CONFIG["model_id"],
        temperature=CONFIG["temperature"],
        max_new_tokens=CONFIG["max_new_tokens"],
        trust_remote_code=True,
        token=CONFIG["huggingface_token"]
    )

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

    print("✅ Manager Agent created successfully!")
    print("\n🎯 Multi-Agent System Ready!")

    return manager_agent


# ============================================================================
# ORCHESTRATION FUNCTION
# ============================================================================

def run_analysis(user_prompt: str, file_path: str = None) -> str:
    """
    Run multi-agent data analysis based on user prompt.

    Args:
        user_prompt: User's analysis request in natural language
        file_path: Path to the data file (optional, uses default if not provided)

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
    print(f"{'='*80}\n")

    # TODO #2: Clear DataFrame cache for new analysis session
    print("🧹 Clearing DataFrame cache for new session...")
    df_state_manager.clear_all()

    # TODO #2: Pre-load dataset into memory for performance
    print(f"📥 Pre-loading dataset into memory: {file_path}")
    try:
        df_state_manager.load_dataframe(file_path)
        print("✓ Dataset cached in memory for fast access\n")
    except Exception as e:
        print(f"⚠️ Warning: Could not pre-load dataset: {str(e)}\n")

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

    # # OLD PROMPT (Commented out - replaced with optimized version below)
    # task = f"""Task: {user_prompt}
    # Data file: {file_path}
    #
    # {formatted_columns}
    #
    # ## IMPORTANT INSTRUCTIONS
    # You must perform only the actions required to complete the given task.
    # Use exclusively the tools that have been explicitly provided to you.
    # Do not invent or assume the existence of any other tools, functions, or agents.
    # If the task cannot be completed with the available tools, state this clearly instead of attempting to improvise.
    # Always choose the simplest possible plan to achieve the objective. Avoid unnecessary steps or complexity.
    # If the user provides additional information or instructions in a future request, you may refine or extend your plan accordingly.
    #
    # ## Your goal
    # Analyze the task and complete it using only the available tools, following a logical and minimal sequence of steps.
    # Provide a clear and complete final answer containing all relevant results.
    # """

    # NEW OPTIMIZED PROMPT - Emphasizes DataFrameStateManager for efficient memory usage
    task = f"""Task: {user_prompt}
Data file: {file_path}

{formatted_columns}

## CRITICAL PERFORMANCE INFORMATION
⚡ DATASET IS ALREADY LOADED IN MEMORY via DataFrameStateManager!
- The dataset has been pre-loaded and cached for optimal performance
- All tools automatically access the in-memory DataFrame - NO repeated file I/O operations
- When you call any tool with the file_path parameter, it retrieves data from memory cache
- First tool call loads data into memory, all subsequent calls reuse the cached version
- This ensures MAXIMUM EFFICIENCY and MINIMAL latency across all operations

## IMPORTANT INSTRUCTIONS
You must perform only the actions required to complete the given task.
Use exclusively the tools that have been explicitly provided to you.
Do not invent or assume the existence of any other tools, functions, or agents.
If the task cannot be completed with the available tools, state this clearly instead of attempting to improvise.
Always choose the simplest possible plan to achieve the objective. Avoid unnecessary steps or complexity.
The dataset is already optimized in memory - focus on analysis, not data loading.

## Your goal
Analyze the task and complete it using only the available tools, following a logical and minimal sequence of steps.
All tools share the same in-memory DataFrame via DataFrameStateManager for seamless integration.
Provide a clear and complete final answer containing all relevant results.

"""


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
