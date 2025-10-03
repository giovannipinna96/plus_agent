#!/usr/bin/env python3
"""
Simple Universal Agent for Plus-Agent System

A streamlined single agent with access to all tools, optimized for reliability.
"""

import os
import sys
import time
from typing import Dict, Any, List
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import all tools from different categories
from tools.data_tools import (
    read_csv_file, read_json_file, get_column_info,
    get_data_summary, preview_data
)
from tools.manipulation_tools import (
    create_dummy_variables, modify_column_values,
    handle_missing_values, convert_data_types
)
from tools.operations_tools import (
    filter_data, perform_math_operations,
    string_operations, aggregate_data
)
from tools.ml_tools import (
    train_regression_model, train_svm_model,
    train_random_forest_model, train_knn_model, evaluate_model
)

from core.config import config


class SimpleUniversalAgent:
    """
    A simplified single agent that directly uses tools without LangChain agent framework.
    More reliable for demonstration purposes.
    """

    def __init__(self):
        # Organize all tools by category
        self.data_tools = [
            read_csv_file, read_json_file, get_column_info,
            get_data_summary, preview_data
        ]

        self.manipulation_tools = [
            create_dummy_variables, modify_column_values,
            handle_missing_values, convert_data_types
        ]

        self.operations_tools = [
            filter_data, perform_math_operations,
            string_operations, aggregate_data
        ]

        self.ml_tools = [
            train_regression_model, train_svm_model,
            train_random_forest_model, train_knn_model, evaluate_model
        ]

        # Combine all tools
        self.all_tools = (self.data_tools + self.manipulation_tools +
                         self.operations_tools + self.ml_tools)

    def analyze_basic(self, dataset_path: str) -> Dict[str, Any]:
        """Perform basic dataset analysis using multiple tools."""
        results = {
            "dataset_path": dataset_path,
            "analysis_steps": [],
            "tools_used": [],
            "execution_time": 0
        }

        start_time = time.time()

        try:
            print("🔍 Step 1: Reading CSV file...")
            csv_result = read_csv_file.invoke({"file_path": dataset_path})
            results["analysis_steps"].append({"step": "read_csv", "result": csv_result})
            results["tools_used"].append("read_csv_file")
            print(f"✅ {csv_result}")

            print("\n🔍 Step 2: Getting data summary...")
            summary_result = get_data_summary.invoke({"file_path": dataset_path})
            results["analysis_steps"].append({"step": "data_summary", "result": summary_result})
            results["tools_used"].append("get_data_summary")
            print(f"✅ {summary_result[:200]}...")

            print("\n🔍 Step 3: Getting column information...")
            column_result = get_column_info.invoke({"file_path": dataset_path})
            results["analysis_steps"].append({"step": "column_info", "result": column_result})
            results["tools_used"].append("get_column_info")
            print(f"✅ {column_result[:200]}...")

            print("\n🔍 Step 4: Previewing data...")
            preview_result = preview_data.invoke({"file_path": dataset_path, "num_rows": 5})
            results["analysis_steps"].append({"step": "preview_data", "result": preview_result})
            results["tools_used"].append("preview_data")
            print(f"✅ {preview_result[:200]}...")

            results["status"] = "success"

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"❌ Error: {e}")

        results["execution_time"] = round(time.time() - start_time, 2)
        return results

    def analyze_preprocessing(self, dataset_path: str) -> Dict[str, Any]:
        """Perform data preprocessing analysis."""
        results = {
            "dataset_path": dataset_path,
            "analysis_steps": [],
            "tools_used": [],
            "execution_time": 0
        }

        start_time = time.time()

        try:
            # First understand the data
            print("🔍 Step 1: Understanding dataset structure...")
            csv_result = read_csv_file.invoke({"file_path": dataset_path})
            results["analysis_steps"].append({"step": "read_csv", "result": csv_result})
            results["tools_used"].append("read_csv_file")

            print("🔍 Step 2: Checking for missing values...")
            column_result = get_column_info.invoke({"file_path": dataset_path})
            results["analysis_steps"].append({"step": "column_info", "result": column_result})
            results["tools_used"].append("get_column_info")

            # Handle missing values in Age column (common in Titanic dataset)
            print("🔧 Step 3: Handling missing values in Age column...")
            if "age" in dataset_path.lower() or "titanic" in dataset_path.lower():
                missing_result = handle_missing_values.invoke({
                    "file_path": dataset_path,
                    "column_name": "age",
                    "method": "mean"
                })
                results["analysis_steps"].append({"step": "handle_missing", "result": missing_result})
                results["tools_used"].append("handle_missing_values")
                print(f"✅ {missing_result}")

            # Create dummy variables for categorical columns
            print("🔧 Step 4: Creating dummy variables for sex column...")
            if "titanic" in dataset_path.lower():
                dummy_result = create_dummy_variables.invoke({
                    "file_path": dataset_path,
                    "column_name": "sex",
                    "prefix": "sex"
                })
                results["analysis_steps"].append({"step": "create_dummies", "result": dummy_result})
                results["tools_used"].append("create_dummy_variables")
                print(f"✅ {dummy_result}")

            results["status"] = "success"

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"❌ Error: {e}")

        results["execution_time"] = round(time.time() - start_time, 2)
        return results

    def analyze_ml_workflow(self, dataset_path: str) -> Dict[str, Any]:
        """Perform complete ML workflow."""
        results = {
            "dataset_path": dataset_path,
            "analysis_steps": [],
            "tools_used": [],
            "execution_time": 0
        }

        start_time = time.time()

        try:
            # Data exploration
            print("🔍 Step 1: Data exploration...")
            csv_result = read_csv_file.invoke({"file_path": dataset_path})
            summary_result = get_data_summary.invoke({"file_path": dataset_path})
            results["analysis_steps"].extend([
                {"step": "read_csv", "result": csv_result},
                {"step": "data_summary", "result": summary_result}
            ])
            results["tools_used"].extend(["read_csv_file", "get_data_summary"])

            # Train a Random Forest model (common for Titanic dataset)
            print("🤖 Step 2: Training Random Forest model...")
            if "titanic" in dataset_path.lower():
                # Use common Titanic features
                features = "pclass,age,sibsp,parch,fare"
                target = "survived"

                rf_result = train_random_forest_model.invoke({
                    "file_path": dataset_path,
                    "target_column": target,
                    "feature_columns": features,
                    "task_type": "classification"
                })
                results["analysis_steps"].append({"step": "train_rf", "result": rf_result})
                results["tools_used"].append("train_random_forest_model")
                print(f"✅ {rf_result[:300]}...")

            # Train an SVM model for comparison
            print("🤖 Step 3: Training SVM model for comparison...")
            if "titanic" in dataset_path.lower():
                svm_result = train_svm_model.invoke({
                    "file_path": dataset_path,
                    "target_column": target,
                    "feature_columns": features,
                    "task_type": "classification"
                })
                results["analysis_steps"].append({"step": "train_svm", "result": svm_result})
                results["tools_used"].append("train_svm_model")
                print(f"✅ {svm_result[:300]}...")

            results["status"] = "success"

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            print(f"❌ Error: {e}")

        results["execution_time"] = round(time.time() - start_time, 2)
        return results

    def get_tools_summary(self) -> Dict[str, Any]:
        """Get summary of all available tools."""
        return {
            "total_tools": len(self.all_tools),
            "data_tools": len(self.data_tools),
            "manipulation_tools": len(self.manipulation_tools),
            "operations_tools": len(self.operations_tools),
            "ml_tools": len(self.ml_tools),
            "tool_names": [tool.name for tool in self.all_tools]
        }


def main():
    """Main function to test the Simple Universal Agent."""
    print("🌟 Simple Universal Agent for Data Science")
    print("=" * 50)

    # Initialize agent
    agent = SimpleUniversalAgent()

    # Show available tools
    tools_summary = agent.get_tools_summary()
    print(f"📊 Agent initialized with {tools_summary['total_tools']} tools")
    print(f"  • Data tools: {tools_summary['data_tools']}")
    print(f"  • Manipulation tools: {tools_summary['manipulation_tools']}")
    print(f"  • Operations tools: {tools_summary['operations_tools']}")
    print(f"  • ML tools: {tools_summary['ml_tools']}")

    print("\n" + "=" * 50)

    # Test with default dataset
    dataset_path = config.default_dataset_path

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run: uv run python data/download_titanic.py")
        return

    # Test basic analysis
    print("🧪 Testing Basic Analysis...")
    basic_result = agent.analyze_basic(dataset_path)

    print(f"\n📊 Basic Analysis Results:")
    print(f"Status: {basic_result['status']}")
    print(f"Execution time: {basic_result['execution_time']} seconds")
    print(f"Tools used: {basic_result['tools_used']}")

    if basic_result['status'] == 'success':
        print("✅ Basic analysis completed successfully!")
    else:
        print(f"❌ Error in basic analysis: {basic_result.get('error')}")


if __name__ == "__main__":
    main()