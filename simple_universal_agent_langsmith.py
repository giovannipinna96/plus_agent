#!/usr/bin/env python3
"""
Simple Universal Agent with LangSmith Monitoring

A streamlined single agent with access to all tools, optimized for reliability,
with LangSmith tracing support for comparison with multi-agent system.
"""

import os
import sys
import time
from typing import Dict, Any, List
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import LangSmith for tracing
from langsmith import Client
import uuid

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


class SimpleUniversalAgentWithLangSmith:
    """
    A simplified single agent with LangSmith monitoring.
    """

    def __init__(self):
        # Initialize LangSmith client
        self.langsmith_client = None
        self.setup_langsmith()

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

    def setup_langsmith(self):
        """Setup LangSmith tracing if configured."""
        try:
            api_key = os.getenv("LANGSMITH_API_KEY")
            project = os.getenv("LANGSMITH_PROJECT", "universal-agent-test")
            tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"

            if api_key and tracing:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
                os.environ["LANGCHAIN_API_KEY"] = api_key
                os.environ["LANGCHAIN_PROJECT"] = project

                self.langsmith_client = Client(api_key=api_key)
                print(f"🔍 LangSmith tracing enabled for project: {project}")
            else:
                print("🔍 LangSmith tracing disabled")
        except Exception as e:
            print(f"⚠️ LangSmith setup failed: {e}")

    def analyze_basic(self, dataset_path: str) -> Dict[str, Any]:
        """Perform basic dataset analysis using multiple tools."""
        results = {
            "dataset_path": dataset_path,
            "analysis_steps": [],
            "tools_used": [],
            "execution_time": 0,
            "workflow_id": str(uuid.uuid4())
        }

        start_time = time.time()

        try:
            with self._trace_step("read_csv_file", {"file_path": dataset_path}):
                print("🔍 Step 1: Reading CSV file...")
                csv_result = read_csv_file.invoke({"file_path": dataset_path})
                results["analysis_steps"].append({"step": "read_csv", "result": csv_result})
                results["tools_used"].append("read_csv_file")
                print(f"✅ {csv_result}")

            with self._trace_step("get_data_summary", {"file_path": dataset_path}):
                print("\n🔍 Step 2: Getting data summary...")
                summary_result = get_data_summary.invoke({"file_path": dataset_path})
                results["analysis_steps"].append({"step": "data_summary", "result": summary_result})
                results["tools_used"].append("get_data_summary")
                print(f"✅ {summary_result[:200]}...")

            with self._trace_step("get_column_info", {"file_path": dataset_path}):
                print("\n🔍 Step 3: Getting column information...")
                column_result = get_column_info.invoke({"file_path": dataset_path})
                results["analysis_steps"].append({"step": "column_info", "result": column_result})
                results["tools_used"].append("get_column_info")
                print(f"✅ {column_result[:200]}...")

            with self._trace_step("preview_data", {"file_path": dataset_path, "num_rows": 5}):
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
            "execution_time": 0,
            "workflow_id": str(uuid.uuid4())
        }

        start_time = time.time()

        try:
            with self._trace_step("read_csv_file", {"file_path": dataset_path}):
                print("🔍 Step 1: Understanding dataset structure...")
                csv_result = read_csv_file.invoke({"file_path": dataset_path})
                results["analysis_steps"].append({"step": "read_csv", "result": csv_result})
                results["tools_used"].append("read_csv_file")

            with self._trace_step("get_column_info", {"file_path": dataset_path}):
                print("🔍 Step 2: Checking for missing values...")
                column_result = get_column_info.invoke({"file_path": dataset_path})
                results["analysis_steps"].append({"step": "column_info", "result": column_result})
                results["tools_used"].append("get_column_info")

            if "age" in dataset_path.lower() or "titanic" in dataset_path.lower():
                with self._trace_step("handle_missing_values", {
                    "file_path": dataset_path,
                    "column_name": "age",
                    "method": "mean"
                }):
                    print("🔧 Step 3: Handling missing values in Age column...")
                    missing_result = handle_missing_values.invoke({
                        "file_path": dataset_path,
                        "column_name": "age",
                        "method": "mean"
                    })
                    results["analysis_steps"].append({"step": "handle_missing", "result": missing_result})
                    results["tools_used"].append("handle_missing_values")
                    print(f"✅ {missing_result}")

            if "titanic" in dataset_path.lower():
                with self._trace_step("create_dummy_variables", {
                    "file_path": dataset_path,
                    "column_name": "sex",
                    "prefix": "sex"
                }):
                    print("🔧 Step 4: Creating dummy variables for sex column...")
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
            "execution_time": 0,
            "workflow_id": str(uuid.uuid4())
        }

        start_time = time.time()

        try:
            with self._trace_step("data_exploration", {"file_path": dataset_path}):
                print("🔍 Step 1: Data exploration...")
                csv_result = read_csv_file.invoke({"file_path": dataset_path})
                summary_result = get_data_summary.invoke({"file_path": dataset_path})
                results["analysis_steps"].extend([
                    {"step": "read_csv", "result": csv_result},
                    {"step": "data_summary", "result": summary_result}
                ])
                results["tools_used"].extend(["read_csv_file", "get_data_summary"])

            if "titanic" in dataset_path.lower():
                features = "pclass,age,sibsp,parch,fare"
                target = "survived"

                with self._trace_step("train_random_forest", {
                    "file_path": dataset_path,
                    "target_column": target,
                    "feature_columns": features
                }):
                    print("🤖 Step 2: Training Random Forest model...")
                    rf_result = train_random_forest_model.invoke({
                        "file_path": dataset_path,
                        "target_column": target,
                        "feature_columns": features,
                        "task_type": "classification"
                    })
                    results["analysis_steps"].append({"step": "train_rf", "result": rf_result})
                    results["tools_used"].append("train_random_forest_model")
                    print(f"✅ {rf_result[:300]}...")

                with self._trace_step("train_svm", {
                    "file_path": dataset_path,
                    "target_column": target,
                    "feature_columns": features
                }):
                    print("🤖 Step 3: Training SVM model for comparison...")
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

    def _trace_step(self, step_name: str, inputs: Dict[str, Any]):
        """Context manager for tracing individual steps."""
        class StepTracer:
            def __init__(self, name, inputs):
                self.name = name
                self.inputs = inputs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return StepTracer(step_name, inputs)

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
    """Main function to test the Simple Universal Agent with LangSmith."""
    print("🌟 Simple Universal Agent with LangSmith Monitoring")
    print("=" * 60)

    # Initialize agent
    agent = SimpleUniversalAgentWithLangSmith()

    # Show available tools
    tools_summary = agent.get_tools_summary()
    print(f"📊 Agent initialized with {tools_summary['total_tools']} tools")
    print(f"  • Data tools: {tools_summary['data_tools']}")
    print(f"  • Manipulation tools: {tools_summary['manipulation_tools']}")
    print(f"  • Operations tools: {tools_summary['operations_tools']}")
    print(f"  • ML tools: {tools_summary['ml_tools']}")

    print("\n" + "=" * 60)

    # Test with default dataset
    dataset_path = config.default_dataset_path

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run: uv run python data/download_titanic.py")
        return

    # Test basic analysis
    print("🧪 Testing Basic Analysis with LangSmith...")
    basic_result = agent.analyze_basic(dataset_path)

    print(f"\n📊 Basic Analysis Results:")
    print(f"Status: {basic_result['status']}")
    print(f"Execution time: {basic_result['execution_time']} seconds")
    print(f"Tools used: {basic_result['tools_used']}")
    print(f"Workflow ID: {basic_result['workflow_id']}")

    if basic_result['status'] == 'success':
        print("✅ Basic analysis completed successfully!")
    else:
        print(f"❌ Error in basic analysis: {basic_result.get('error')}")


if __name__ == "__main__":
    main()