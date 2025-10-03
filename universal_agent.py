#!/usr/bin/env python3
"""
Universal Agent for Plus-Agent System

A single powerful agent that has access to all tools from the multi-agent system,
capable of performing complete data science workflows independently.
"""

import os
import sys
import time
from typing import Dict, Any, List
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import tool

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

from core.llm_wrapper import llm_wrapper
from core.config import config


class UniversalAgent:
    """
    A single agent that combines all data science capabilities.
    Has access to all 18+ tools from the original multi-agent system.
    """

    def __init__(self):
        self.llm = llm_wrapper.get_llm_for_agent("universal")

        # Combine all tools from all categories
        self.tools = [
            # Data Reading Tools (5 tools)
            read_csv_file,
            read_json_file,
            get_column_info,
            get_data_summary,
            preview_data,

            # Data Manipulation Tools (4 tools)
            create_dummy_variables,
            modify_column_values,
            handle_missing_values,
            convert_data_types,

            # Data Operations Tools (4 tools)
            filter_data,
            perform_math_operations,
            string_operations,
            aggregate_data,

            # Machine Learning Tools (5 tools)
            train_regression_model,
            train_svm_model,
            train_random_forest_model,
            train_knn_model,
            evaluate_model
        ]

        # Create comprehensive prompt
        self.prompt = PromptTemplate(
            template="""You are a Universal Data Science Agent with access to all tools needed for complete data analysis workflows.

You are an expert data scientist capable of performing complete data science workflows using the available tools.

CAPABILITIES:
- Data Reading: CSV/JSON files, column analysis, summaries, previews
- Data Manipulation: Missing values, dummy variables, type conversion
- Data Operations: Filtering, math operations, string operations, aggregation
- Machine Learning: Multiple algorithms, evaluation, feature importance

INSTRUCTIONS:
1. Always use the correct ReAct format: Thought -> Action -> Action Input -> Observation
2. Start by reading and understanding the dataset structure
3. Use appropriate tools based on the user's request
4. Provide clear explanations of your analysis

Available tools: {tool_names}

{tools}

User Request: {input}

{agent_scratchpad}

You must follow this format:

Thought: I need to understand what the user wants and plan my approach.
Action: tool_name
Action Input: tool_parameters
Observation: [tool result will be inserted here]

Continue this process until you have completed the analysis.""",
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )

        # Create the agent with optimized settings
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=3,  # Reduced for stability
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )

    def analyze(self, user_prompt: str, dataset_path: str) -> Dict[str, Any]:
        """
        Perform data analysis based on user prompt.

        Args:
            user_prompt: User's data analysis request
            dataset_path: Path to the dataset file

        Returns:
            Dictionary containing analysis results
        """
        try:
            print(f"\n🤖 Universal Agent analyzing: {user_prompt}")
            print(f"📊 Dataset: {os.path.basename(dataset_path)}")
            print(f"🔧 Available tools: {len(self.tools)}")
            print("-" * 60)

            # Include dataset path in the prompt
            full_prompt = f"Dataset file path: {dataset_path}\n\nUser request: {user_prompt}"

            start_time = time.time()
            result = self.agent_executor.invoke({"input": full_prompt})
            execution_time = time.time() - start_time

            return {
                "status": "success",
                "user_prompt": user_prompt,
                "dataset_path": dataset_path,
                "result": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "execution_time": round(execution_time, 2),
                "tools_used": len(result.get("intermediate_steps", [])),
                "agent_type": "universal"
            }

        except Exception as e:
            return {
                "status": "error",
                "user_prompt": user_prompt,
                "dataset_path": dataset_path,
                "error": str(e),
                "agent_type": "universal"
            }

    def get_tools_summary(self) -> Dict[str, Any]:
        """Get summary of all available tools."""
        tool_categories = {
            "Data Reading": [
                "read_csv_file", "read_json_file", "get_column_info",
                "get_data_summary", "preview_data"
            ],
            "Data Manipulation": [
                "create_dummy_variables", "modify_column_values",
                "handle_missing_values", "convert_data_types"
            ],
            "Data Operations": [
                "filter_data", "perform_math_operations",
                "string_operations", "aggregate_data"
            ],
            "Machine Learning": [
                "train_regression_model", "train_svm_model",
                "train_random_forest_model", "train_knn_model", "evaluate_model"
            ]
        }

        return {
            "total_tools": len(self.tools),
            "categories": tool_categories,
            "tool_names": [tool.name for tool in self.tools]
        }


def main():
    """Main function to test the Universal Agent."""
    print("🌟 Universal Agent for Data Science")
    print("=" * 50)

    # Initialize agent
    agent = UniversalAgent()

    # Show available tools
    tools_summary = agent.get_tools_summary()
    print(f"📊 Agent initialized with {tools_summary['total_tools']} tools")

    for category, tools in tools_summary['categories'].items():
        print(f"  • {category}: {len(tools)} tools")

    print("\n" + "=" * 50)

    # Test with default dataset
    dataset_path = config.default_dataset_path

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run: uv run python data/download_titanic.py")
        return

    # Simple test prompt
    test_prompt = "Show me basic information about this dataset and provide a summary"

    print(f"🧪 Testing with prompt: {test_prompt}")
    result = agent.analyze(test_prompt, dataset_path)

    print("\n" + "=" * 50)
    print("📊 RESULTS:")
    print(f"Status: {result['status']}")

    if result['status'] == 'success':
        print(f"⏱️  Execution time: {result['execution_time']} seconds")
        print(f"🔧 Tools used: {result['tools_used']}")
        print(f"📝 Result: {result['result'][:500]}...")
        print("\n✅ Universal Agent test completed successfully!")
    else:
        print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    main()