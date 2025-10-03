#!/usr/bin/env python3
"""
Comprehensive Use Cases for Plus-Agent Multi-Agent Data Analysis System

This module defines three detailed use cases that demonstrate the capabilities
of the multi-agent system for different complexity levels of data analysis.
"""

import os
import sys
import time
from typing import Dict, Any, List
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.orchestrator import MultiAgentOrchestrator
from core.config import config
from tests.test_prompts import TestPrompts


class UseCase:
    """Base class for use case scenarios."""

    def __init__(self, name: str, description: str, complexity: str):
        self.name = name
        self.description = description
        self.complexity = complexity
        self.prompts = []
        self.expected_agents = []
        self.results = []

    def add_prompt(self, prompt: str, expected_agents: List[str]):
        """Add a prompt to this use case."""
        self.prompts.append(prompt)
        self.expected_agents.append(expected_agents)

    def execute(self, orchestrator: MultiAgentOrchestrator, dataset_path: str = None) -> Dict[str, Any]:
        """Execute this use case."""
        if dataset_path is None:
            dataset_path = config.default_dataset_path

        print(f"\n{'='*60}")
        print(f"🚀 EXECUTING USE CASE: {self.name}")
        print(f"📋 Description: {self.description}")
        print(f"🎯 Complexity: {self.complexity}")
        print(f"📊 Dataset: {os.path.basename(dataset_path)}")
        print(f"{'='*60}")

        use_case_results = {
            "name": self.name,
            "complexity": self.complexity,
            "description": self.description,
            "dataset_path": dataset_path,
            "prompt_results": [],
            "total_execution_time": 0,
            "success_rate": 0,
            "agents_used": set()
        }

        start_time = time.time()
        successful_prompts = 0

        for i, (prompt, expected_agents) in enumerate(zip(self.prompts, self.expected_agents)):
            print(f"\n📝 Step {i+1}/{len(self.prompts)}: {prompt[:50]}...")
            print(f"🎯 Expected agents: {', '.join(expected_agents)}")

            try:
                # Execute the prompt
                result = orchestrator.run_analysis(prompt, dataset_path)

                # Track results
                use_case_results["prompt_results"].append({
                    "prompt": prompt,
                    "expected_agents": expected_agents,
                    "result": result,
                    "success": result.get("status") == "success"
                })

                # Track agents used
                if "agent_results" in result:
                    use_case_results["agents_used"].update(result["agent_results"].keys())

                if result.get("status") == "success":
                    successful_prompts += 1
                    print(f"✅ Success: {result.get('completed_steps', [])} steps completed")
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"💥 Exception: {str(e)}")
                use_case_results["prompt_results"].append({
                    "prompt": prompt,
                    "expected_agents": expected_agents,
                    "result": {"status": "error", "error": str(e)},
                    "success": False
                })

        # Calculate metrics
        use_case_results["total_execution_time"] = round(time.time() - start_time, 2)
        use_case_results["success_rate"] = round(successful_prompts / len(self.prompts), 2)
        use_case_results["agents_used"] = list(use_case_results["agents_used"])

        print(f"\n📊 USE CASE SUMMARY:")
        print(f"   ⏱️  Total time: {use_case_results['total_execution_time']} seconds")
        print(f"   ✅ Success rate: {use_case_results['success_rate']*100}%")
        print(f"   🤖 Agents used: {', '.join(use_case_results['agents_used'])}")

        return use_case_results


class UseCaseLibrary:
    """Library of comprehensive use cases for the multi-agent system."""

    @staticmethod
    def get_use_case_1_basic_analysis() -> UseCase:
        """
        Use Case 1: Basic Data Analysis (Simple)

        Scenario: A new user wants to explore the Titanic dataset and understand
        its basic structure, contents, and characteristics.

        Expected Flow: DataReader → DataOperations
        """
        use_case = UseCase(
            name="Basic Data Analysis and Exploration",
            description="New user exploring the Titanic dataset to understand its structure and basic statistics",
            complexity="simple"
        )

        # Step 1: Basic dataset overview
        use_case.add_prompt(
            "Show me the basic information about this dataset including shape, columns, and data types",
            ["data_reader"]
        )

        # Step 2: Preview the data
        use_case.add_prompt(
            "Display the first 10 rows of the dataset so I can see what the data looks like",
            ["data_reader"]
        )

        # Step 3: Summary statistics
        use_case.add_prompt(
            "Provide summary statistics for all numeric columns in the dataset",
            ["data_reader", "data_operations"]
        )

        # Step 4: Missing values analysis
        use_case.add_prompt(
            "Identify which columns have missing values and how many are missing",
            ["data_reader"]
        )

        # Step 5: Basic group analysis
        use_case.add_prompt(
            "Show the survival rate by gender to understand basic patterns",
            ["data_operations"]
        )

        return use_case

    @staticmethod
    def get_use_case_2_preprocessing_pipeline() -> UseCase:
        """
        Use Case 2: Data Preprocessing Pipeline (Medium)

        Scenario: A data scientist needs to prepare the Titanic dataset for analysis
        by cleaning the data, handling missing values, and creating new features.

        Expected Flow: DataReader → DataManipulation → DataOperations
        """
        use_case = UseCase(
            name="Data Preprocessing and Feature Engineering Pipeline",
            description="Data scientist preparing the Titanic dataset for machine learning analysis",
            complexity="medium"
        )

        # Step 1: Initial data assessment
        use_case.add_prompt(
            "Analyze the dataset structure and identify data quality issues including missing values, data types, and potential outliers",
            ["data_reader"]
        )

        # Step 2: Handle missing values
        use_case.add_prompt(
            "Handle missing values in the age column using mean imputation and in the embarked column using mode imputation",
            ["data_manipulation"]
        )

        # Step 3: Create dummy variables
        use_case.add_prompt(
            "Create dummy variables for categorical columns including sex, embarked, and class to prepare for machine learning",
            ["data_manipulation"]
        )

        # Step 4: Feature engineering
        use_case.add_prompt(
            "Create a new feature called 'family_size' by adding sibsp and parch columns, then analyze its relationship with survival",
            ["data_manipulation", "data_operations"]
        )

        # Step 5: Data transformation
        use_case.add_prompt(
            "Convert the fare column to integer type and create age groups (child: <18, adult: 18-64, senior: 65+) for better analysis",
            ["data_manipulation"]
        )

        # Step 6: Final data validation
        use_case.add_prompt(
            "Verify the preprocessing results by showing the updated dataset structure, confirming no missing values remain, and displaying sample transformed data",
            ["data_reader", "data_operations"]
        )

        return use_case

    @staticmethod
    def get_use_case_3_complete_ml_workflow() -> UseCase:
        """
        Use Case 3: Complete Machine Learning Workflow (Complex)

        Scenario: A data scientist wants to build a comprehensive machine learning
        model to predict Titanic survival, including full data exploration,
        preprocessing, model training, and evaluation.

        Expected Flow: Planner → DataReader → DataManipulation → DataOperations → MLPrediction
        """
        use_case = UseCase(
            name="End-to-End Machine Learning Prediction Workflow",
            description="Complete data science workflow from exploration to model deployment for Titanic survival prediction",
            complexity="complex"
        )

        # Step 1: Comprehensive planning
        use_case.add_prompt(
            "Create a comprehensive machine learning workflow plan for predicting Titanic passenger survival, including data exploration, preprocessing, feature engineering, model training, and evaluation phases",
            ["planner", "data_reader"]
        )

        # Step 2: Thorough data exploration
        use_case.add_prompt(
            "Perform thorough exploratory data analysis including dataset overview, statistical summaries, missing value analysis, and correlation analysis between features and survival",
            ["data_reader", "data_operations"]
        )

        # Step 3: Advanced preprocessing
        use_case.add_prompt(
            "Implement comprehensive data preprocessing: handle all missing values appropriately, create dummy variables for categorical features, engineer new features (family_size, title extraction from names), and prepare the dataset for machine learning",
            ["data_manipulation", "data_operations"]
        )

        # Step 4: Feature analysis and selection
        use_case.add_prompt(
            "Analyze feature correlations with survival, identify the most important features for prediction, and create the final feature set for model training",
            ["data_operations"]
        )

        # Step 5: Train multiple models
        use_case.add_prompt(
            "Train and compare multiple machine learning models including Random Forest, SVM, and K-Nearest Neighbors for survival prediction, showing their performance metrics",
            ["ml_prediction"]
        )

        # Step 6: Model evaluation and selection
        use_case.add_prompt(
            "Evaluate all trained models using appropriate metrics (accuracy, precision, recall, F1-score), identify the best performing model, and provide feature importance analysis to understand which factors most influence survival",
            ["ml_prediction", "data_operations"]
        )

        # Step 7: Final insights and recommendations
        use_case.add_prompt(
            "Provide comprehensive insights about survival factors based on the analysis and model results, including recommendations for improving passenger safety and key survival predictors",
            ["data_operations"]
        )

        return use_case

    @staticmethod
    def get_all_use_cases() -> List[UseCase]:
        """Get all available use cases."""
        return [
            UseCaseLibrary.get_use_case_1_basic_analysis(),
            UseCaseLibrary.get_use_case_2_preprocessing_pipeline(),
            UseCaseLibrary.get_use_case_3_complete_ml_workflow()
        ]


def main():
    """Main function to demonstrate use cases."""
    print("🤖 Plus-Agent Multi-Agent System Use Cases")
    print("=" * 50)

    # Initialize the system
    print("⚙️  Initializing multi-agent system...")
    try:
        config.setup_langsmith()
        orchestrator = MultiAgentOrchestrator()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return

    # Get all use cases
    use_cases = UseCaseLibrary.get_all_use_cases()

    # Display available use cases
    print(f"\n📋 Available Use Cases ({len(use_cases)}):")
    for i, use_case in enumerate(use_cases, 1):
        print(f"   {i}. {use_case.name} ({use_case.complexity})")
        print(f"      {use_case.description}")

    # Interactive selection
    print(f"\n🎯 Select a use case to execute (1-{len(use_cases)}, or 'all' for all use cases):")
    choice = input("Your choice: ").strip().lower()

    if choice == 'all':
        # Execute all use cases
        all_results = []
        for use_case in use_cases:
            result = use_case.execute(orchestrator)
            all_results.append(result)

        # Summary report
        print(f"\n{'='*60}")
        print("📊 OVERALL SUMMARY REPORT")
        print(f"{'='*60}")
        for result in all_results:
            print(f"🎯 {result['name']}: {result['success_rate']*100}% success in {result['total_execution_time']}s")

    elif choice.isdigit() and 1 <= int(choice) <= len(use_cases):
        # Execute selected use case
        selected_use_case = use_cases[int(choice) - 1]
        result = selected_use_case.execute(orchestrator)

        print(f"\n✅ Use case '{result['name']}' completed successfully!")

    else:
        print("❌ Invalid selection. Please run the script again.")


if __name__ == "__main__":
    main()