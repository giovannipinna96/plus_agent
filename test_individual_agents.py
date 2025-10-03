#!/usr/bin/env python3
"""
Individual Agent Testing Script for Plus-Agent Multi-Agent System

This script tests each agent individually to verify their functionality
and tool integration before running the complete orchestrated workflow.
"""

import os
import sys
import time
from typing import Dict, Any, List
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import config
from core.langsmith_integration import langsmith_logger
from agents.planner_agent import PlannerAgent
from agents.data_reader_agent import DataReaderAgent
from agents.data_manipulation_agent import DataManipulationAgent
from agents.data_operations_agent import DataOperationsAgent
from agents.ml_prediction_agent import MLPredictionAgent


class AgentTester:
    """Individual agent testing framework."""

    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0

    def test_agent(self, agent_name: str, agent_instance, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test an individual agent with multiple test cases."""
        print(f"\n{'='*60}")
        print(f"🧪 TESTING {agent_name.upper()} AGENT")
        print(f"{'='*60}")

        agent_results = {
            "agent_name": agent_name,
            "test_cases": [],
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "execution_time": 0,
            "success_rate": 0
        }

        start_time = time.time()

        for i, test_case in enumerate(test_cases):
            test_name = test_case["name"]
            test_method = test_case["method"]
            test_args = test_case.get("args", [])
            test_kwargs = test_case.get("kwargs", {})
            expected_keys = test_case.get("expected_keys", [])

            print(f"\n📝 Test {i+1}/{len(test_cases)}: {test_name}")
            print(f"   Method: {test_method}")

            try:
                # Execute the test
                if hasattr(agent_instance, test_method):
                    method = getattr(agent_instance, test_method)
                    result = method(*test_args, **test_kwargs)

                    # Validate result
                    success = self._validate_result(result, expected_keys)

                    if success:
                        print(f"   ✅ PASSED: {result.get('status', 'Success')}")
                        agent_results["passed_tests"] += 1
                        self.passed_tests += 1
                    else:
                        print(f"   ❌ FAILED: Missing expected keys or invalid result")

                    agent_results["test_cases"].append({
                        "name": test_name,
                        "method": test_method,
                        "success": success,
                        "result": result
                    })

                else:
                    print(f"   💥 ERROR: Method '{test_method}' not found")
                    agent_results["test_cases"].append({
                        "name": test_name,
                        "method": test_method,
                        "success": False,
                        "result": {"error": f"Method '{test_method}' not found"}
                    })

            except Exception as e:
                print(f"   💥 EXCEPTION: {str(e)}")
                agent_results["test_cases"].append({
                    "name": test_name,
                    "method": test_method,
                    "success": False,
                    "result": {"error": str(e)}
                })

            self.total_tests += 1

        # Calculate metrics
        agent_results["execution_time"] = round(time.time() - start_time, 2)
        agent_results["success_rate"] = round(agent_results["passed_tests"] / agent_results["total_tests"], 2)

        print(f"\n📊 {agent_name.upper()} AGENT SUMMARY:")
        print(f"   ✅ Passed: {agent_results['passed_tests']}/{agent_results['total_tests']}")
        print(f"   📈 Success Rate: {agent_results['success_rate']*100}%")
        print(f"   ⏱️  Execution Time: {agent_results['execution_time']} seconds")

        self.results[agent_name] = agent_results
        return agent_results

    def _validate_result(self, result: Any, expected_keys: List[str]) -> bool:
        """Validate that the result contains expected keys and structure."""
        if not isinstance(result, dict):
            return False

        # Check for expected keys
        for key in expected_keys:
            if key not in result:
                return False

        # Check for status if it's expected
        if "status" in result:
            return result["status"] in ["success", "completed"]

        return True

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        overall_success_rate = round(self.passed_tests / self.total_tests, 2) if self.total_tests > 0 else 0

        report = {
            "overall_summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "success_rate": overall_success_rate,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "agent_results": self.results
        }

        print(f"\n{'='*60}")
        print("📊 COMPREHENSIVE TEST REPORT")
        print(f"{'='*60}")
        print(f"🎯 Overall Success Rate: {overall_success_rate*100}%")
        print(f"✅ Total Passed: {self.passed_tests}/{self.total_tests}")
        print(f"📅 Timestamp: {report['overall_summary']['timestamp']}")

        print(f"\n📋 Agent-by-Agent Summary:")
        for agent_name, results in self.results.items():
            print(f"   🤖 {agent_name}: {results['success_rate']*100}% ({results['passed_tests']}/{results['total_tests']})")

        return report


def main():
    """Main function to test all individual agents."""
    print("🧪 Plus-Agent Individual Agent Testing Suite")
    print("=" * 50)

    # Initialize configuration
    config.setup_langsmith()

    # Ensure default dataset exists
    if not os.path.exists(config.default_dataset_path):
        print("⚠️  Default dataset not found. Downloading...")
        try:
            from data.download_titanic import main as download_titanic
            download_titanic()
        except Exception as e:
            print(f"❌ Failed to download dataset: {e}")
            return

    # Initialize tester
    tester = AgentTester()

    print("🚀 Starting individual agent tests...")

    # Test 1: Planner Agent
    print("\n🎭 Initializing Planner Agent...")
    try:
        planner = PlannerAgent()
        planner_tests = [
            {
                "name": "Simple Planning Request",
                "method": "plan",
                "args": ["Show me basic information about the dataset"],
                "expected_keys": ["plan", "status"]
            },
            {
                "name": "Complex Planning Request",
                "method": "plan",
                "args": ["Train a machine learning model and evaluate its performance"],
                "expected_keys": ["plan", "status"]
            }
        ]
        tester.test_agent("planner", planner, planner_tests)
    except Exception as e:
        print(f"❌ Failed to initialize Planner Agent: {e}")

    # Test 2: Data Reader Agent
    print("\n📊 Initializing Data Reader Agent...")
    try:
        data_reader = DataReaderAgent()
        data_reader_tests = [
            {
                "name": "Analyze Dataset Structure",
                "method": "analyze_data",
                "args": [config.default_dataset_path, "basic"],
                "expected_keys": ["analysis", "status"]
            },
            {
                "name": "Comprehensive Data Analysis",
                "method": "analyze_data",
                "args": [config.default_dataset_path, "comprehensive"],
                "expected_keys": ["analysis", "status"]
            }
        ]
        tester.test_agent("data_reader", data_reader, data_reader_tests)
    except Exception as e:
        print(f"❌ Failed to initialize Data Reader Agent: {e}")

    # Test 3: Data Manipulation Agent
    print("\n🔧 Initializing Data Manipulation Agent...")
    try:
        data_manipulation = DataManipulationAgent()
        manipulation_tests = [
            {
                "name": "Handle Missing Values",
                "method": "manipulate_data",
                "args": [config.default_dataset_path, "Handle missing values in the age column using mean imputation"],
                "expected_keys": ["result", "status"]
            },
            {
                "name": "Create Dummy Variables",
                "method": "manipulate_data",
                "args": [config.default_dataset_path, "Create dummy variables for the sex column"],
                "expected_keys": ["result", "status"]
            }
        ]
        tester.test_agent("data_manipulation", data_manipulation, manipulation_tests)
    except Exception as e:
        print(f"❌ Failed to initialize Data Manipulation Agent: {e}")

    # Test 4: Data Operations Agent
    print("\n⚡ Initializing Data Operations Agent...")
    try:
        data_operations = DataOperationsAgent()
        operations_tests = [
            {
                "name": "Calculate Survival Rate by Gender",
                "method": "perform_operations",
                "args": [config.default_dataset_path, "Calculate the survival rate by gender"],
                "expected_keys": ["result", "status"]
            },
            {
                "name": "Filter and Analyze Data",
                "method": "perform_operations",
                "args": [config.default_dataset_path, "Filter passengers older than 30 and show their survival rate"],
                "expected_keys": ["result", "status"]
            }
        ]
        tester.test_agent("data_operations", data_operations, operations_tests)
    except Exception as e:
        print(f"❌ Failed to initialize Data Operations Agent: {e}")

    # Test 5: ML Prediction Agent
    print("\n🎯 Initializing ML Prediction Agent...")
    try:
        ml_prediction = MLPredictionAgent()
        ml_tests = [
            {
                "name": "Train Random Forest Model",
                "method": "train_model",
                "args": [config.default_dataset_path, "Train a random forest model to predict survival"],
                "expected_keys": ["result", "status"]
            },
            {
                "name": "Train and Compare Multiple Models",
                "method": "train_model",
                "args": [config.default_dataset_path, "Train and compare Random Forest and SVM models for survival prediction"],
                "expected_keys": ["result", "status"]
            }
        ]
        tester.test_agent("ml_prediction", ml_prediction, ml_tests)
    except Exception as e:
        print(f"❌ Failed to initialize ML Prediction Agent: {e}")

    # Generate final report
    report = tester.generate_report()

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if report["overall_summary"]["success_rate"] >= 0.8:
        print("   🎉 Excellent! All agents are functioning well.")
        print("   ✅ System is ready for production use.")
    elif report["overall_summary"]["success_rate"] >= 0.6:
        print("   ⚠️  Good performance, but some issues detected.")
        print("   🔧 Review failed tests and address any issues.")
    else:
        print("   ❌ Critical issues detected with multiple agents.")
        print("   🚨 System requires attention before production use.")

    # Save report
    import json
    report_path = "test_results_individual_agents.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Test report saved to: {report_path}")


if __name__ == "__main__":
    main()