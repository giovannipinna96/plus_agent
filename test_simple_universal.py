#!/usr/bin/env python3
"""
Test Simple Universal Agent with Use Cases

This script tests the Simple Universal Agent with use cases similar to the multi-agent system
to compare performance and functionality.
"""

import os
import sys
import time
from typing import Dict, Any, List
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from simple_universal_agent import SimpleUniversalAgent
from core.config import config


class SimpleUniversalTester:
    """Test the Simple Universal Agent with predefined use cases."""

    def __init__(self):
        self.agent = SimpleUniversalAgent()

    def run_use_case_1_basic(self, dataset_path: str = None) -> Dict[str, Any]:
        """Run Use Case 1: Basic Data Analysis."""
        if dataset_path is None:
            dataset_path = config.default_dataset_path

        print(f"\n{'='*60}")
        print("🚀 USE CASE 1: Basic Data Analysis and Exploration")
        print("📋 Description: New user exploring the Titanic dataset")
        print("🎯 Complexity: Simple")
        print(f"📊 Dataset: {os.path.basename(dataset_path)}")
        print(f"{'='*60}")

        start_time = time.time()
        result = self.agent.analyze_basic(dataset_path)
        result["use_case"] = "Basic Data Analysis"
        result["complexity"] = "simple"

        print(f"\n📊 USE CASE 1 SUMMARY:")
        print(f"   ⏱️  Total time: {result['execution_time']} seconds")
        print(f"   ✅ Status: {result['status']}")
        print(f"   🔧 Tools used: {len(result['tools_used'])}")
        print(f"   📝 Steps completed: {len(result['analysis_steps'])}")

        return result

    def run_use_case_2_preprocessing(self, dataset_path: str = None) -> Dict[str, Any]:
        """Run Use Case 2: Data Preprocessing Pipeline."""
        if dataset_path is None:
            dataset_path = config.default_dataset_path

        print(f"\n{'='*60}")
        print("🚀 USE CASE 2: Data Preprocessing and Feature Engineering Pipeline")
        print("📋 Description: Data scientist preparing data for analysis")
        print("🎯 Complexity: Medium")
        print(f"📊 Dataset: {os.path.basename(dataset_path)}")
        print(f"{'='*60}")

        start_time = time.time()
        result = self.agent.analyze_preprocessing(dataset_path)
        result["use_case"] = "Data Preprocessing Pipeline"
        result["complexity"] = "medium"

        print(f"\n📊 USE CASE 2 SUMMARY:")
        print(f"   ⏱️  Total time: {result['execution_time']} seconds")
        print(f"   ✅ Status: {result['status']}")
        print(f"   🔧 Tools used: {len(result['tools_used'])}")
        print(f"   📝 Steps completed: {len(result['analysis_steps'])}")

        return result

    def run_use_case_3_ml_workflow(self, dataset_path: str = None) -> Dict[str, Any]:
        """Run Use Case 3: Complete ML Workflow."""
        if dataset_path is None:
            dataset_path = config.default_dataset_path

        print(f"\n{'='*60}")
        print("🚀 USE CASE 3: End-to-End Machine Learning Prediction Workflow")
        print("📋 Description: ML engineer building complete analysis pipeline")
        print("🎯 Complexity: Complex")
        print(f"📊 Dataset: {os.path.basename(dataset_path)}")
        print(f"{'='*60}")

        start_time = time.time()
        result = self.agent.analyze_ml_workflow(dataset_path)
        result["use_case"] = "Complete ML Workflow"
        result["complexity"] = "complex"

        print(f"\n📊 USE CASE 3 SUMMARY:")
        print(f"   ⏱️  Total time: {result['execution_time']} seconds")
        print(f"   ✅ Status: {result['status']}")
        print(f"   🔧 Tools used: {len(result['tools_used'])}")
        print(f"   📝 Steps completed: {len(result['analysis_steps'])}")

        return result

    def run_all_use_cases(self) -> List[Dict[str, Any]]:
        """Run all three use cases with the Simple Universal Agent."""
        print("🤖 Simple Universal Agent - Use Case Testing")
        print("=" * 60)

        all_results = []

        # Execute each use case
        try:
            result1 = self.run_use_case_1_basic()
            all_results.append(result1)
        except Exception as e:
            print(f"❌ Use Case 1 failed: {e}")
            all_results.append({"status": "error", "error": str(e), "use_case": "Basic Data Analysis"})

        try:
            result2 = self.run_use_case_2_preprocessing()
            all_results.append(result2)
        except Exception as e:
            print(f"❌ Use Case 2 failed: {e}")
            all_results.append({"status": "error", "error": str(e), "use_case": "Data Preprocessing Pipeline"})

        try:
            result3 = self.run_use_case_3_ml_workflow()
            all_results.append(result3)
        except Exception as e:
            print(f"❌ Use Case 3 failed: {e}")
            all_results.append({"status": "error", "error": str(e), "use_case": "Complete ML Workflow"})

        # Print overall summary
        self.print_overall_summary(all_results)

        return all_results

    def run_single_use_case(self, use_case_number: int) -> Dict[str, Any]:
        """Run a single use case."""
        if use_case_number == 1:
            return self.run_use_case_1_basic()
        elif use_case_number == 2:
            return self.run_use_case_2_preprocessing()
        elif use_case_number == 3:
            return self.run_use_case_3_ml_workflow()
        else:
            raise ValueError(f"Invalid use case number: {use_case_number}")

    def print_overall_summary(self, results: List[Dict[str, Any]]):
        """Print overall summary of all use case results."""
        print(f"\n{'='*60}")
        print("🎉 SIMPLE UNIVERSAL AGENT - OVERALL SUMMARY")
        print(f"{'='*60}")

        successful_results = [r for r in results if r.get("status") == "success"]
        total_time = sum(r.get("execution_time", 0) for r in successful_results)
        success_rate = (len(successful_results) / len(results)) * 100
        total_tools = sum(len(r.get("tools_used", [])) for r in successful_results)

        print(f"📊 Executed {len(results)} use cases")
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"✅ Success rate: {success_rate:.1f}%")
        print(f"🔧 Total tools used: {total_tools}")

        print(f"\n📋 Use Case Breakdown:")
        for i, result in enumerate(results, 1):
            status = "✅" if result.get("status") == "success" else "❌"
            complexity = result.get("complexity", "unknown")
            exec_time = result.get("execution_time", 0)
            print(f"   {status} Use Case {i} ({complexity}): {result.get('status')} - {exec_time}s")

        print(f"\n🤖 Simple Universal Agent Performance:")
        tools_summary = self.agent.get_tools_summary()
        print(f"   • Single agent with {tools_summary['total_tools']} tools")
        print(f"   • Direct tool invocation (no LLM overhead)")
        if successful_results:
            print(f"   • Average time per use case: {total_time/len(successful_results):.2f}s")
            print(f"   • Average tools per use case: {total_tools/len(successful_results):.1f}")

        # Performance assessment
        if success_rate >= 90:
            assessment = "🌟 EXCELLENT - Simple Universal Agent working perfectly!"
        elif success_rate >= 70:
            assessment = "👍 GOOD - Simple Universal Agent performing well"
        elif success_rate >= 50:
            assessment = "⚠️ MODERATE - Some issues detected"
        else:
            assessment = "❌ POOR - Significant issues detected"

        print(f"\n🎯 Overall Assessment: {assessment}")

        # Comparison with multi-agent system
        print(f"\n🔄 Comparison Benefits:")
        print(f"   ✅ Simplified architecture - single agent vs 5 agents")
        print(f"   ✅ No LLM prompt processing overhead")
        print(f"   ✅ Direct tool access and execution")
        print(f"   ✅ Predictable and consistent results")
        print(f"   ✅ Faster execution for simple operations")


def main():
    """Main function for interactive use case testing."""
    print("🌟 Simple Universal Agent Use Case Tester")
    print("=" * 50)

    # Check dataset
    dataset_path = config.default_dataset_path
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run: uv run python data/download_titanic.py")
        return

    # Initialize tester
    tester = SimpleUniversalTester()

    # Show available tools
    tools_summary = tester.agent.get_tools_summary()
    print(f"🔧 Simple Universal Agent loaded with {tools_summary['total_tools']} tools")

    print("\n📋 Available Use Cases:")
    print("   1. Basic Data Analysis and Exploration (simple)")
    print("   2. Data Preprocessing and Feature Engineering Pipeline (medium)")
    print("   3. End-to-End Machine Learning Prediction Workflow (complex)")
    print("   all. Run all use cases sequentially")

    # Get user choice
    while True:
        try:
            choice = input("\n🎯 Select a use case to execute (1-3, or 'all'): ").strip().lower()

            if choice == 'all':
                results = tester.run_all_use_cases()
                break
            elif choice in ['1', '2', '3']:
                use_case_num = int(choice)
                result = tester.run_single_use_case(use_case_num)
                print(f"\n✅ Use case {use_case_num} completed!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 'all'")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()