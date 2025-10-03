#!/usr/bin/env python3
"""
Test Universal Agent with Use Cases

This script tests the Universal Agent with the same use cases as the multi-agent system
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

from universal_agent import UniversalAgent
from use_cases import UseCaseLibrary
from core.config import config


class UniversalAgentTester:
    """Test the Universal Agent with predefined use cases."""

    def __init__(self):
        self.agent = UniversalAgent()
        self.use_case_library = UseCaseLibrary()

    def execute_use_case_with_universal_agent(self, use_case, dataset_path: str = None) -> Dict[str, Any]:
        """Execute a use case using the Universal Agent."""
        if dataset_path is None:
            dataset_path = config.default_dataset_path

        print(f"\n{'='*60}")
        print(f"🚀 EXECUTING USE CASE WITH UNIVERSAL AGENT: {use_case.name}")
        print(f"📋 Description: {use_case.description}")
        print(f"🎯 Complexity: {use_case.complexity}")
        print(f"📊 Dataset: {os.path.basename(dataset_path)}")
        print(f"🔧 Total prompts: {len(use_case.prompts)}")
        print(f"{'='*60}")

        use_case_results = {
            "name": use_case.name,
            "complexity": use_case.complexity,
            "description": use_case.description,
            "dataset_path": dataset_path,
            "prompt_results": [],
            "total_execution_time": 0,
            "success_rate": 0,
            "total_tools_used": 0
        }

        start_time = time.time()
        successful_prompts = 0

        for i, prompt in enumerate(use_case.prompts):
            print(f"\n📝 Step {i+1}/{len(use_case.prompts)}: {prompt[:60]}...")

            try:
                # Execute the prompt with Universal Agent
                result = self.agent.analyze(prompt, dataset_path)

                # Track results
                use_case_results["prompt_results"].append({
                    "prompt": prompt,
                    "result": result,
                    "success": result.get("status") == "success"
                })

                # Track tools used
                tools_used = result.get("tools_used", 0)
                use_case_results["total_tools_used"] += tools_used

                if result.get("status") == "success":
                    successful_prompts += 1
                    print(f"✅ Success: {tools_used} tools used, {result.get('execution_time', 0)}s")
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"💥 Exception: {str(e)}")
                use_case_results["prompt_results"].append({
                    "prompt": prompt,
                    "result": {"status": "error", "error": str(e)},
                    "success": False
                })

        # Calculate final metrics
        total_time = time.time() - start_time
        use_case_results["total_execution_time"] = round(total_time, 2)
        use_case_results["success_rate"] = round((successful_prompts / len(use_case.prompts)) * 100, 1)

        # Print summary
        print(f"\n📊 USE CASE SUMMARY:")
        print(f"   ⏱️  Total time: {use_case_results['total_execution_time']} seconds")
        print(f"   ✅ Success rate: {use_case_results['success_rate']}%")
        print(f"   🔧 Total tools used: {use_case_results['total_tools_used']}")
        print(f"   📝 Prompts completed: {successful_prompts}/{len(use_case.prompts)}")

        return use_case_results

    def run_all_use_cases(self) -> List[Dict[str, Any]]:
        """Run all three use cases with the Universal Agent."""
        print("🤖 Universal Agent - Use Case Testing")
        print("=" * 60)

        # Get all use cases
        use_case_1 = self.use_case_library.get_use_case_1_basic_analysis()
        use_case_2 = self.use_case_library.get_use_case_2_preprocessing()
        use_case_3 = self.use_case_library.get_use_case_3_ml_workflow()

        all_results = []

        # Execute each use case
        for i, use_case in enumerate([use_case_1, use_case_2, use_case_3], 1):
            print(f"\n🎯 Starting Use Case {i}/3...")
            result = self.execute_use_case_with_universal_agent(use_case)
            all_results.append(result)

        # Print overall summary
        self.print_overall_summary(all_results)

        return all_results

    def run_single_use_case(self, use_case_number: int) -> Dict[str, Any]:
        """Run a single use case."""
        use_cases = {
            1: self.use_case_library.get_use_case_1_basic_analysis(),
            2: self.use_case_library.get_use_case_2_preprocessing(),
            3: self.use_case_library.get_use_case_3_ml_workflow()
        }

        if use_case_number not in use_cases:
            raise ValueError(f"Invalid use case number: {use_case_number}")

        use_case = use_cases[use_case_number]
        return self.execute_use_case_with_universal_agent(use_case)

    def print_overall_summary(self, results: List[Dict[str, Any]]):
        """Print overall summary of all use case results."""
        print(f"\n{'='*60}")
        print("🎉 UNIVERSAL AGENT - OVERALL SUMMARY")
        print(f"{'='*60}")

        total_time = sum(r["total_execution_time"] for r in results)
        total_success_rate = sum(r["success_rate"] for r in results) / len(results)
        total_tools = sum(r["total_tools_used"] for r in results)

        print(f"📊 Executed {len(results)} use cases")
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"✅ Average success rate: {total_success_rate:.1f}%")
        print(f"🔧 Total tools used: {total_tools}")

        print(f"\n📋 Use Case Breakdown:")
        for i, result in enumerate(results, 1):
            status = "✅" if result["success_rate"] >= 80 else "⚠️" if result["success_rate"] >= 50 else "❌"
            print(f"   {status} Use Case {i} ({result['complexity']}): {result['success_rate']}% - {result['total_execution_time']}s")

        print(f"\n🤖 Universal Agent Performance:")
        print(f"   • Single agent with {len(self.agent.tools)} tools")
        print(f"   • Average time per use case: {total_time/len(results):.2f}s")
        print(f"   • Average tools per use case: {total_tools/len(results):.1f}")

        # Performance assessment
        if total_success_rate >= 90:
            assessment = "🌟 EXCELLENT - Universal Agent working perfectly!"
        elif total_success_rate >= 70:
            assessment = "👍 GOOD - Universal Agent performing well"
        elif total_success_rate >= 50:
            assessment = "⚠️ MODERATE - Some issues detected"
        else:
            assessment = "❌ POOR - Significant issues detected"

        print(f"\n🎯 Overall Assessment: {assessment}")


def main():
    """Main function for interactive use case testing."""
    print("🌟 Universal Agent Use Case Tester")
    print("=" * 50)

    # Check dataset
    dataset_path = config.default_dataset_path
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run: uv run python data/download_titanic.py")
        return

    # Initialize tester
    tester = UniversalAgentTester()

    # Show available tools
    tools_summary = tester.agent.get_tools_summary()
    print(f"🔧 Universal Agent loaded with {tools_summary['total_tools']} tools")

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