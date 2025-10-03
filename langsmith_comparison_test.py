#!/usr/bin/env python3
"""
LangSmith Monitoring Comparison Test

This script runs both Multi-Agent and Universal Agent systems with LangSmith monitoring
to provide detailed performance comparison and observability insights.
"""

import os
import sys
import time
from typing import Dict, Any, List
from pathlib import Path
import uuid

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from simple_universal_agent_langsmith import SimpleUniversalAgentWithLangSmith
from core.orchestrator import MultiAgentOrchestrator
from use_cases import UseCaseLibrary
from core.config import config


class LangSmithComparisonTester:
    """
    Comprehensive comparison tester with LangSmith monitoring.
    """

    def __init__(self):
        self.setup_langsmith()
        self.universal_agent = SimpleUniversalAgentWithLangSmith()
        self.multi_agent_orchestrator = MultiAgentOrchestrator()
        self.use_case_library = UseCaseLibrary()

    def setup_langsmith(self):
        """Setup LangSmith for both systems."""
        api_key = ""

        # Set environment variables
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGSMITH_TRACING"] = "true"

        print("🔍 LangSmith monitoring configured for both systems")

    def test_multi_agent_system(self, use_case_number: int) -> Dict[str, Any]:
        """Test the multi-agent system with LangSmith monitoring."""
        print(f"\n{'='*60}")
        print(f"🔬 TESTING MULTI-AGENT SYSTEM - USE CASE {use_case_number}")
        print(f"{'='*60}")

        # Set project for multi-agent system
        os.environ["LANGCHAIN_PROJECT"] = f"multi-agent-use-case-{use_case_number}"

        # Get use case
        use_cases = {
            1: self.use_case_library.get_use_case_1_basic_analysis(),
            2: self.use_case_library.get_use_case_2_preprocessing_pipeline(),
            3: self.use_case_library.get_use_case_3_complete_ml_workflow()
        }

        use_case = use_cases[use_case_number]
        dataset_path = config.default_dataset_path

        print(f"📋 Use Case: {use_case.name}")
        print(f"🎯 Complexity: {use_case.complexity}")
        print(f"📊 Prompts: {len(use_case.prompts)}")

        results = {
            "system": "multi-agent",
            "use_case": use_case.name,
            "complexity": use_case.complexity,
            "execution_times": [],
            "total_time": 0,
            "success_count": 0,
            "error_count": 0,
            "workflow_id": str(uuid.uuid4()),
            "langsmith_project": os.environ["LANGCHAIN_PROJECT"]
        }

        overall_start = time.time()

        for i, prompt in enumerate(use_case.prompts, 1):
            print(f"\n📝 Executing prompt {i}/{len(use_case.prompts)}: {prompt[:60]}...")

            try:
                start_time = time.time()
                result = self.multi_agent_orchestrator.run_analysis(prompt, dataset_path)
                execution_time = time.time() - start_time

                results["execution_times"].append(execution_time)

                if result.get("status") == "success":
                    results["success_count"] += 1
                    print(f"✅ Success in {execution_time:.2f}s")
                else:
                    results["error_count"] += 1
                    print(f"❌ Failed in {execution_time:.2f}s")

            except Exception as e:
                results["error_count"] += 1
                print(f"💥 Exception: {e}")

        results["total_time"] = time.time() - overall_start
        results["success_rate"] = (results["success_count"] / len(use_case.prompts)) * 100

        print(f"\n📊 MULTI-AGENT RESULTS:")
        print(f"   ⏱️  Total time: {results['total_time']:.2f}s")
        print(f"   ✅ Success rate: {results['success_rate']:.1f}%")
        print(f"   🔗 LangSmith project: {results['langsmith_project']}")

        return results

    def test_universal_agent_system(self, use_case_number: int) -> Dict[str, Any]:
        """Test the universal agent system with LangSmith monitoring."""
        print(f"\n{'='*60}")
        print(f"🔬 TESTING UNIVERSAL AGENT SYSTEM - USE CASE {use_case_number}")
        print(f"{'='*60}")

        # Set project for universal agent system
        os.environ["LANGCHAIN_PROJECT"] = f"universal-agent-use-case-{use_case_number}"

        dataset_path = config.default_dataset_path

        results = {
            "system": "universal-agent",
            "use_case": f"Use Case {use_case_number}",
            "execution_time": 0,
            "success": False,
            "workflow_id": str(uuid.uuid4()),
            "langsmith_project": os.environ["LANGCHAIN_PROJECT"]
        }

        print(f"📊 Dataset: {os.path.basename(dataset_path)}")

        try:
            start_time = time.time()

            if use_case_number == 1:
                result = self.universal_agent.analyze_basic(dataset_path)
            elif use_case_number == 2:
                result = self.universal_agent.analyze_preprocessing(dataset_path)
            elif use_case_number == 3:
                result = self.universal_agent.analyze_ml_workflow(dataset_path)
            else:
                raise ValueError(f"Invalid use case number: {use_case_number}")

            results["execution_time"] = time.time() - start_time
            results["success"] = result.get("status") == "success"
            results["tools_used"] = len(result.get("tools_used", []))
            results["steps_completed"] = len(result.get("analysis_steps", []))

        except Exception as e:
            results["execution_time"] = time.time() - start_time
            results["success"] = False
            results["error"] = str(e)
            print(f"💥 Exception: {e}")

        print(f"\n📊 UNIVERSAL AGENT RESULTS:")
        print(f"   ⏱️  Execution time: {results['execution_time']:.2f}s")
        print(f"   ✅ Success: {results['success']}")
        if results['success']:
            print(f"   🔧 Tools used: {results.get('tools_used', 0)}")
            print(f"   📝 Steps completed: {results.get('steps_completed', 0)}")
        print(f"   🔗 LangSmith project: {results['langsmith_project']}")

        return results

    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """Run comprehensive comparison of both systems."""
        print("🔬 COMPREHENSIVE LANGSMITH COMPARISON TEST")
        print("=" * 60)
        print("This test will run both systems with LangSmith monitoring")
        print("for detailed performance analysis and observability.")
        print("=" * 60)

        comparison_results = {
            "multi_agent_results": [],
            "universal_agent_results": [],
            "comparison_summary": {}
        }

        # Test all three use cases
        for use_case_num in [1, 2, 3]:
            print(f"\n🎯 TESTING USE CASE {use_case_num}")
            print("-" * 40)

            # Test Multi-Agent System
            multi_agent_result = self.test_multi_agent_system(use_case_num)
            comparison_results["multi_agent_results"].append(multi_agent_result)

            # Test Universal Agent System
            universal_agent_result = self.test_universal_agent_system(use_case_num)
            comparison_results["universal_agent_results"].append(universal_agent_result)

            # Compare results
            self.print_use_case_comparison(multi_agent_result, universal_agent_result, use_case_num)

        # Generate overall summary
        self.generate_final_summary(comparison_results)

        return comparison_results

    def print_use_case_comparison(self, multi_result: Dict[str, Any],
                                universal_result: Dict[str, Any], use_case_num: int):
        """Print comparison for a single use case."""
        print(f"\n📊 USE CASE {use_case_num} COMPARISON:")
        print(f"{'='*50}")

        # Performance comparison
        multi_time = multi_result.get("total_time", 0)
        universal_time = universal_result.get("execution_time", 0)

        if universal_time > 0:
            speedup = multi_time / universal_time
            print(f"⏱️  Execution Time:")
            print(f"   • Multi-Agent: {multi_time:.2f}s")
            print(f"   • Universal Agent: {universal_time:.2f}s")
            print(f"   • Speedup: {speedup:.1f}x faster")

        # Success rate comparison
        multi_success = multi_result.get("success_rate", 0)
        universal_success = 100 if universal_result.get("success", False) else 0

        print(f"✅ Success Rate:")
        print(f"   • Multi-Agent: {multi_success:.1f}%")
        print(f"   • Universal Agent: {universal_success:.1f}%")

        # LangSmith projects
        print(f"🔗 LangSmith Projects:")
        print(f"   • Multi-Agent: {multi_result.get('langsmith_project')}")
        print(f"   • Universal Agent: {universal_result.get('langsmith_project')}")

    def generate_final_summary(self, results: Dict[str, Any]):
        """Generate final comprehensive summary."""
        print(f"\n{'='*60}")
        print("🎉 FINAL LANGSMITH COMPARISON SUMMARY")
        print(f"{'='*60}")

        multi_results = results["multi_agent_results"]
        universal_results = results["universal_agent_results"]

        # Calculate totals
        total_multi_time = sum(r.get("total_time", 0) for r in multi_results)
        total_universal_time = sum(r.get("execution_time", 0) for r in universal_results)

        print(f"📊 OVERALL PERFORMANCE:")
        print(f"   • Multi-Agent Total Time: {total_multi_time:.2f}s")
        print(f"   • Universal Agent Total Time: {total_universal_time:.2f}s")

        if total_universal_time > 0:
            overall_speedup = total_multi_time / total_universal_time
            print(f"   • Overall Speedup: {overall_speedup:.1f}x faster")

        print(f"\n🔗 LANGSMITH PROJECTS CREATED:")
        print(f"   Multi-Agent Projects:")
        for i, result in enumerate(multi_results, 1):
            print(f"     • Use Case {i}: {result.get('langsmith_project')}")

        print(f"   Universal Agent Projects:")
        for i, result in enumerate(universal_results, 1):
            print(f"     • Use Case {i}: {result.get('langsmith_project')}")

        print(f"\n📈 KEY INSIGHTS:")
        print(f"   ✅ Both systems monitored with LangSmith")
        print(f"   ✅ Detailed traces available in LangSmith dashboard")
        print(f"   ✅ Performance comparison completed")
        print(f"   ✅ Ready for production observability")

        print(f"\n🎯 RECOMMENDATION:")
        if overall_speedup > 10:
            print(f"   🏆 Universal Agent shows SIGNIFICANT performance advantage")
        elif overall_speedup > 2:
            print(f"   👍 Universal Agent shows GOOD performance advantage")
        else:
            print(f"   📊 Performance is comparable between systems")


def main():
    """Main function to run the comprehensive comparison."""
    print("🌟 LangSmith Monitoring Comparison Test")
    print("=" * 50)

    # Check dataset
    dataset_path = config.default_dataset_path
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run: uv run python data/download_titanic.py")
        return

    # Initialize and run comparison
    tester = LangSmithComparisonTester()
    results = tester.run_comprehensive_comparison()

    print(f"\n✅ Comparison completed! Check LangSmith dashboard for detailed traces:")
    print(f"   📊 LangSmith URL: https://smith.langchain.com/")
    print(f"   🔍 Search for projects starting with 'multi-agent-use-case-' and 'universal-agent-use-case-'")


if __name__ == "__main__":
    main()