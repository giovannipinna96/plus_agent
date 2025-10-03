#!/usr/bin/env python3
"""
Comprehensive Use Cases Test Script for Plus-Agent Multi-Agent System

This script executes all three use cases in sequence and generates
detailed performance reports and system validation.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.orchestrator import MultiAgentOrchestrator
from core.config import config
from core.langsmith_integration import langsmith_logger
from use_cases import UseCaseLibrary


class SystemValidator:
    """Comprehensive system validation and testing framework."""

    def __init__(self):
        self.orchestrator = None
        self.results = {
            "system_info": {},
            "use_case_results": [],
            "performance_metrics": {},
            "validation_summary": {}
        }

    def initialize_system(self) -> bool:
        """Initialize the multi-agent system."""
        print("⚙️  Initializing Plus-Agent Multi-Agent System...")

        try:
            # Setup configuration
            config.setup_langsmith()

            # Initialize orchestrator
            self.orchestrator = MultiAgentOrchestrator()

            # Collect system information
            self.results["system_info"] = {
                "model_name": config.model_name,
                "device": config.device,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "langsmith_enabled": config.langsmith_tracing,
                "default_dataset": config.default_dataset_path,
                "initialization_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            print("✅ System initialized successfully!")
            return True

        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False

    def validate_dataset(self) -> bool:
        """Validate that the default dataset is available."""
        print("📊 Validating dataset availability...")

        if not os.path.exists(config.default_dataset_path):
            print("⚠️  Default dataset not found. Attempting download...")
            try:
                from data.download_titanic import main as download_titanic
                download_titanic()
                print("✅ Dataset downloaded successfully!")
                return True
            except Exception as e:
                print(f"❌ Failed to download dataset: {e}")
                return False
        else:
            print("✅ Dataset found!")
            return True

    def execute_all_use_cases(self) -> List[Dict[str, Any]]:
        """Execute all use cases and collect results."""
        print(f"\n{'='*60}")
        print("🚀 EXECUTING ALL USE CASES")
        print(f"{'='*60}")

        use_cases = UseCaseLibrary.get_all_use_cases()
        use_case_results = []

        for i, use_case in enumerate(use_cases, 1):
            print(f"\n📋 Executing Use Case {i}/{len(use_cases)}: {use_case.name}")

            try:
                result = use_case.execute(self.orchestrator)
                use_case_results.append(result)

                # Log summary
                print(f"   ✅ Completed: {result['success_rate']*100}% success rate")
                print(f"   ⏱️  Duration: {result['total_execution_time']} seconds")
                print(f"   🤖 Agents used: {', '.join(result['agents_used'])}")

            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                use_case_results.append({
                    "name": use_case.name,
                    "complexity": use_case.complexity,
                    "success_rate": 0,
                    "total_execution_time": 0,
                    "error": str(e)
                })

        self.results["use_case_results"] = use_case_results
        return use_case_results

    def calculate_performance_metrics(self, use_case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        print(f"\n📊 Calculating performance metrics...")

        total_time = sum(result.get("total_execution_time", 0) for result in use_case_results)
        overall_success_rate = sum(result.get("success_rate", 0) for result in use_case_results) / len(use_case_results)

        # Collect all agents used
        all_agents = set()
        for result in use_case_results:
            all_agents.update(result.get("agents_used", []))

        # Calculate complexity distribution
        complexity_counts = {}
        for result in use_case_results:
            complexity = result.get("complexity", "unknown")
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

        # Agent usage statistics
        agent_usage = {}
        for result in use_case_results:
            for agent in result.get("agents_used", []):
                agent_usage[agent] = agent_usage.get(agent, 0) + 1

        metrics = {
            "total_execution_time": round(total_time, 2),
            "average_execution_time": round(total_time / len(use_case_results), 2),
            "overall_success_rate": round(overall_success_rate, 2),
            "total_use_cases": len(use_case_results),
            "successful_use_cases": sum(1 for result in use_case_results if result.get("success_rate", 0) > 0.5),
            "agents_tested": list(all_agents),
            "total_agents_tested": len(all_agents),
            "complexity_distribution": complexity_counts,
            "agent_usage_frequency": agent_usage,
            "most_used_agent": max(agent_usage, key=agent_usage.get) if agent_usage else None
        }

        self.results["performance_metrics"] = metrics
        return metrics

    def generate_validation_summary(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate system validation summary and recommendations."""
        print(f"\n🎯 Generating validation summary...")

        overall_success = performance_metrics["overall_success_rate"]
        agent_coverage = performance_metrics["total_agents_tested"]

        # Determine system status
        if overall_success >= 0.8 and agent_coverage >= 4:
            status = "EXCELLENT"
            recommendation = "System is production-ready with excellent performance across all use cases."
        elif overall_success >= 0.6 and agent_coverage >= 3:
            status = "GOOD"
            recommendation = "System performs well with minor issues. Review failed tests before production."
        elif overall_success >= 0.4:
            status = "NEEDS_IMPROVEMENT"
            recommendation = "System has significant issues that need to be addressed."
        else:
            status = "CRITICAL"
            recommendation = "System has critical failures and requires immediate attention."

        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []

        if overall_success >= 0.7:
            strengths.append("High overall success rate")
        else:
            weaknesses.append("Low overall success rate")

        if agent_coverage >= 4:
            strengths.append("Good agent coverage")
        else:
            weaknesses.append("Limited agent coverage")

        if performance_metrics["average_execution_time"] <= 30:
            strengths.append("Fast execution times")
        else:
            weaknesses.append("Slow execution times")

        summary = {
            "status": status,
            "overall_score": round(overall_success * 100, 1),
            "recommendation": recommendation,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "agent_performance": {
                agent: f"{count} use cases"
                for agent, count in performance_metrics["agent_usage_frequency"].items()
            },
            "complexity_handling": performance_metrics["complexity_distribution"],
            "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        self.results["validation_summary"] = summary
        return summary

    def print_comprehensive_report(self):
        """Print a comprehensive validation report."""
        print(f"\n{'='*80}")
        print("📋 COMPREHENSIVE SYSTEM VALIDATION REPORT")
        print(f"{'='*80}")

        # System Information
        print(f"\n🖥️  SYSTEM CONFIGURATION:")
        sys_info = self.results["system_info"]
        print(f"   Model: {sys_info.get('model_name', 'Unknown')}")
        print(f"   Device: {sys_info.get('device', 'Unknown')}")
        print(f"   LangSmith: {'Enabled' if sys_info.get('langsmith_enabled') else 'Disabled'}")
        print(f"   Initialized: {sys_info.get('initialization_time', 'Unknown')}")

        # Performance Metrics
        print(f"\n📊 PERFORMANCE METRICS:")
        metrics = self.results["performance_metrics"]
        print(f"   Overall Success Rate: {metrics.get('overall_success_rate', 0)*100}%")
        print(f"   Total Execution Time: {metrics.get('total_execution_time', 0)} seconds")
        print(f"   Average per Use Case: {metrics.get('average_execution_time', 0)} seconds")
        print(f"   Agents Tested: {metrics.get('total_agents_tested', 0)}")
        print(f"   Most Used Agent: {metrics.get('most_used_agent', 'None')}")

        # Use Case Results
        print(f"\n🎯 USE CASE RESULTS:")
        for i, result in enumerate(self.results["use_case_results"], 1):
            status_emoji = "✅" if result.get("success_rate", 0) >= 0.5 else "❌"
            print(f"   {status_emoji} Use Case {i}: {result['name']}")
            print(f"      Success Rate: {result.get('success_rate', 0)*100}%")
            print(f"      Execution Time: {result.get('total_execution_time', 0)} seconds")
            print(f"      Complexity: {result.get('complexity', 'Unknown')}")

        # Validation Summary
        print(f"\n🎖️  VALIDATION SUMMARY:")
        summary = self.results["validation_summary"]
        print(f"   Status: {summary.get('status', 'Unknown')}")
        print(f"   Overall Score: {summary.get('overall_score', 0)}/100")
        print(f"   Recommendation: {summary.get('recommendation', 'No recommendation')}")

        if summary.get("strengths"):
            print(f"   Strengths: {', '.join(summary['strengths'])}")
        if summary.get("weaknesses"):
            print(f"   Weaknesses: {', '.join(summary['weaknesses'])}")

    def save_results(self, filename: str = "comprehensive_test_results.json"):
        """Save test results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Comprehensive test results saved to: {filename}")


def main():
    """Main function to execute comprehensive system validation."""
    print("🧪 Plus-Agent Comprehensive System Validation")
    print("=" * 50)

    # Initialize validator
    validator = SystemValidator()

    # Step 1: Initialize system
    if not validator.initialize_system():
        print("❌ System initialization failed. Exiting.")
        return

    # Step 2: Validate dataset
    if not validator.validate_dataset():
        print("❌ Dataset validation failed. Exiting.")
        return

    # Step 3: Execute all use cases
    use_case_results = validator.execute_all_use_cases()

    # Step 4: Calculate performance metrics
    performance_metrics = validator.calculate_performance_metrics(use_case_results)

    # Step 5: Generate validation summary
    validation_summary = validator.generate_validation_summary(performance_metrics)

    # Step 6: Print comprehensive report
    validator.print_comprehensive_report()

    # Step 7: Save results
    validator.save_results()

    # Final recommendations
    print(f"\n{'='*60}")
    print("🎯 FINAL RECOMMENDATIONS")
    print(f"{'='*60}")

    status = validation_summary.get("status", "UNKNOWN")
    if status == "EXCELLENT":
        print("🎉 Congratulations! Your Plus-Agent system is performing excellently.")
        print("✅ The system is ready for production deployment.")
        print("🚀 Consider scaling up for larger datasets or more complex workflows.")
    elif status == "GOOD":
        print("👍 Your Plus-Agent system is performing well overall.")
        print("🔧 Review any failed test cases and optimize as needed.")
        print("📈 Consider fine-tuning model parameters for better performance.")
    elif status == "NEEDS_IMPROVEMENT":
        print("⚠️  Your Plus-Agent system needs improvement.")
        print("🔍 Investigate failed agents and use cases.")
        print("⚙️  Check system configuration and dependencies.")
    else:
        print("🚨 Critical issues detected with your Plus-Agent system.")
        print("❌ System requires immediate attention before use.")
        print("🆘 Consider reviewing logs and system configuration.")

    print(f"\nValidation completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()