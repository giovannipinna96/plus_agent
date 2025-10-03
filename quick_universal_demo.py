#!/usr/bin/env python3
"""
Quick demonstration of the universal multi-agent statistics system.
Shows how to integrate with any multi-agent framework in just a few lines.
"""

from universal_multiagent_statistics import track_multiagent_session, export_universal_stats, export_universal_report
from datetime import datetime


def quick_demo():
    """Demonstrate the universal statistics system with a simple example."""

    print("🚀 Universal Multi-Agent Statistics - Quick Demo")
    print("=" * 60)

    # Example 1: Simple session tracking
    print("\n📊 Example 1: Basic Session Tracking")
    with track_multiagent_session("Analyze customer data", {"complexity": "simple"}) as tracker:
        # Simulate some agent work
        tracker.log_agent(
            agent_name="DataAnalyzer",
            execution_time=2.1,
            tools_used=["read_csv", "calculate_stats"],
            success=True,
            result_summary="Processed 1000 customer records"
        )

        # Log some tool usage
        tracker.log_tool(
            tool_name="read_csv",
            agent_name="DataAnalyzer",
            parameters={"file": "customers.csv"},
            execution_time=0.8,
            success=True
        )

    print("✅ Session tracked successfully!")

    # Example 2: Multiple sessions
    print("\n📊 Example 2: Multiple Sessions")
    requests = [
        "Generate sales report",
        "Predict customer churn",
        "Optimize pricing strategy"
    ]

    for i, request in enumerate(requests):
        print(f"   Processing: {request}")
        with track_multiagent_session(request, {"type": "business_analysis"}) as tracker:
            # Different agents for different tasks
            if "report" in request:
                tracker.log_agent("ReportAgent", 1.5, ["generate_charts", "export_pdf"], success=True)
            elif "predict" in request:
                tracker.log_agent("MLAgent", 3.2, ["train_model", "evaluate"], success=True)
            elif "optimize" in request:
                tracker.log_agent("OptimizationAgent", 2.8, ["analyze_pricing", "recommend"], success=True)

    print("✅ All sessions completed!")

    # Export comprehensive statistics
    print("\n📈 Exporting Universal Statistics...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    stats_file = export_universal_stats(f"demo_stats_{timestamp}.json")
    report_file = export_universal_report(f"demo_report_{timestamp}.md")

    print(f"✅ Statistics exported to: {stats_file}")
    print(f"✅ Report generated: {report_file}")

    print("\n🎉 Demo completed! Check the generated files for detailed analysis.")


if __name__ == "__main__":
    quick_demo()