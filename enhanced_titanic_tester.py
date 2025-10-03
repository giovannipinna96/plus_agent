"""
Enhanced Titanic Multi-Agent System Tester with Detailed Statistics Collection.
"""

import json
import sys
import os
from typing import Dict, Any, List
import time
from datetime import datetime
import uuid

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.enhanced_orchestrator import EnhancedMultiAgentOrchestrator
from core.tool_tracking_wrapper import global_tracker, global_stats_collector


class EnhancedTitanicTester:
    """Enhanced tester with detailed statistics collection."""

    def __init__(self, data_file_path: str = "data/titanic.csv"):
        self.data_file_path = data_file_path
        self.orchestrator = EnhancedMultiAgentOrchestrator()
        self.detailed_results = {}

    def test_single_question_enhanced(self, question_id: str, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single question with enhanced tracking and statistics."""

        print(f"\\n{'='*70}")
        print(f"🔬 ENHANCED Testing - Question {question_id}")
        print(f"Text: {question_data['text']}")
        print(f"Expected Agents: {question_data.get('expected_agents', [])}")
        print(f"Expected Tools: {question_data['tools_sequence']}")
        print(f"{'='*70}")

        # Generate unique session ID
        session_id = f"{question_id}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        result = {
            "question_id": question_id,
            "session_id": session_id,
            "question_text": question_data["text"],
            "complexity": question_data.get("complexity", "unknown"),
            "expected_agents": question_data.get("expected_agents", []),
            "expected_tools": question_data["tools_sequence"],

            # Enhanced tracking fields
            "actual_agents_used": [],
            "actual_tools_used": [],
            "agent_execution_details": {},
            "tool_execution_details": {},
            "execution_timeline": [],
            "performance_metrics": {},

            "execution_time": 0,
            "success": False,
            "response": None,
            "detailed_response": None,
            "error_message": None
        }

        try:
            print(f"\\n🚀 Executing with enhanced multi-agent system...")
            print(f"📊 Session ID: {session_id}")

            # Execute with enhanced orchestrator
            detailed_result = self.orchestrator.run_enhanced_analysis(
                user_prompt=question_data["text"],
                file_path=self.data_file_path,
                session_id=session_id
            )

            execution_time = time.time() - start_time

            if detailed_result and detailed_result.get("status") == "success":
                result["success"] = True
                result["execution_time"] = execution_time
                result["detailed_response"] = detailed_result

                # Extract basic response
                messages = detailed_result.get("messages", [])
                if messages:
                    result["response"] = " ".join(messages)

                # Extract enhanced tracking information
                execution_metadata = detailed_result.get("execution_metadata", {})
                tools_used = detailed_result.get("tools_used", [])
                agents_activated = detailed_result.get("agents_activated", [])
                tracking_session_data = detailed_result.get("tracking_session_data", {})

                # Process agent information
                result["actual_agents_used"] = [agent.get("agent", "") for agent in agents_activated]

                # Process tool information
                all_tools = []
                for tool_group in tools_used:
                    all_tools.extend(tool_group.get("tools", []))
                result["actual_tools_used"] = list(set(all_tools))  # Remove duplicates

                # Detailed agent execution breakdown
                agent_details = {}
                for agent_info in agents_activated:
                    agent_name = agent_info.get("agent", "unknown")
                    agent_details[agent_name] = {
                        "execution_time": agent_info.get("time", 0),
                        "timestamp": agent_info.get("timestamp", 0),
                        "iterations": 1  # This could be enhanced further
                    }
                result["agent_execution_details"] = agent_details

                # Detailed tool execution breakdown
                tool_details = {}
                for tool_group in tools_used:
                    agent_name = tool_group.get("agent", "unknown")
                    for tool_name in tool_group.get("tools", []):
                        if tool_name not in tool_details:
                            tool_details[tool_name] = {
                                "used_by_agents": [],
                                "total_execution_time": tool_group.get("execution_time", 0),
                                "call_count": 1
                            }
                        tool_details[tool_name]["used_by_agents"].append(agent_name)
                result["tool_execution_details"] = tool_details

                # Performance metrics
                summary_stats = detailed_result.get("summary_stats", {})
                result["performance_metrics"] = {
                    "total_workflow_time": detailed_result.get("total_execution_time", execution_time),
                    "agent_count": summary_stats.get("total_agents_used", 0),
                    "tool_count": summary_stats.get("total_tools_used", 0),
                    "steps_completed": summary_stats.get("total_steps_completed", 0),
                    "average_time_per_agent": 0,
                    "average_time_per_tool": 0
                }

                # Calculate averages
                if result["performance_metrics"]["agent_count"] > 0:
                    result["performance_metrics"]["average_time_per_agent"] = (
                        result["performance_metrics"]["total_workflow_time"] /
                        result["performance_metrics"]["agent_count"]
                    )

                if result["performance_metrics"]["tool_count"] > 0:
                    result["performance_metrics"]["average_time_per_tool"] = (
                        result["performance_metrics"]["total_workflow_time"] /
                        result["performance_metrics"]["tool_count"]
                    )

                # Execution timeline from tracking session
                if tracking_session_data:
                    timeline = []
                    for tool_call in tracking_session_data.get("tool_calls", []):
                        timeline.append({
                            "type": "tool_call",
                            "timestamp": tool_call.get("timestamp", 0),
                            "name": tool_call.get("tool_name", "unknown"),
                            "agent": tool_call.get("called_by_agent", "unknown"),
                            "duration": tool_call.get("execution_time", 0),
                            "success": tool_call.get("success", False)
                        })

                    for agent_call in tracking_session_data.get("agent_calls", []):
                        timeline.append({
                            "type": "agent_activation",
                            "timestamp": agent_call.get("timestamp", 0),
                            "name": agent_call.get("agent_name", "unknown"),
                            "duration": agent_call.get("execution_time", 0),
                            "tools_used": agent_call.get("tools_used", [])
                        })

                    # Sort timeline by timestamp
                    timeline.sort(key=lambda x: x.get("timestamp", 0))
                    result["execution_timeline"] = timeline

                    # Add session data to global collector
                    global_stats_collector.add_session(tracking_session_data)

                print(f"\\n✅ Question {question_id} completed successfully")
                print(f"⏱️  Total execution time: {execution_time:.2f} seconds")
                print(f"🤖 Agents used: {result['actual_agents_used']}")
                print(f"🔧 Tools used: {result['actual_tools_used']}")
                print(f"📊 Performance: {result['performance_metrics']['agent_count']} agents, {result['performance_metrics']['tool_count']} tools")

            else:
                result["error_message"] = detailed_result.get("error", "Unknown error")
                print(f"\\n❌ Question {question_id} failed: {result['error_message']}")

        except Exception as e:
            execution_time = time.time() - start_time
            result["error_message"] = str(e)
            result["execution_time"] = execution_time
            print(f"\\n❌ Question {question_id} failed with exception: {str(e)}")
            print(f"⏱️  Execution time: {execution_time:.2f} seconds")

        return result

    def calculate_accuracy_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed accuracy metrics comparing expected vs actual."""

        expected_tools = set(result["expected_tools"])
        actual_tools = set(result["actual_tools_used"])
        expected_agents = set(result["expected_agents"])
        actual_agents = set(result["actual_agents_used"])

        # Tool accuracy metrics
        tool_intersection = expected_tools & actual_tools
        tool_accuracy = len(tool_intersection) / len(expected_tools) if expected_tools else 1.0
        tool_precision = len(tool_intersection) / len(actual_tools) if actual_tools else 0.0
        tool_recall = len(tool_intersection) / len(expected_tools) if expected_tools else 0.0
        tool_f1 = 2 * (tool_precision * tool_recall) / (tool_precision + tool_recall) if (tool_precision + tool_recall) > 0 else 0.0

        # Agent accuracy metrics
        agent_intersection = expected_agents & actual_agents
        agent_accuracy = len(agent_intersection) / len(expected_agents) if expected_agents else 1.0
        agent_precision = len(agent_intersection) / len(actual_agents) if actual_agents else 0.0
        agent_recall = len(agent_intersection) / len(expected_agents) if expected_agents else 0.0
        agent_f1 = 2 * (agent_precision * agent_recall) / (agent_precision + agent_recall) if (agent_precision + agent_recall) > 0 else 0.0

        return {
            "tool_metrics": {
                "accuracy": tool_accuracy,
                "precision": tool_precision,
                "recall": tool_recall,
                "f1_score": tool_f1,
                "tools_matched": list(tool_intersection),
                "extra_tools": list(actual_tools - expected_tools),
                "missing_tools": list(expected_tools - actual_tools)
            },
            "agent_metrics": {
                "accuracy": agent_accuracy,
                "precision": agent_precision,
                "recall": agent_recall,
                "f1_score": agent_f1,
                "agents_matched": list(agent_intersection),
                "extra_agents": list(actual_agents - expected_agents),
                "missing_agents": list(expected_agents - actual_agents)
            },
            "overall_score": (tool_f1 + agent_f1) / 2
        }

    def test_all_questions_enhanced(self, questions_file: str = "titanic_questions.json") -> Dict[str, Any]:
        """Test all questions with enhanced tracking and detailed statistics."""

        print("🚢 Enhanced Titanic Multi-Agent System Tester")
        print("=" * 50)
        print(f"Loading questions from: {questions_file}")

        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        except FileNotFoundError:
            print(f"❌ Questions file {questions_file} not found!")
            return {}

        enhanced_results = {}
        total_questions = len(questions)
        successful_tests = 0
        total_execution_time = 0

        print(f"\\n🎯 Testing {total_questions} questions with enhanced tracking...")

        for i, (question_id, question_data) in enumerate(questions.items(), 1):
            print(f"\\n[{i}/{total_questions}] Processing {question_id}...")

            try:
                # Test the question
                result = self.test_single_question_enhanced(question_id, question_data)

                # Calculate accuracy metrics
                accuracy_metrics = self.calculate_accuracy_metrics(result)
                result["accuracy_metrics"] = accuracy_metrics

                enhanced_results[question_id] = result

                if result["success"]:
                    successful_tests += 1
                    total_execution_time += result["execution_time"]

            except Exception as e:
                print(f"\\n❌ Error testing {question_id}: {str(e)}")
                enhanced_results[question_id] = {
                    "question_id": question_id,
                    "success": False,
                    "error_message": str(e)
                }

        # Calculate overall statistics
        overall_stats = {
            "test_summary": {
                "total_questions": total_questions,
                "successful_tests": successful_tests,
                "failed_tests": total_questions - successful_tests,
                "success_rate": successful_tests / total_questions * 100 if total_questions > 0 else 0,
                "average_execution_time": total_execution_time / successful_tests if successful_tests > 0 else 0,
                "total_execution_time": total_execution_time
            },
            "detailed_results": enhanced_results
        }

        # Add global statistics
        tool_stats = global_stats_collector.get_tool_usage_stats()
        agent_stats = global_stats_collector.get_agent_usage_stats()

        overall_stats["global_tool_statistics"] = tool_stats
        overall_stats["global_agent_statistics"] = agent_stats

        # Save enhanced results
        results_file = f"enhanced_titanic_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, indent=2, ensure_ascii=False, default=str)

        print(f"\\n📊 Enhanced test results saved to: {results_file}")

        # Export detailed statistics
        stats_file = global_stats_collector.export_detailed_statistics()
        if stats_file:
            print(f"📈 Detailed statistics exported to: {stats_file}")

        return overall_stats

    def print_enhanced_summary(self, results: Dict[str, Any]):
        """Print an enhanced summary of test results."""

        test_summary = results.get("test_summary", {})
        tool_stats = results.get("global_tool_statistics", {})
        agent_stats = results.get("global_agent_statistics", {})

        print(f"\\n{'='*70}")
        print("🔬 ENHANCED MULTI-AGENT SYSTEM ANALYSIS REPORT")
        print(f"{'='*70}")

        print(f"\\n📊 Test Execution Summary:")
        print(f"   Total Questions: {test_summary.get('total_questions', 0)}")
        print(f"   Successfully Executed: {test_summary.get('successful_tests', 0)}")
        print(f"   Failed: {test_summary.get('failed_tests', 0)}")
        print(f"   Success Rate: {test_summary.get('success_rate', 0):.1f}%")
        print(f"   Average Execution Time: {test_summary.get('average_execution_time', 0):.2f} seconds")
        print(f"   Total Execution Time: {test_summary.get('total_execution_time', 0):.2f} seconds")

        print(f"\\n🔧 Tool Usage Analysis:")
        print(f"   Unique Tools Used: {tool_stats.get('total_unique_tools', 0)}")
        print(f"   Total Tool Calls: {tool_stats.get('total_tool_calls', 0)}")

        if tool_stats.get("tool_details"):
            print(f"\\n   Top Tools by Usage:")
            sorted_tools = sorted(
                tool_stats["tool_details"].items(),
                key=lambda x: x[1]["total_calls"],
                reverse=True
            )
            for i, (tool_name, stats) in enumerate(sorted_tools[:5], 1):
                print(f"     {i}. {tool_name}: {stats['total_calls']} calls, "
                      f"{stats['average_time']:.3f}s avg, {stats['success_rate']:.1%} success")

        print(f"\\n🤖 Agent Usage Analysis:")
        print(f"   Unique Agents Used: {agent_stats.get('total_unique_agents', 0)}")
        print(f"   Total Agent Activations: {agent_stats.get('total_agent_activations', 0)}")

        if agent_stats.get("agent_details"):
            print(f"\\n   Agent Performance:")
            for agent_name, stats in agent_stats["agent_details"].items():
                print(f"     {agent_name}: {stats['total_activations']} activations, "
                      f"{stats['average_time']:.2f}s avg, {stats['unique_tools_count']} unique tools")

        print(f"\\n🎯 Accuracy Analysis:")
        detailed_results = results.get("detailed_results", {})
        total_accuracy_scores = []
        tool_accuracy_scores = []
        agent_accuracy_scores = []

        for question_id, result in detailed_results.items():
            if result.get("success") and "accuracy_metrics" in result:
                metrics = result["accuracy_metrics"]
                tool_accuracy_scores.append(metrics["tool_metrics"]["f1_score"])
                agent_accuracy_scores.append(metrics["agent_metrics"]["f1_score"])
                total_accuracy_scores.append(metrics["overall_score"])

        if total_accuracy_scores:
            avg_tool_f1 = sum(tool_accuracy_scores) / len(tool_accuracy_scores)
            avg_agent_f1 = sum(agent_accuracy_scores) / len(agent_accuracy_scores)
            avg_overall_score = sum(total_accuracy_scores) / len(total_accuracy_scores)

            print(f"   Average Tool Selection F1: {avg_tool_f1:.3f}")
            print(f"   Average Agent Selection F1: {avg_agent_f1:.3f}")
            print(f"   Average Overall Accuracy: {avg_overall_score:.3f}")

        print(f"\\n{'='*70}")


def main():
    """Main function to run enhanced testing."""

    tester = EnhancedTitanicTester()

    # Check if data file exists
    if not os.path.exists(tester.data_file_path):
        print(f"❌ Data file not found: {tester.data_file_path}")
        print("Please ensure the Titanic dataset is available at the specified path.")
        return

    # Run all enhanced tests
    results = tester.test_all_questions_enhanced()

    if results:
        # Print enhanced summary
        tester.print_enhanced_summary(results)

        print(f"\\n🎉 Enhanced testing completed!")
        print(f"📋 Check the generated files for detailed statistics and analysis.")

    return results


if __name__ == "__main__":
    main()