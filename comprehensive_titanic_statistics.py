"""
Comprehensive Titanic Statistics Generator
Creates a detailed JSON file with all statistics, comparisons, and analysis results.
"""

import json
import sys
import os
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ComprehensiveTitanicStatsGenerator:
    """Generate comprehensive statistics from all test results."""

    def __init__(self):
        self.questions = {}
        self.original_results = {}
        self.enhanced_results = {}
        self.detailed_statistics = {}

    def load_all_data(self):
        """Load all available data files."""

        # Load questions
        try:
            with open("titanic_questions.json", 'r', encoding='utf-8') as f:
                self.questions = json.load(f)
            print(f"✅ Loaded {len(self.questions)} questions")
        except FileNotFoundError:
            print("❌ Questions file not found")

        # Load original multi-agent results
        original_files = [f for f in os.listdir('.') if f.startswith('titanic_multiagent_test_results_')]
        if original_files:
            latest_original = max(original_files)
            try:
                with open(latest_original, 'r', encoding='utf-8') as f:
                    self.original_results = json.load(f)
                print(f"✅ Loaded original results from {latest_original}")
            except FileNotFoundError:
                print(f"❌ Could not load {latest_original}")

        # Load enhanced results
        enhanced_files = [f for f in os.listdir('.') if f.startswith('enhanced_titanic_test_results_')]
        if enhanced_files:
            latest_enhanced = max(enhanced_files)
            try:
                with open(latest_enhanced, 'r', encoding='utf-8') as f:
                    self.enhanced_results = json.load(f)
                print(f"✅ Loaded enhanced results from {latest_enhanced}")
            except FileNotFoundError:
                print(f"❌ Could not load {latest_enhanced}")

        # Load detailed statistics
        stats_files = [f for f in os.listdir('.') if f.startswith('detailed_statistics_')]
        if stats_files:
            latest_stats = max(stats_files)
            try:
                with open(latest_stats, 'r', encoding='utf-8') as f:
                    self.detailed_statistics = json.load(f)
                print(f"✅ Loaded detailed statistics from {latest_stats}")
            except FileNotFoundError:
                print(f"❌ Could not load {latest_stats}")

    def generate_comprehensive_statistics(self) -> Dict[str, Any]:
        """Generate the comprehensive statistics JSON."""

        stats = {
            "metadata": {
                "generated_timestamp": datetime.now().isoformat(),
                "generator_version": "1.0.0",
                "total_questions_analyzed": len(self.questions),
                "analysis_scope": "Titanic Dataset Multi-Agent Tool Selection Validation"
            },

            "questions_overview": {},
            "performance_analysis": {},
            "tool_selection_analysis": {},
            "agent_selection_analysis": {},
            "system_comparison": {},
            "accuracy_metrics": {},
            "detailed_breakdowns": {}
        }

        # Process each question
        for question_id, question_data in self.questions.items():
            question_stats = self.analyze_single_question(question_id, question_data)
            stats["questions_overview"][question_id] = question_stats

        # Generate performance analysis
        stats["performance_analysis"] = self.generate_performance_analysis()

        # Generate tool selection analysis
        stats["tool_selection_analysis"] = self.generate_tool_selection_analysis()

        # Generate agent selection analysis
        stats["agent_selection_analysis"] = self.generate_agent_selection_analysis()

        # Generate system comparison
        stats["system_comparison"] = self.generate_system_comparison()

        # Generate accuracy metrics
        stats["accuracy_metrics"] = self.generate_accuracy_metrics()

        # Generate detailed breakdowns
        stats["detailed_breakdowns"] = self.generate_detailed_breakdowns()

        return stats

    def analyze_single_question(self, question_id: str, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single question across all test results."""

        analysis = {
            "question_text": question_data.get("text", ""),
            "complexity": question_data.get("complexity", "unknown"),
            "expected_tools": question_data.get("tools_sequence", []),
            "expected_agents": question_data.get("expected_agents", []),

            "original_test_results": {},
            "enhanced_test_results": {},

            "comparison": {
                "execution_time_difference": 0,
                "success_comparison": "",
                "tool_detection_improved": False,
                "agent_detection_improved": False
            }
        }

        # Extract original results
        if question_id in self.original_results:
            orig_result = self.original_results[question_id]
            analysis["original_test_results"] = {
                "execution_time": orig_result.get("execution_time", 0),
                "success": orig_result.get("success", False),
                "actual_agents": orig_result.get("actual_agents_used", []),
                "actual_tools": orig_result.get("actual_tools_used", []),
                "response": orig_result.get("response", "")
            }

        # Extract enhanced results
        enhanced_data = self.enhanced_results.get("detailed_results", {})
        if question_id in enhanced_data:
            enhanced_result = enhanced_data[question_id]
            analysis["enhanced_test_results"] = {
                "execution_time": enhanced_result.get("execution_time", 0),
                "success": enhanced_result.get("success", False),
                "actual_agents": enhanced_result.get("actual_agents_used", []),
                "actual_tools": enhanced_result.get("actual_tools_used", []),
                "agent_execution_details": enhanced_result.get("agent_execution_details", {}),
                "tool_execution_details": enhanced_result.get("tool_execution_details", {}),
                "performance_metrics": enhanced_result.get("performance_metrics", {}),
                "accuracy_metrics": enhanced_result.get("accuracy_metrics", {}),
                "execution_timeline": enhanced_result.get("execution_timeline", [])
            }

        # Generate comparison
        if analysis["original_test_results"] and analysis["enhanced_test_results"]:
            orig_time = analysis["original_test_results"]["execution_time"]
            enhanced_time = analysis["enhanced_test_results"]["execution_time"]
            analysis["comparison"]["execution_time_difference"] = enhanced_time - orig_time

            orig_success = analysis["original_test_results"]["success"]
            enhanced_success = analysis["enhanced_test_results"]["success"]

            if orig_success and enhanced_success:
                analysis["comparison"]["success_comparison"] = "Both successful"
            elif not orig_success and enhanced_success:
                analysis["comparison"]["success_comparison"] = "Enhanced improved"
            elif orig_success and not enhanced_success:
                analysis["comparison"]["success_comparison"] = "Enhanced degraded"
            else:
                analysis["comparison"]["success_comparison"] = "Both failed"

            # Tool and agent detection comparison
            orig_tools = len(analysis["original_test_results"]["actual_tools"])
            enhanced_tools = len(analysis["enhanced_test_results"]["actual_tools"])
            analysis["comparison"]["tool_detection_improved"] = enhanced_tools > orig_tools

            orig_agents = len(analysis["original_test_results"]["actual_agents"])
            enhanced_agents = len(analysis["enhanced_test_results"]["actual_agents"])
            analysis["comparison"]["agent_detection_improved"] = enhanced_agents > orig_agents

        return analysis

    def generate_performance_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis."""

        analysis = {
            "execution_times": {
                "by_complexity": {},
                "by_question_type": {},
                "overall_stats": {}
            },
            "success_rates": {
                "overall": {},
                "by_complexity": {},
                "by_expected_agent_count": {}
            },
            "resource_usage": {
                "agent_utilization": {},
                "tool_utilization": {},
                "efficiency_metrics": {}
            }
        }

        # Analyze execution times by complexity
        complexity_times = {"simple": [], "medium": [], "complex": []}

        enhanced_detailed = self.enhanced_results.get("detailed_results", {})
        for question_id, result in enhanced_detailed.items():
            complexity = result.get("complexity", "unknown")
            exec_time = result.get("execution_time", 0)

            if complexity in complexity_times:
                complexity_times[complexity].append(exec_time)

        for complexity, times in complexity_times.items():
            if times:
                analysis["execution_times"]["by_complexity"][complexity] = {
                    "count": len(times),
                    "average": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times)
                }

        # Overall execution stats
        all_times = []
        all_successes = []

        for result in enhanced_detailed.values():
            all_times.append(result.get("execution_time", 0))
            all_successes.append(result.get("success", False))

        if all_times:
            analysis["execution_times"]["overall_stats"] = {
                "total_questions": len(all_times),
                "average_time": sum(all_times) / len(all_times),
                "total_time": sum(all_times),
                "min_time": min(all_times),
                "max_time": max(all_times)
            }

        if all_successes:
            success_count = sum(all_successes)
            analysis["success_rates"]["overall"] = {
                "total_tests": len(all_successes),
                "successful_tests": success_count,
                "success_rate": success_count / len(all_successes) * 100,
                "failure_rate": (len(all_successes) - success_count) / len(all_successes) * 100
            }

        return analysis

    def generate_tool_selection_analysis(self) -> Dict[str, Any]:
        """Generate tool selection accuracy analysis."""

        analysis = {
            "expected_vs_actual": {},
            "tool_categories": {
                "data_reading": {"expected": 0, "actual": 0, "accuracy": 0},
                "data_operations": {"expected": 0, "actual": 0, "accuracy": 0},
                "data_manipulation": {"expected": 0, "actual": 0, "accuracy": 0},
                "machine_learning": {"expected": 0, "actual": 0, "accuracy": 0},
                "titanic_specific": {"expected": 0, "actual": 0, "accuracy": 0}
            },
            "selection_accuracy": {
                "per_question": {},
                "overall_metrics": {}
            },
            "common_deviations": []
        }

        tool_categories = {
            "data_reading": ["read_csv_file", "preview_data", "get_column_info", "get_data_summary"],
            "data_operations": ["filter_data", "aggregate_data", "perform_math_operations"],
            "data_manipulation": ["create_dummy_variables", "handle_missing_values", "modify_column_values"],
            "machine_learning": ["train_random_forest_model", "train_svm_model", "train_regression_model", "evaluate_model"],
            "titanic_specific": ["calculate_survival_rate_by_group", "get_statistics_for_profile",
                               "calculate_survival_probability_by_features", "get_fare_estimate_by_profile",
                               "count_passengers_by_criteria"]
        }

        enhanced_detailed = self.enhanced_results.get("detailed_results", {})

        for question_id, result in enhanced_detailed.items():
            expected_tools = set(result.get("expected_tools", []))
            actual_tools = set(result.get("actual_tools_used", []))

            # Calculate accuracy for this question
            if expected_tools:
                intersection = expected_tools & actual_tools
                precision = len(intersection) / len(actual_tools) if actual_tools else 0
                recall = len(intersection) / len(expected_tools)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                analysis["selection_accuracy"]["per_question"][question_id] = {
                    "expected_count": len(expected_tools),
                    "actual_count": len(actual_tools),
                    "matched_count": len(intersection),
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "matched_tools": list(intersection),
                    "missing_tools": list(expected_tools - actual_tools),
                    "extra_tools": list(actual_tools - expected_tools)
                }

            # Categorize tools
            for tool in expected_tools:
                for category, tools in tool_categories.items():
                    if tool in tools:
                        analysis["tool_categories"][category]["expected"] += 1

            for tool in actual_tools:
                for category, tools in tool_categories.items():
                    if tool in tools:
                        analysis["tool_categories"][category]["actual"] += 1

        # Calculate overall accuracy metrics
        all_f1_scores = []
        all_precisions = []
        all_recalls = []

        for question_metrics in analysis["selection_accuracy"]["per_question"].values():
            all_f1_scores.append(question_metrics["f1_score"])
            all_precisions.append(question_metrics["precision"])
            all_recalls.append(question_metrics["recall"])

        if all_f1_scores:
            analysis["selection_accuracy"]["overall_metrics"] = {
                "average_f1_score": sum(all_f1_scores) / len(all_f1_scores),
                "average_precision": sum(all_precisions) / len(all_precisions),
                "average_recall": sum(all_recalls) / len(all_recalls),
                "questions_with_perfect_recall": sum(1 for r in all_recalls if r == 1.0),
                "questions_with_perfect_precision": sum(1 for p in all_precisions if p == 1.0)
            }

        # Calculate category accuracy
        for category, counts in analysis["tool_categories"].items():
            if counts["expected"] > 0:
                counts["accuracy"] = min(counts["actual"] / counts["expected"], 1.0)

        return analysis

    def generate_agent_selection_analysis(self) -> Dict[str, Any]:
        """Generate agent selection accuracy analysis."""

        analysis = {
            "expected_vs_actual": {},
            "agent_utilization": {},
            "selection_accuracy": {
                "per_question": {},
                "overall_metrics": {}
            },
            "workflow_patterns": {}
        }

        enhanced_detailed = self.enhanced_results.get("detailed_results", {})

        agent_usage_count = {}

        for question_id, result in enhanced_detailed.items():
            expected_agents = set(result.get("expected_agents", []))
            actual_agents = set(result.get("actual_agents_used", []))

            # Track agent usage
            for agent in actual_agents:
                agent_usage_count[agent] = agent_usage_count.get(agent, 0) + 1

            # Calculate accuracy for this question
            if expected_agents:
                intersection = expected_agents & actual_agents
                precision = len(intersection) / len(actual_agents) if actual_agents else 0
                recall = len(intersection) / len(expected_agents)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                analysis["selection_accuracy"]["per_question"][question_id] = {
                    "expected_count": len(expected_agents),
                    "actual_count": len(actual_agents),
                    "matched_count": len(intersection),
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "matched_agents": list(intersection),
                    "missing_agents": list(expected_agents - actual_agents),
                    "extra_agents": list(actual_agents - expected_agents)
                }

        # Agent utilization analysis
        total_activations = sum(agent_usage_count.values())
        for agent, count in agent_usage_count.items():
            analysis["agent_utilization"][agent] = {
                "activation_count": count,
                "utilization_percentage": count / total_activations * 100 if total_activations > 0 else 0
            }

        # Calculate overall accuracy metrics
        all_f1_scores = []
        all_precisions = []
        all_recalls = []

        for question_metrics in analysis["selection_accuracy"]["per_question"].values():
            all_f1_scores.append(question_metrics["f1_score"])
            all_precisions.append(question_metrics["precision"])
            all_recalls.append(question_metrics["recall"])

        if all_f1_scores:
            analysis["selection_accuracy"]["overall_metrics"] = {
                "average_f1_score": sum(all_f1_scores) / len(all_f1_scores),
                "average_precision": sum(all_precisions) / len(all_precisions),
                "average_recall": sum(all_recalls) / len(all_recalls),
                "questions_with_perfect_recall": sum(1 for r in all_recalls if r == 1.0),
                "questions_with_perfect_precision": sum(1 for p in all_precisions if p == 1.0)
            }

        return analysis

    def generate_system_comparison(self) -> Dict[str, Any]:
        """Compare original vs enhanced system performance."""

        comparison = {
            "performance_comparison": {},
            "detection_improvement": {},
            "system_reliability": {},
            "enhancement_impact": {}
        }

        # Performance comparison
        original_times = []
        enhanced_times = []

        for question_id in self.questions.keys():
            if question_id in self.original_results:
                original_times.append(self.original_results[question_id].get("execution_time", 0))

            enhanced_detailed = self.enhanced_results.get("detailed_results", {})
            if question_id in enhanced_detailed:
                enhanced_times.append(enhanced_detailed[question_id].get("execution_time", 0))

        if original_times and enhanced_times and len(original_times) == len(enhanced_times):
            comparison["performance_comparison"] = {
                "original_average_time": sum(original_times) / len(original_times),
                "enhanced_average_time": sum(enhanced_times) / len(enhanced_times),
                "time_difference": sum(enhanced_times) / len(enhanced_times) - sum(original_times) / len(original_times),
                "performance_change_percentage": ((sum(enhanced_times) / len(enhanced_times)) - (sum(original_times) / len(original_times))) / (sum(original_times) / len(original_times)) * 100 if original_times else 0
            }

        # Detection improvement analysis
        tool_detection_improvements = 0
        agent_detection_improvements = 0

        for question_id in self.questions.keys():
            orig_tools = len(self.original_results.get(question_id, {}).get("actual_tools_used", []))
            enhanced_tools = len(self.enhanced_results.get("detailed_results", {}).get(question_id, {}).get("actual_tools_used", []))

            if enhanced_tools > orig_tools:
                tool_detection_improvements += 1

            orig_agents = len(self.original_results.get(question_id, {}).get("actual_agents_used", []))
            enhanced_agents = len(self.enhanced_results.get("detailed_results", {}).get(question_id, {}).get("actual_agents_used", []))

            if enhanced_agents > orig_agents:
                agent_detection_improvements += 1

        total_questions = len(self.questions)
        comparison["detection_improvement"] = {
            "tool_detection_improved_count": tool_detection_improvements,
            "agent_detection_improved_count": agent_detection_improvements,
            "tool_detection_improvement_rate": tool_detection_improvements / total_questions * 100 if total_questions > 0 else 0,
            "agent_detection_improvement_rate": agent_detection_improvements / total_questions * 100 if total_questions > 0 else 0
        }

        return comparison

    def generate_accuracy_metrics(self) -> Dict[str, Any]:
        """Generate comprehensive accuracy metrics."""

        metrics = {
            "tool_selection_accuracy": {
                "perfect_matches": 0,
                "partial_matches": 0,
                "no_matches": 0,
                "average_accuracy_score": 0
            },
            "agent_selection_accuracy": {
                "perfect_matches": 0,
                "partial_matches": 0,
                "no_matches": 0,
                "average_accuracy_score": 0
            },
            "complexity_based_accuracy": {
                "simple": {"tool_avg": 0, "agent_avg": 0},
                "medium": {"tool_avg": 0, "agent_avg": 0},
                "complex": {"tool_avg": 0, "agent_avg": 0}
            }
        }

        enhanced_detailed = self.enhanced_results.get("detailed_results", {})

        tool_f1_scores = []
        agent_f1_scores = []
        complexity_scores = {"simple": {"tool": [], "agent": []},
                           "medium": {"tool": [], "agent": []},
                           "complex": {"tool": [], "agent": []}}

        for question_id, result in enhanced_detailed.items():
            if "accuracy_metrics" in result:
                acc_metrics = result["accuracy_metrics"]

                # Tool accuracy
                tool_f1 = acc_metrics.get("tool_metrics", {}).get("f1_score", 0)
                tool_f1_scores.append(tool_f1)

                if tool_f1 == 1.0:
                    metrics["tool_selection_accuracy"]["perfect_matches"] += 1
                elif tool_f1 > 0:
                    metrics["tool_selection_accuracy"]["partial_matches"] += 1
                else:
                    metrics["tool_selection_accuracy"]["no_matches"] += 1

                # Agent accuracy
                agent_f1 = acc_metrics.get("agent_metrics", {}).get("f1_score", 0)
                agent_f1_scores.append(agent_f1)

                if agent_f1 == 1.0:
                    metrics["agent_selection_accuracy"]["perfect_matches"] += 1
                elif agent_f1 > 0:
                    metrics["agent_selection_accuracy"]["partial_matches"] += 1
                else:
                    metrics["agent_selection_accuracy"]["no_matches"] += 1

                # Complexity-based tracking
                complexity = result.get("complexity", "unknown")
                if complexity in complexity_scores:
                    complexity_scores[complexity]["tool"].append(tool_f1)
                    complexity_scores[complexity]["agent"].append(agent_f1)

        # Calculate averages
        if tool_f1_scores:
            metrics["tool_selection_accuracy"]["average_accuracy_score"] = sum(tool_f1_scores) / len(tool_f1_scores)

        if agent_f1_scores:
            metrics["agent_selection_accuracy"]["average_accuracy_score"] = sum(agent_f1_scores) / len(agent_f1_scores)

        # Complexity averages
        for complexity, scores in complexity_scores.items():
            if scores["tool"]:
                metrics["complexity_based_accuracy"][complexity]["tool_avg"] = sum(scores["tool"]) / len(scores["tool"])
            if scores["agent"]:
                metrics["complexity_based_accuracy"][complexity]["agent_avg"] = sum(scores["agent"]) / len(scores["agent"])

        return metrics

    def generate_detailed_breakdowns(self) -> Dict[str, Any]:
        """Generate detailed breakdowns for further analysis."""

        breakdowns = {
            "execution_timeline_analysis": {},
            "error_analysis": {},
            "resource_efficiency": {},
            "recommendation_engine": {}
        }

        enhanced_detailed = self.enhanced_results.get("detailed_results", {})

        # Error analysis
        planning_errors = 0
        execution_errors = 0
        parsing_errors = 0

        for result in enhanced_detailed.values():
            if not result.get("success", True):
                error_msg = result.get("error_message", "").lower()
                if "planning" in error_msg:
                    planning_errors += 1
                elif "parsing" in error_msg:
                    parsing_errors += 1
                else:
                    execution_errors += 1

        breakdowns["error_analysis"] = {
            "total_errors": planning_errors + execution_errors + parsing_errors,
            "planning_errors": planning_errors,
            "execution_errors": execution_errors,
            "parsing_errors": parsing_errors,
            "error_rate": (planning_errors + execution_errors + parsing_errors) / len(enhanced_detailed) * 100 if enhanced_detailed else 0
        }

        # Resource efficiency
        total_agents = 0
        total_tools = 0
        total_time = 0

        for result in enhanced_detailed.values():
            if result.get("success"):
                perf_metrics = result.get("performance_metrics", {})
                total_agents += perf_metrics.get("agent_count", 0)
                total_tools += perf_metrics.get("tool_count", 0)
                total_time += perf_metrics.get("total_workflow_time", 0)

        successful_questions = sum(1 for r in enhanced_detailed.values() if r.get("success"))

        if successful_questions > 0:
            breakdowns["resource_efficiency"] = {
                "average_agents_per_question": total_agents / successful_questions,
                "average_tools_per_question": total_tools / successful_questions,
                "average_time_per_question": total_time / successful_questions,
                "total_resource_utilization": total_agents + total_tools,
                "efficiency_score": successful_questions / (total_time + 1)  # Adding 1 to avoid division by zero
            }

        # Recommendations
        recommendations = []

        # Based on error analysis
        if parsing_errors > 0:
            recommendations.append("Fix LLM output parsing in PlannerAgent to improve reliability")

        if total_tools == 0:
            recommendations.append("Implement proper tool usage tracking in agent execution")

        if planning_errors > execution_errors:
            recommendations.append("Focus on improving planning agent robustness")

        breakdowns["recommendation_engine"] = {
            "high_priority_recommendations": recommendations,
            "system_health_score": (successful_questions / len(enhanced_detailed) * 100) if enhanced_detailed else 0,
            "improvement_areas": ["tool_tracking", "agent_coordination", "error_handling"]
        }

        return breakdowns

    def export_comprehensive_statistics(self, filename: str = None) -> str:
        """Export comprehensive statistics to JSON file."""

        if not filename:
            filename = f"comprehensive_titanic_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        print("\\n🔄 Generating comprehensive statistics...")
        stats = self.generate_comprehensive_statistics()

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
            print(f"✅ Comprehensive statistics exported to: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error exporting statistics: {str(e)}")
            return ""

    def print_key_findings(self, stats: Dict[str, Any]):
        """Print key findings from the comprehensive analysis."""

        print(f"\\n{'='*70}")
        print("🔍 KEY FINDINGS - COMPREHENSIVE TITANIC ANALYSIS")
        print(f"{'='*70}")

        # Metadata
        metadata = stats.get("metadata", {})
        print(f"\\n📊 Analysis Overview:")
        print(f"   Questions Analyzed: {metadata.get('total_questions_analyzed', 0)}")
        print(f"   Generated: {metadata.get('generated_timestamp', 'Unknown')}")

        # Performance findings
        perf_analysis = stats.get("performance_analysis", {})
        overall_stats = perf_analysis.get("execution_times", {}).get("overall_stats", {})
        success_rates = perf_analysis.get("success_rates", {}).get("overall", {})

        if overall_stats:
            print(f"\\n⏱️ Performance Metrics:")
            print(f"   Average Execution Time: {overall_stats.get('average_time', 0):.2f} seconds")
            print(f"   Total Testing Time: {overall_stats.get('total_time', 0):.2f} seconds")
            print(f"   Fastest Question: {overall_stats.get('min_time', 0):.2f} seconds")
            print(f"   Slowest Question: {overall_stats.get('max_time', 0):.2f} seconds")

        if success_rates:
            print(f"   Success Rate: {success_rates.get('success_rate', 0):.1f}%")

        # Tool selection findings
        tool_analysis = stats.get("tool_selection_analysis", {})
        tool_accuracy = tool_analysis.get("selection_accuracy", {}).get("overall_metrics", {})

        if tool_accuracy:
            print(f"\\n🔧 Tool Selection Analysis:")
            print(f"   Average F1 Score: {tool_accuracy.get('average_f1_score', 0):.3f}")
            print(f"   Average Precision: {tool_accuracy.get('average_precision', 0):.3f}")
            print(f"   Average Recall: {tool_accuracy.get('average_recall', 0):.3f}")
            print(f"   Perfect Precision Questions: {tool_accuracy.get('questions_with_perfect_precision', 0)}")

        # Agent selection findings
        agent_analysis = stats.get("agent_selection_analysis", {})
        agent_accuracy = agent_analysis.get("selection_accuracy", {}).get("overall_metrics", {})

        if agent_accuracy:
            print(f"\\n🤖 Agent Selection Analysis:")
            print(f"   Average F1 Score: {agent_accuracy.get('average_f1_score', 0):.3f}")
            print(f"   Average Precision: {agent_accuracy.get('average_precision', 0):.3f}")
            print(f"   Average Recall: {agent_accuracy.get('average_recall', 0):.3f}")

        # System comparison
        comparison = stats.get("system_comparison", {})
        perf_comp = comparison.get("performance_comparison", {})

        if perf_comp:
            print(f"\\n📈 System Comparison:")
            print(f"   Original Avg Time: {perf_comp.get('original_average_time', 0):.2f}s")
            print(f"   Enhanced Avg Time: {perf_comp.get('enhanced_average_time', 0):.2f}s")
            print(f"   Performance Change: {perf_comp.get('performance_change_percentage', 0):.1f}%")

        # Key issues identified
        detailed_breakdowns = stats.get("detailed_breakdowns", {})
        error_analysis = detailed_breakdowns.get("error_analysis", {})

        if error_analysis:
            print(f"\\n⚠️ Issues Identified:")
            print(f"   Total Errors: {error_analysis.get('total_errors', 0)}")
            print(f"   Planning Errors: {error_analysis.get('planning_errors', 0)}")
            print(f"   Parsing Errors: {error_analysis.get('parsing_errors', 0)}")

        # Recommendations
        recommendations = detailed_breakdowns.get("recommendation_engine", {})
        high_priority = recommendations.get("high_priority_recommendations", [])

        if high_priority:
            print(f"\\n💡 High Priority Recommendations:")
            for i, rec in enumerate(high_priority, 1):
                print(f"   {i}. {rec}")

        print(f"\\n{'='*70}")


def main():
    """Main function to generate comprehensive statistics."""

    print("📊 Comprehensive Titanic Statistics Generator")
    print("=" * 50)

    generator = ComprehensiveTitanicStatsGenerator()

    # Load all data
    generator.load_all_data()

    # Generate and export comprehensive statistics
    filename = generator.export_comprehensive_statistics()

    if filename:
        # Load the generated stats to print key findings
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                stats = json.load(f)

            generator.print_key_findings(stats)

            print(f"\\n🎉 Comprehensive analysis completed!")
            print(f"📋 Full statistics available in: {filename}")

        except Exception as e:
            print(f"❌ Error reading generated file: {str(e)}")

    return filename


if __name__ == "__main__":
    main()