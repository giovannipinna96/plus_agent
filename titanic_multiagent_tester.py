"""
Titanic Multi-Agent System Tester
Tests the multi-agent system with the 10 Titanic questions and compares tool selection.
"""

import json
import sys
import os
from typing import Dict, Any, List
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import MultiAgentOrchestrator


class TitanicMultiAgentTester:
    """Test the multi-agent system with Titanic questions."""

    def __init__(self, data_file_path: str = "data/titanic.csv"):
        self.data_file_path = data_file_path
        self.orchestrator = MultiAgentOrchestrator()
        self.results = {}

    def test_single_question(self, question_id: str, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single question with the multi-agent system."""

        print(f"\\n{'='*60}")
        print(f"Testing Question {question_id} with Multi-Agent System")
        print(f"Text: {question_data['text']}")
        print(f"Expected Agents: {question_data.get('expected_agents', [])}")
        print(f"Expected Tools: {question_data['tools_sequence']}")
        print(f"{'='*60}")

        start_time = time.time()

        result = {
            "question_id": question_id,
            "question_text": question_data["text"],
            "expected_agents": question_data.get("expected_agents", []),
            "expected_tools": question_data["tools_sequence"],
            "actual_agents_used": [],
            "actual_tools_used": [],
            "execution_time": 0,
            "success": False,
            "response": None,
            "error_message": None
        }

        try:
            print(f"\\n🚀 Executing with multi-agent system...")

            # Execute with orchestrator
            final_state = self.orchestrator.run_analysis(
                user_prompt=question_data["text"],
                file_path=self.data_file_path
            )

            execution_time = time.time() - start_time

            # Extract information from the final state
            if final_state:
                result["success"] = True
                result["execution_time"] = execution_time

                # Extract response
                messages = final_state.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    result["response"] = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # Try to extract agents and tools used from results
                results_dict = final_state.get("results", {})
                for agent_name, agent_result in results_dict.items():
                    result["actual_agents_used"].append(agent_name)

                # Extract tool usage if available
                # This would require modifications to the orchestrator to track tool usage
                # For now, we'll just log what we can see

                print(f"\\n✅ Question {question_id} completed successfully")
                print(f"⏱️  Execution time: {execution_time:.2f} seconds")
                print(f"🤖 Agents used: {result['actual_agents_used']}")
                print(f"📝 Response preview: {(result['response'] or '')[:200]}...")

            else:
                result["error_message"] = "No final state returned"
                print(f"\\n❌ Question {question_id} failed: No final state returned")

        except Exception as e:
            execution_time = time.time() - start_time
            result["error_message"] = str(e)
            result["execution_time"] = execution_time
            print(f"\\n❌ Question {question_id} failed: {str(e)}")
            print(f"⏱️  Execution time: {execution_time:.2f} seconds")

        return result

    def test_all_questions(self, questions_file: str = "titanic_questions.json") -> Dict[str, Any]:
        """Test all questions with the multi-agent system."""

        print("🚢 Titanic Multi-Agent System Tester")
        print("====================================")
        print(f"Loading questions from: {questions_file}")

        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        except FileNotFoundError:
            print(f"❌ Questions file {questions_file} not found!")
            return {}

        test_results = {}
        total_questions = len(questions)

        for i, (question_id, question_data) in enumerate(questions.items(), 1):
            print(f"\\n[{i}/{total_questions}] Processing {question_id}...")

            try:
                result = self.test_single_question(question_id, question_data)
                test_results[question_id] = result

            except Exception as e:
                print(f"\\n❌ Error testing {question_id}: {str(e)}")
                test_results[question_id] = {
                    "question_id": question_id,
                    "success": False,
                    "error_message": str(e)
                }

        # Save results
        results_file = f"titanic_multiagent_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)

        print(f"\\n📊 Test results saved to: {results_file}")
        return test_results

    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the test results and compare expected vs actual."""

        analysis = {
            "total_questions": len(results),
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0,
            "agent_accuracy": {},
            "tool_accuracy": {},
            "detailed_comparison": []
        }

        total_execution_time = 0
        successful_results = []

        for question_id, result in results.items():
            if result.get("success", False):
                analysis["successful_executions"] += 1
                total_execution_time += result.get("execution_time", 0)
                successful_results.append(result)
            else:
                analysis["failed_executions"] += 1

            # Detailed comparison
            comparison = {
                "question_id": question_id,
                "expected_agents": result.get("expected_agents", []),
                "actual_agents": result.get("actual_agents_used", []),
                "expected_tools": result.get("expected_tools", []),
                "actual_tools": result.get("actual_tools_used", []),
                "agent_match": False,
                "tool_match": False,
                "success": result.get("success", False)
            }

            # Check agent accuracy
            expected_agents = set(comparison["expected_agents"])
            actual_agents = set(comparison["actual_agents"])
            if expected_agents and actual_agents:
                agent_overlap = len(expected_agents & actual_agents)
                agent_total = len(expected_agents)
                comparison["agent_match"] = (agent_overlap / agent_total) >= 0.5 if agent_total > 0 else False

            # Check tool accuracy (would need tool tracking)
            # For now, we'll mark as unknown
            comparison["tool_match"] = "unknown"

            analysis["detailed_comparison"].append(comparison)

        # Calculate averages
        if analysis["successful_executions"] > 0:
            analysis["average_execution_time"] = total_execution_time / analysis["successful_executions"]

        # Calculate success rate
        analysis["success_rate"] = analysis["successful_executions"] / analysis["total_questions"] * 100

        return analysis

    def print_analysis_report(self, analysis: Dict[str, Any]):
        """Print a detailed analysis report."""

        print(f"\\n{'='*60}")
        print("MULTI-AGENT SYSTEM ANALYSIS REPORT")
        print(f"{'='*60}")

        print(f"📊 Overall Statistics:")
        print(f"   Total Questions: {analysis['total_questions']}")
        print(f"   Successful Executions: {analysis['successful_executions']}")
        print(f"   Failed Executions: {analysis['failed_executions']}")
        print(f"   Success Rate: {analysis['success_rate']:.1f}%")
        print(f"   Average Execution Time: {analysis['average_execution_time']:.2f} seconds")

        print(f"\\n🔍 Detailed Question Analysis:")
        for comparison in analysis["detailed_comparison"]:
            qid = comparison["question_id"]
            status = "✅" if comparison["success"] else "❌"
            agent_status = "🎯" if comparison["agent_match"] else "❌"

            print(f"\\n   {status} {qid}:")
            print(f"      Agent Selection {agent_status}: Expected {comparison['expected_agents']} → Got {comparison['actual_agents']}")
            print(f"      Tool Selection: Expected {len(comparison['expected_tools'])} tools → Analysis pending")

        print(f"\\n{'='*60}")


def main():
    """Main testing function."""

    tester = TitanicMultiAgentTester()

    # Check if data file exists
    if not os.path.exists(tester.data_file_path):
        print(f"❌ Data file not found: {tester.data_file_path}")
        print("Please ensure the Titanic dataset is available at the specified path.")
        return

    # Run all tests
    results = tester.test_all_questions()

    if results:
        # Analyze results
        analysis = tester.analyze_results(results)

        # Print report
        tester.print_analysis_report(analysis)

        # Save analysis
        analysis_file = f"titanic_multiagent_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        print(f"\\n📋 Analysis saved to: {analysis_file}")

    return results


if __name__ == "__main__":
    main()