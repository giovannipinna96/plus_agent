"""
Universal Statistics Integration Example
Shows how to integrate the universal statistics system with any multi-agent framework.
"""

import json
import time
from typing import Dict, Any, List
from datetime import datetime

# Import the universal statistics system
from universal_multiagent_statistics import (
    track_multiagent_session,
    export_universal_stats,
    export_universal_report,
    global_stats_collector
)


class UniversalMultiAgentTester:
    """Universal tester that works with any dataset and question types."""

    def __init__(self, orchestrator_class, data_file_path: str = None):
        """
        Initialize with any orchestrator and optional data file.

        Args:
            orchestrator_class: Any multi-agent orchestrator class
            data_file_path: Optional path to data file
        """
        self.orchestrator = orchestrator_class()
        self.data_file_path = data_file_path
        self.test_results = []

    def test_single_request(self, request_text: str, expected_tools: List[str] = None,
                          expected_agents: List[str] = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Test a single request with universal statistics tracking.

        Args:
            request_text: The request/question to process
            expected_tools: Optional list of expected tools (for validation)
            expected_agents: Optional list of expected agents (for validation)
            metadata: Optional metadata for categorization

        Returns:
            Dictionary with test results and statistics
        """

        # Prepare metadata
        test_metadata = {
            "request_type": metadata.get("type", "unknown") if metadata else "unknown",
            "complexity": metadata.get("complexity", "unknown") if metadata else "unknown",
            "dataset": self.data_file_path or "unknown",
            "expected_tools": expected_tools or [],
            "expected_agents": expected_agents or [],
            "timestamp": datetime.now().isoformat()
        }

        print(f"\\n🔬 Testing Request: {request_text[:100]}...")
        print(f"📊 Metadata: {test_metadata}")

        result = {
            "request_text": request_text,
            "metadata": test_metadata,
            "start_time": time.time(),
            "success": False,
            "error": None,
            "orchestrator_result": None,
            "statistics_session_id": None
        }

        try:
            # Track the session with universal statistics
            with track_multiagent_session(request_text, test_metadata) as tracker:
                result["statistics_session_id"] = tracker.session_id

                # Execute with the orchestrator (adapt this based on your orchestrator's API)
                orchestrator_result = self._execute_with_orchestrator(request_text, tracker)

                result["orchestrator_result"] = orchestrator_result
                result["success"] = orchestrator_result.get("status") == "success" if orchestrator_result else False

        except Exception as e:
            result["error"] = str(e)
            result["success"] = False

        finally:
            result["end_time"] = time.time()
            result["execution_time"] = result["end_time"] - result["start_time"]

        self.test_results.append(result)
        return result

    def _execute_with_orchestrator(self, request_text: str, tracker) -> Dict[str, Any]:
        """
        Execute request with orchestrator and track statistics.
        Adapt this method based on your specific orchestrator.
        """
        import time

        # For the LangGraph-based MultiAgentOrchestrator
        if hasattr(self.orchestrator, 'workflow') and hasattr(self.orchestrator.workflow, 'invoke'):
            # This is our LangGraph orchestrator
            start_time = time.time()

            # Create initial state
            initial_state = {
                "messages": [{"role": "user", "content": request_text}],
                "current_file_path": self.data_file_path
            }

            try:
                # Execute the workflow
                result = self.orchestrator.workflow.invoke(initial_state)
                execution_time = time.time() - start_time

                # Convert LangGraph result to our expected format
                orchestrator_result = {
                    "status": "success",
                    "request": request_text,
                    "file_path": self.data_file_path,
                    "total_execution_time": execution_time,
                    "workflow_result": result,
                    "agents_activated": self._extract_agents_from_result(result),
                    "tools_used": self._extract_tools_from_result(result),
                    "final_response": self._extract_final_response(result)
                }

                # Extract and log statistics from the result
                self._extract_and_log_statistics(orchestrator_result, tracker)

                return orchestrator_result

            except Exception as e:
                return {
                    "status": "error",
                    "request": request_text,
                    "file_path": self.data_file_path,
                    "total_execution_time": time.time() - start_time,
                    "error": str(e),
                    "agents_activated": [],
                    "tools_used": []
                }

        elif hasattr(self.orchestrator, 'run_analysis'):
            # For orchestrators with run_analysis method
            result = self.orchestrator.run_analysis(request_text, self.data_file_path)
            self._extract_and_log_statistics(result, tracker)
            return result

        elif hasattr(self.orchestrator, 'process_request'):
            # For orchestrators with process_request method
            result = self.orchestrator.process_request(request_text)
            self._extract_and_log_statistics(result, tracker)
            return result

        else:
            # Mock fallback for demonstration
            return self._create_mock_result(request_text, tracker)

    def _extract_agents_from_result(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract agent information from LangGraph workflow result."""
        agents = []

        # Look for completed steps or agent names in the result
        if isinstance(result, dict) and "completed_steps" in result:
            for step in result.get("completed_steps", []):
                agents.append({
                    "agent": step.replace("Agent", ""),
                    "time": 1.0,  # Placeholder
                    "success": True
                })

        # Check messages for agent activity
        messages = result.get("messages", []) if isinstance(result, dict) else []
        agent_names = set()
        for msg in messages:
            if isinstance(msg, dict) and "name" in msg:
                agent_names.add(msg["name"])

        for agent_name in agent_names:
            agents.append({
                "agent": agent_name,
                "time": 1.0,
                "success": True
            })

        return agents if agents else [{"agent": "UnknownAgent", "time": 0.5, "success": True}]

    def _extract_tools_from_result(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool information from LangGraph workflow result."""
        tools = []

        # Try to extract tool usage from workflow result
        if isinstance(result, dict):
            # Look for tool calls in messages or results
            messages = result.get("messages", [])
            for msg in messages:
                if isinstance(msg, dict) and "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        tools.append({
                            "tool": tool_call.get("name", "unknown_tool"),
                            "agent": "UnknownAgent",
                            "execution_time": 0.3,
                            "success": True,
                            "parameters": tool_call.get("args", {})
                        })

        return tools

    def _extract_final_response(self, result: Dict[str, Any]) -> str:
        """Extract final response from LangGraph workflow result."""
        if isinstance(result, dict):
            # Look for final message in messages list
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    return last_message.get("content", str(result))
                return str(last_message)

        return str(result)

    def _create_mock_result(self, request_text: str, tracker) -> Dict[str, Any]:
        """Create mock result when orchestrator cannot be called."""
        import random
        import time

        time.sleep(random.uniform(0.1, 1.0))

        return {
            "status": "success",
            "request": request_text,
            "file_path": self.data_file_path,
            "total_execution_time": random.uniform(0.5, 2.0),
            "agents_activated": [
                {"agent": "MockAgent", "time": 0.5, "success": True}
            ],
            "tools_used": [
                {"tool": "mock_tool", "agent": "MockAgent", "execution_time": 0.2, "success": True}
            ]
        }

    def _extract_and_log_statistics(self, orchestrator_result: Dict[str, Any], tracker) -> None:
        """
        Extract statistics from orchestrator result and log to universal tracker.
        Adapt this based on your orchestrator's result format.
        """

        # Example extraction - adapt based on your orchestrator's result structure

        # Extract tool usage
        tools_used = orchestrator_result.get("tools_used", [])
        for tool_info in tools_used:
            if isinstance(tool_info, dict):
                tracker.log_tool(
                    tool_name=tool_info.get("tool", "unknown"),
                    agent_name=tool_info.get("agent", "unknown"),
                    parameters=tool_info.get("parameters", {}),
                    execution_time=tool_info.get("execution_time", 0),
                    success=tool_info.get("success", True),
                    result=tool_info.get("result", ""),
                    error=tool_info.get("error")
                )

        # Extract agent activations
        agents_activated = orchestrator_result.get("agents_activated", [])
        for agent_info in agents_activated:
            if isinstance(agent_info, dict):
                tracker.log_agent(
                    agent_name=agent_info.get("agent", "unknown"),
                    execution_time=agent_info.get("time", 0),
                    tools_used=agent_info.get("tools_used", []),
                    iterations=agent_info.get("iterations", 1),
                    success=agent_info.get("success", True),
                    result_summary=agent_info.get("result_summary", ""),
                    error=agent_info.get("error")
                )

        # Alternative: If your orchestrator doesn't provide detailed stats,
        # you can manually log based on known patterns
        if not tools_used and not agents_activated:
            self._log_fallback_statistics(orchestrator_result, tracker)

    def _log_fallback_statistics(self, orchestrator_result: Dict[str, Any], tracker) -> None:
        """
        Fallback method to log statistics when detailed tracking is not available.
        """

        # Log basic execution info
        execution_time = orchestrator_result.get("total_execution_time", 0)
        success = orchestrator_result.get("status") == "success"

        # Try to infer agents from results
        agent_results = orchestrator_result.get("agent_results", {})
        for agent_name in agent_results.keys():
            tracker.log_agent(
                agent_name=agent_name,
                execution_time=execution_time / len(agent_results) if agent_results else execution_time,
                tools_used=[],  # Cannot determine tools used
                iterations=1,
                success=success,
                result_summary=str(agent_results[agent_name])[:100]
            )

    def test_question_set(self, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test a set of questions/requests.

        Args:
            questions: Dictionary of questions in format:
                {
                    "question_id": {
                        "text": "Question text",
                        "expected_tools": [...],
                        "expected_agents": [...],
                        "metadata": {...}
                    }
                }

        Returns:
            Dictionary with comprehensive test results
        """

        print(f"\\n🧪 Universal Multi-Agent Testing")
        print(f"Questions to test: {len(questions)}")
        print(f"Orchestrator: {type(self.orchestrator).__name__}")
        print(f"Data file: {self.data_file_path or 'None'}")

        results = {}
        successful_tests = 0

        for question_id, question_data in questions.items():
            print(f"\\n[{len(results)+1}/{len(questions)}] Testing {question_id}")

            result = self.test_single_request(
                request_text=question_data["text"],
                expected_tools=question_data.get("expected_tools", []),
                expected_agents=question_data.get("expected_agents", []),
                metadata=question_data.get("metadata", {})
            )

            results[question_id] = result

            if result["success"]:
                successful_tests += 1
                print(f"✅ {question_id} completed in {result['execution_time']:.2f}s")
            else:
                print(f"❌ {question_id} failed: {result.get('error', 'Unknown error')}")

        # Generate summary
        summary = {
            "test_metadata": {
                "total_questions": len(questions),
                "successful_tests": successful_tests,
                "failed_tests": len(questions) - successful_tests,
                "success_rate": successful_tests / len(questions) * 100 if questions else 0,
                "orchestrator_type": type(self.orchestrator).__name__,
                "data_file": self.data_file_path,
                "test_timestamp": datetime.now().isoformat()
            },
            "individual_results": results
        }

        return summary

    def export_universal_statistics(self, base_filename: str = None) -> Dict[str, str]:
        """Export universal statistics and reports."""

        if not base_filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"universal_test_{timestamp}"

        # Export comprehensive statistics
        stats_file = export_universal_stats(f"{base_filename}_stats.json")

        # Export summary report
        report_file = export_universal_report(f"{base_filename}_report.md")

        # Export test results
        results_file = f"{base_filename}_results.json"
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"⚠️ Failed to export test results: {str(e)}")
            results_file = None

        return {
            "statistics_file": stats_file,
            "report_file": report_file,
            "results_file": results_file
        }


# Adapter functions for different orchestrator types
def adapt_titanic_orchestrator():
    """Adapter for the Titanic-specific orchestrator."""

    # Import your specific orchestrator
    try:
        from core.enhanced_orchestrator import EnhancedMultiAgentOrchestrator
        return EnhancedMultiAgentOrchestrator
    except ImportError:
        try:
            from core.orchestrator import MultiAgentOrchestrator
            return MultiAgentOrchestrator
        except ImportError:
            print("⚠️ Could not import orchestrator - using mock")
            return MockOrchestrator


class MockOrchestrator:
    """Mock orchestrator for demonstration purposes."""

    def run_analysis(self, request_text: str, data_file: str = None) -> Dict[str, Any]:
        """Mock analysis method."""
        import random
        time.sleep(random.uniform(0.1, 2.0))  # Simulate processing time

        return {
            "status": "success",
            "request": request_text,
            "file_path": data_file,
            "total_execution_time": random.uniform(0.5, 3.0),
            "agents_activated": [
                {"agent": "MockAgent1", "time": 0.5, "success": True},
                {"agent": "MockAgent2", "time": 1.0, "success": True}
            ],
            "tools_used": [
                {"tool": "mock_tool1", "agent": "MockAgent1", "execution_time": 0.2, "success": True},
                {"tool": "mock_tool2", "agent": "MockAgent2", "execution_time": 0.3, "success": True}
            ]
        }


def main():
    """Example usage of the universal statistics system."""

    print("🚀 Universal Multi-Agent Statistics System")
    print("=" * 50)

    # Initialize with any orchestrator
    orchestrator_class = adapt_titanic_orchestrator()
    tester = UniversalMultiAgentTester(orchestrator_class, "data/titanic.csv")

    # Example question set (works with any domain)
    questions = {
        "analysis_1": {
            "text": "Analyze the data and provide summary statistics",
            "expected_tools": ["data_reader", "analyzer"],
            "expected_agents": ["DataAgent", "AnalysisAgent"],
            "metadata": {"complexity": "simple", "type": "analysis"}
        },
        "prediction_1": {
            "text": "Create a predictive model for the target variable",
            "expected_tools": ["data_reader", "preprocessor", "model_trainer"],
            "expected_agents": ["DataAgent", "MLAgent"],
            "metadata": {"complexity": "complex", "type": "prediction"}
        }
    }

    # Run tests
    results = tester.test_question_set(questions)

    # Export universal statistics
    exported_files = tester.export_universal_statistics()

    print(f"\\n📊 Testing Completed!")
    print(f"Success Rate: {results['test_metadata']['success_rate']:.1f}%")
    print(f"Files exported:")
    for file_type, filename in exported_files.items():
        if filename:
            print(f"  - {file_type}: {filename}")

    return results


if __name__ == "__main__":
    main()