"""
Tool Tracking Wrapper - System for detailed tool usage monitoring.
"""

import time
import json
import functools
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import threading

class ToolTracker:
    """Thread-safe tracker for tool usage statistics."""

    def __init__(self):
        self.tool_calls = []
        self.agent_calls = []
        self.current_session = None
        self.lock = threading.Lock()

    def start_session(self, session_id: str, question_text: str):
        """Start a new tracking session."""
        with self.lock:
            self.current_session = {
                "session_id": session_id,
                "question_text": question_text,
                "start_time": time.time(),
                "tool_calls": [],
                "agent_calls": [],
                "total_tools_used": 0,
                "total_agents_used": 0
            }

    def end_session(self) -> Dict[str, Any]:
        """End the current session and return statistics."""
        with self.lock:
            if self.current_session:
                self.current_session["end_time"] = time.time()
                self.current_session["total_execution_time"] = (
                    self.current_session["end_time"] - self.current_session["start_time"]
                )
                session_data = self.current_session.copy()
                self.current_session = None
                return session_data
            return {}

    def log_tool_call(self, tool_name: str, agent_name: str, parameters: Dict[str, Any],
                     execution_time: float, result: str, success: bool, error: str = None):
        """Log a tool call with detailed information."""
        with self.lock:
            if self.current_session:
                tool_call = {
                    "timestamp": time.time(),
                    "tool_name": tool_name,
                    "called_by_agent": agent_name,
                    "parameters": parameters,
                    "execution_time": execution_time,
                    "result_preview": result[:200] if result else "",
                    "result_length": len(result) if result else 0,
                    "success": success,
                    "error": error
                }
                self.current_session["tool_calls"].append(tool_call)
                self.current_session["total_tools_used"] += 1

    def log_agent_activation(self, agent_name: str, execution_time: float,
                           tools_used: List[str], iterations: int = 1,
                           result_summary: str = ""):
        """Log agent activation and execution details."""
        with self.lock:
            if self.current_session:
                agent_call = {
                    "timestamp": time.time(),
                    "agent_name": agent_name,
                    "execution_time": execution_time,
                    "tools_used": tools_used,
                    "iterations": iterations,
                    "result_summary": result_summary[:150]
                }
                self.current_session["agent_calls"].append(agent_call)
                self.current_session["total_agents_used"] += 1

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        with self.lock:
            if self.current_session:
                return {
                    "tools_used_count": len(self.current_session["tool_calls"]),
                    "agents_used_count": len(self.current_session["agent_calls"]),
                    "current_duration": time.time() - self.current_session["start_time"],
                    "tools_list": [call["tool_name"] for call in self.current_session["tool_calls"]],
                    "agents_list": [call["agent_name"] for call in self.current_session["agent_calls"]]
                }
            return {}


# Global tracker instance
global_tracker = ToolTracker()


def track_tool_usage(tool_name: str, agent_name: str = "unknown"):
    """Decorator to track tool usage automatically."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            result = ""
            error = None

            try:
                # Call the original function
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                # Log the tool call
                execution_time = time.time() - start_time
                global_tracker.log_tool_call(
                    tool_name=tool_name,
                    agent_name=agent_name,
                    parameters={"args_count": len(args), "kwargs": list(kwargs.keys())},
                    execution_time=execution_time,
                    result=str(result) if result else "",
                    success=success,
                    error=error
                )
        return wrapper
    return decorator


def track_agent_execution(agent_name: str):
    """Decorator to track agent execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            tools_used = []

            # Get tools used before this call
            initial_stats = global_tracker.get_session_stats()
            initial_tool_count = initial_stats.get("tools_used_count", 0)

            try:
                result = func(*args, **kwargs)

                # Get tools used after this call
                final_stats = global_tracker.get_session_stats()
                final_tool_count = final_stats.get("tools_used_count", 0)

                # Calculate tools used by this agent
                new_tools = final_tool_count - initial_tool_count
                if new_tools > 0:
                    all_tools = final_stats.get("tools_list", [])
                    tools_used = all_tools[-new_tools:] if len(all_tools) >= new_tools else all_tools

                return result
            finally:
                execution_time = time.time() - start_time
                result_summary = str(result)[:150] if 'result' in locals() else ""

                global_tracker.log_agent_activation(
                    agent_name=agent_name,
                    execution_time=execution_time,
                    tools_used=tools_used,
                    iterations=1,
                    result_summary=result_summary
                )
        return wrapper
    return decorator


class DetailedStatsCollector:
    """Collect and analyze detailed statistics."""

    def __init__(self):
        self.sessions = []

    def add_session(self, session_data: Dict[str, Any]):
        """Add a completed session to the collection."""
        self.sessions.append(session_data)

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive tool usage statistics."""
        tool_stats = {}
        total_tool_calls = 0

        for session in self.sessions:
            for tool_call in session.get("tool_calls", []):
                tool_name = tool_call["tool_name"]
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {
                        "total_calls": 0,
                        "total_time": 0,
                        "success_count": 0,
                        "error_count": 0,
                        "agents_using": set(),
                        "average_time": 0
                    }

                stats = tool_stats[tool_name]
                stats["total_calls"] += 1
                stats["total_time"] += tool_call["execution_time"]
                stats["agents_using"].add(tool_call["called_by_agent"])

                if tool_call["success"]:
                    stats["success_count"] += 1
                else:
                    stats["error_count"] += 1

                total_tool_calls += 1

        # Calculate averages and convert sets to lists
        for tool_name, stats in tool_stats.items():
            stats["average_time"] = stats["total_time"] / stats["total_calls"] if stats["total_calls"] > 0 else 0
            stats["success_rate"] = stats["success_count"] / stats["total_calls"] if stats["total_calls"] > 0 else 0
            stats["agents_using"] = list(stats["agents_using"])

        return {
            "total_unique_tools": len(tool_stats),
            "total_tool_calls": total_tool_calls,
            "tool_details": tool_stats
        }

    def get_agent_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent usage statistics."""
        agent_stats = {}
        total_agent_calls = 0

        for session in self.sessions:
            for agent_call in session.get("agent_calls", []):
                agent_name = agent_call["agent_name"]
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = {
                        "total_activations": 0,
                        "total_time": 0,
                        "tools_used": set(),
                        "average_time": 0,
                        "average_tools_per_activation": 0
                    }

                stats = agent_stats[agent_name]
                stats["total_activations"] += 1
                stats["total_time"] += agent_call["execution_time"]
                stats["tools_used"].update(agent_call["tools_used"])

                total_agent_calls += 1

        # Calculate averages and convert sets to lists
        for agent_name, stats in agent_stats.items():
            stats["average_time"] = stats["total_time"] / stats["total_activations"] if stats["total_activations"] > 0 else 0
            stats["tools_used"] = list(stats["tools_used"])
            stats["unique_tools_count"] = len(stats["tools_used"])

        return {
            "total_unique_agents": len(agent_stats),
            "total_agent_activations": total_agent_calls,
            "agent_details": agent_stats
        }

    def generate_session_comparison(self, expected_tools: List[str], expected_agents: List[str], session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare expected vs actual tools and agents for a session."""
        actual_tools = [call["tool_name"] for call in session_data.get("tool_calls", [])]
        actual_agents = [call["agent_name"] for call in session_data.get("agent_calls", [])]

        # Remove duplicates while preserving order
        actual_tools_unique = list(dict.fromkeys(actual_tools))
        actual_agents_unique = list(dict.fromkeys(actual_agents))

        # Calculate accuracy metrics
        tool_intersection = set(expected_tools) & set(actual_tools_unique)
        agent_intersection = set(expected_agents) & set(actual_agents_unique)

        tool_accuracy = len(tool_intersection) / len(expected_tools) if expected_tools else 1.0
        agent_accuracy = len(agent_intersection) / len(expected_agents) if expected_agents else 1.0

        return {
            "expected_vs_actual": {
                "expected_tools": expected_tools,
                "actual_tools": actual_tools_unique,
                "expected_agents": expected_agents,
                "actual_agents": actual_agents_unique
            },
            "accuracy_metrics": {
                "tool_selection_accuracy": tool_accuracy,
                "agent_selection_accuracy": agent_accuracy,
                "tools_matched": list(tool_intersection),
                "agents_matched": list(agent_intersection),
                "extra_tools_used": list(set(actual_tools_unique) - set(expected_tools)),
                "extra_agents_used": list(set(actual_agents_unique) - set(expected_agents)),
                "missing_tools": list(set(expected_tools) - set(actual_tools_unique)),
                "missing_agents": list(set(expected_agents) - set(actual_agents_unique))
            }
        }

    def export_detailed_statistics(self, filename: str = None) -> str:
        """Export all statistics to a JSON file."""
        if not filename:
            filename = f"detailed_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        stats = {
            "export_timestamp": datetime.now().isoformat(),
            "total_sessions": len(self.sessions),
            "tool_usage_stats": self.get_tool_usage_stats(),
            "agent_usage_stats": self.get_agent_usage_stats(),
            "session_details": self.sessions
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
            return filename
        except Exception as e:
            print(f"Error exporting statistics: {str(e)}")
            return ""


# Global stats collector
global_stats_collector = DetailedStatsCollector()