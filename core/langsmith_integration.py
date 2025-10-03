"""Enhanced LangSmith integration for comprehensive observability."""

import os
import time
import psutil
from typing import Dict, Any, Optional, List
from langsmith import traceable
from functools import wraps
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import config


def setup_langsmith():
    """Setup LangSmith environment if configured."""
    if config.langsmith_api_key and config.langsmith_tracing:
        os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
        os.environ["LANGSMITH_TRACING"] = str(config.langsmith_tracing).lower()
        return True
    return False


def trace_agent_execution(agent_name: str):
    """Decorator to trace agent execution with LangSmith."""
    def decorator(func):
        if setup_langsmith():
            @traceable(
                run_type="agent",
                name=f"{agent_name}_execution",
                metadata={"agent_type": agent_name}
            )
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        else:
            return func
    return decorator


def trace_tool_execution(tool_name: str):
    """Decorator to trace tool execution with LangSmith."""
    def decorator(func):
        if setup_langsmith():
            @traceable(
                run_type="tool",
                name=f"{tool_name}_execution",
                metadata={"tool_type": tool_name}
            )
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        else:
            return func
    return decorator


def trace_workflow_execution(workflow_name: str):
    """Decorator to trace overall workflow execution with LangSmith."""
    def decorator(func):
        if setup_langsmith():
            @traceable(
                run_type="chain", 
                name=f"{workflow_name}_workflow",
                metadata={"workflow_type": workflow_name}
            )
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        else:
            return func
    return decorator


class LangSmithLogger:
    """Enhanced logger for LangSmith integration with performance monitoring."""

    def __init__(self):
        self.enabled = setup_langsmith()
        self.session_start = time.time()
        self.workflow_counter = 0

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "timestamp": datetime.now().isoformat()
            }
        except Exception:
            return {}

    def log_user_interaction(self, user_prompt: str, file_path: Optional[str] = None):
        """Log user interaction with enhanced metadata."""
        if self.enabled:
            self.workflow_counter += 1
            metadata = {
                "interaction_type": "user_prompt",
                "workflow_id": f"workflow_{self.workflow_counter}",
                "session_duration_seconds": round(time.time() - self.session_start, 2),
                "prompt_length": len(user_prompt),
                "prompt_complexity": self._assess_prompt_complexity(user_prompt),
                "file_provided": file_path is not None,
                "system_metrics": self.get_system_metrics()
            }

            if file_path:
                metadata.update({
                    "file_name": os.path.basename(file_path),
                    "file_extension": os.path.splitext(file_path)[1].lower(),
                    "file_exists": os.path.exists(file_path)
                })

                try:
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        metadata["file_size_mb"] = round(file_size / (1024**2), 2)
                except Exception:
                    pass

            return metadata
        return {}

    def _assess_prompt_complexity(self, prompt: str) -> str:
        """Assess prompt complexity based on keywords and length."""
        prompt_lower = prompt.lower()

        # Keywords that indicate complexity levels
        simple_keywords = ['show', 'display', 'preview', 'basic', 'info', 'summary']
        medium_keywords = ['calculate', 'filter', 'group', 'analyze', 'transform', 'convert']
        complex_keywords = ['train', 'model', 'predict', 'machine learning', 'workflow', 'complete']
        comprehensive_keywords = ['comprehensive', 'end-to-end', 'complete analysis', 'workflow']

        # Count keyword matches
        comprehensive_count = sum(1 for kw in comprehensive_keywords if kw in prompt_lower)
        complex_count = sum(1 for kw in complex_keywords if kw in prompt_lower)
        medium_count = sum(1 for kw in medium_keywords if kw in prompt_lower)
        simple_count = sum(1 for kw in simple_keywords if kw in prompt_lower)

        # Determine complexity
        if comprehensive_count > 0 or len(prompt) > 200:
            return "comprehensive"
        elif complex_count > 0 or len(prompt) > 100:
            return "complex"
        elif medium_count > 0 or len(prompt) > 50:
            return "medium"
        else:
            return "simple"

    def log_agent_result(self, agent_name: str, result: Dict[str, Any], execution_time: float = 0):
        """Log agent result with performance metrics."""
        if self.enabled:
            metadata = {
                "agent": agent_name,
                "status": result.get("status", "unknown"),
                "has_error": "error" in result,
                "execution_time_seconds": round(execution_time, 3),
                "system_metrics": self.get_system_metrics()
            }

            # Add agent-specific metrics
            if agent_name == "data_reader":
                if "analysis" in result:
                    metadata["analysis_length"] = len(str(result["analysis"]))
            elif agent_name == "ml_prediction":
                if "model_type" in result:
                    metadata["model_type"] = result["model_type"]
                if "accuracy" in result:
                    metadata["model_accuracy"] = result.get("accuracy", 0)

            return metadata
        return {}

    def log_workflow_completion(self, workflow_result: Dict[str, Any]):
        """Log workflow completion with comprehensive metrics."""
        if self.enabled:
            completed_steps = workflow_result.get("completed_steps", [])
            agent_results = workflow_result.get("agent_results", {})

            metadata = {
                "workflow_status": workflow_result.get("status", "unknown"),
                "completed_steps_count": len(completed_steps),
                "agents_used_count": len(agent_results),
                "agents_used": list(agent_results.keys()),
                "has_error": workflow_result.get("status") == "error",
                "system_metrics": self.get_system_metrics()
            }

            # Calculate success rates
            successful_agents = sum(1 for result in agent_results.values()
                                  if result.get("status") == "success")
            if agent_results:
                metadata["agent_success_rate"] = round(successful_agents / len(agent_results), 2)

            # Add timing information
            if "start_time" in workflow_result:
                total_time = time.time() - workflow_result["start_time"]
                metadata["total_execution_time_seconds"] = round(total_time, 3)

            return metadata
        return {}

    def log_tool_execution(self, tool_name: str, input_data: Any, output_data: Any,
                          execution_time: float, success: bool):
        """Log individual tool execution with detailed metrics."""
        if self.enabled:
            metadata = {
                "tool_name": tool_name,
                "execution_time_seconds": round(execution_time, 3),
                "success": success,
                "input_size": len(str(input_data)) if input_data else 0,
                "output_size": len(str(output_data)) if output_data else 0,
                "timestamp": datetime.now().isoformat(),
                "system_metrics": self.get_system_metrics()
            }

            return metadata
        return {}


# Global logger instance
langsmith_logger = LangSmithLogger()