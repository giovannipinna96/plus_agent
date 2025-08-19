"""LangSmith integration for observability."""

import os
from typing import Dict, Any, Optional
from langsmith import traceable
from functools import wraps
from plus_agent.core.config import config


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
    """Logger for LangSmith integration."""
    
    def __init__(self):
        self.enabled = setup_langsmith()
    
    def log_user_interaction(self, user_prompt: str, file_path: Optional[str] = None):
        """Log user interaction."""
        if self.enabled:
            metadata = {
                "interaction_type": "user_prompt",
                "file_provided": file_path is not None
            }
            if file_path:
                metadata["file_name"] = os.path.basename(file_path)
            
            # This would be logged as part of the traced functions
            return metadata
        return {}
    
    def log_agent_result(self, agent_name: str, result: Dict[str, Any]):
        """Log agent result."""
        if self.enabled:
            metadata = {
                "agent": agent_name,
                "status": result.get("status", "unknown"),
                "has_error": "error" in result
            }
            return metadata
        return {}
    
    def log_workflow_completion(self, workflow_result: Dict[str, Any]):
        """Log workflow completion."""
        if self.enabled:
            metadata = {
                "workflow_status": workflow_result.get("status", "unknown"),
                "completed_steps": len(workflow_result.get("completed_steps", [])),
                "agents_used": len(workflow_result.get("agent_results", {})),
                "has_error": workflow_result.get("status") == "error"
            }
            return metadata
        return {}


# Global logger instance
langsmith_logger = LangSmithLogger()