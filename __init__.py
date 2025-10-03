"""Plus-Agent: Multi-agent data analysis system."""

__version__ = "0.1.0"
__author__ = "Plus-Agent Team"
__description__ = "Multi-agent system for automated data analysis using LangChain and LangGraph"

# Import main modules
from . import agents
from . import core
from . import tools

__all__ = ["agents", "core", "tools"]