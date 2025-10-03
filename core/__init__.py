"""Core module for multi-agent data analysis system."""

from .config import Config, config
from .llm_wrapper import LLMWrapper, llm_wrapper

__all__ = ["Config", "config", "LLMWrapper", "llm_wrapper"]