"""Configuration management for the multi-agent system."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the multi-agent system."""
    
    def __init__(self):
        # LangSmith Configuration
        self.langsmith_api_key: Optional[str] = os.getenv("LANGSMITH_API_KEY")
        self.langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "multi-agent-data-analysis")
        self.langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
        
        # Model Configuration
        self.model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.device: str = os.getenv("DEVICE", "auto")
        
        # Hugging Face Configuration
        self.huggingface_token: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
        
        # Application Configuration
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
        
        # Data Configuration
        self.default_dataset_path: str = "data/titanic.csv"
        self.upload_dir: str = "uploads"
        
        # Ensure upload directory exists
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def setup_langsmith(self):
        """Set up LangSmith environment variables."""
        if self.langsmith_api_key and self.langsmith_tracing:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
            os.environ["LANGSMITH_TRACING"] = str(self.langsmith_tracing).lower()
            print(f"LangSmith tracing enabled for project: {self.langsmith_project}")
        else:
            print("LangSmith not configured - tracing disabled")


# Global config instance
config = Config()