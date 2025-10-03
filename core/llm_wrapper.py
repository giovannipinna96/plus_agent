"""LLM wrapper for consistent API across agents."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, Any, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import config


class HuggingFaceLLM(LLM):
    """Custom LangChain LLM wrapper for Hugging Face models."""

    model_name: str = ""
    pipeline_instance: Optional[Any] = None

    def __init__(self, model_name: str = None, **kwargs):
        # Set model_name before calling super().__init__
        if model_name:
            kwargs['model_name'] = model_name
        else:
            kwargs['model_name'] = config.model_name
        super().__init__(**kwargs)
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the Hugging Face pipeline."""
        try:
            print(f"Loading model: {self.model_name}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=config.huggingface_token
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=config.device,
                token=config.huggingface_token,
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            self.pipeline_instance = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=config.device,
            )
            
            # Set pad token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            print(f"Model loaded successfully on device: {config.device}")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to a smaller model or using CPU...")
            # Fallback logic could be implemented here
            raise e
    
    @property
    def _llm_type(self) -> str:
        return "huggingface"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the model with the given prompt."""
        if self.pipeline_instance is None:
            raise ValueError("Pipeline not initialized")
        
        # Generate response
        try:
            outputs = self.pipeline_instance(
                prompt,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                do_sample=True,
                pad_token_id=self.pipeline_instance.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Extract the generated text (remove the original prompt)
            generated_text = outputs[0]["generated_text"]
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"Error generating response: {str(e)}"


class LLMWrapper:
    """Wrapper class for managing LLM instances."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.model_name
        self._llm_instance = None
    
    @property
    def llm(self) -> HuggingFaceLLM:
        """Get or create LLM instance."""
        if self._llm_instance is None:
            self._llm_instance = HuggingFaceLLM(model_name=self.model_name)
        return self._llm_instance
    
    def get_llm_for_agent(self, agent_name: str) -> HuggingFaceLLM:
        """Get LLM instance configured for a specific agent."""
        # Could customize per agent if needed
        return self.llm


# Global LLM wrapper instance
llm_wrapper = LLMWrapper()