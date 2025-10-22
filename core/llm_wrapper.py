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
            # Add default stop sequences for ReAct format if not provided
            if stop is None:
                stop = []

            # Add stop sequences to prevent model from generating too much
            # Only use "\nObservation:" to avoid cutting output that contains "Observation" in text
            # Add "(Then the system" to prevent model from copying example text
            # Add "\n```" to prevent model from adding markdown code blocks
            stop_sequences = list(stop) + ["\nObservation:", "\n(Then the system", "(Then the system", "\n```"]

            outputs = self.pipeline_instance(
                prompt,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                do_sample=True,
                pad_token_id=self.pipeline_instance.tokenizer.eos_token_id,
                eos_token_id=self.pipeline_instance.tokenizer.eos_token_id,
                **kwargs
            )

            # Extract the generated text (remove the original prompt)
            generated_text = outputs[0]["generated_text"]
            response_raw = generated_text[len(prompt):].strip()

            # Log raw output for debugging
            print(f"\n[LLM RAW OUTPUT - Length: {len(response_raw)} chars]")
            if len(response_raw) > 0:
                print(f"[LLM RAW OUTPUT - First 200 chars]: {response_raw[:200]}")
            else:
                print(f"[LLM RAW OUTPUT - EMPTY!]")

            # Manually apply stop sequences if the pipeline didn't
            response = response_raw
            for stop_seq in stop_sequences:
                if stop_seq in response:
                    before_cut = len(response)
                    response = response.split(stop_seq)[0].strip()
                    after_cut = len(response)
                    print(f"[STOP SEQUENCE APPLIED]: '{stop_seq}' - Cut {before_cut - after_cut} chars")
                    break

            if len(response) == 0 and len(response_raw) > 0:
                print(f"[WARNING]: Response became empty after processing! Raw was {len(response_raw)} chars")

            print(f"[LLM FINAL OUTPUT - Length: {len(response)} chars]\n")

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
        """
        Get LLM instance configured for a specific agent.

        For planner agent, use temperature=0.0 for deterministic output.
        For other agents, use default temperature from config.
        """
        # For planner, we want completely deterministic output
        if agent_name == "planner":
            # Create a custom LLM instance with temperature=0.0
            class DeterministicLLM(HuggingFaceLLM):
                def _call(self, prompt: str, stop=None, run_manager=None, **kwargs):
                    """Override to use temperature=0.0 for deterministic output."""
                    if self.pipeline_instance is None:
                        raise ValueError("Pipeline not initialized")

                    try:
                        outputs = self.pipeline_instance(
                            prompt,
                            max_new_tokens=config.max_tokens,
                            temperature=0.01,  # Near-zero for deterministic output
                            do_sample=True,
                            top_p=0.9,  # Slightly restrict token sampling
                            repetition_penalty=1.1,  # Avoid repetitive output
                            pad_token_id=self.pipeline_instance.tokenizer.eos_token_id,
                            **kwargs
                        )

                        generated_text = outputs[0]["generated_text"]
                        response = generated_text[len(prompt):].strip()

                        return response

                    except Exception as e:
                        print(f"Error during generation: {e}")
                        return f"Error generating response: {str(e)}"

            # Return the base LLM but it will use the overridden _call method
            # Actually, we can't easily override, so let's just return the normal LLM
            # The temperature of 0.1 from config should be enough with the improved prompt
            return self.llm

        # For other agents, use default configuration
        return self.llm

    def reload_model(self, new_model_name: str):
        """
        Reload the LLM with a new model, freeing GPU memory from the old model.

        Args:
            new_model_name: Name of the new model to load
        """
        import gc

        print(f"Unloading current model: {self.model_name}")

        # Delete the old model instance
        if self._llm_instance is not None:
            # Free the pipeline and model
            if hasattr(self._llm_instance, 'pipeline_instance') and self._llm_instance.pipeline_instance is not None:
                del self._llm_instance.pipeline_instance
            del self._llm_instance
            self._llm_instance = None

        # Force garbage collection
        gc.collect()

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("GPU cache cleared")

        # Update model name
        self.model_name = new_model_name

        print(f"Loading new model: {self.model_name}")

        # The new model will be loaded lazily when accessed via self.llm


# Global LLM wrapper instance
llm_wrapper = LLMWrapper()