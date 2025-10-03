"""Machine learning prediction agent."""

from typing import List, Dict, Any
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.llm_wrapper import llm_wrapper
from tools.ml_tools import train_regression_model, train_svm_model, train_random_forest_model, train_knn_model, evaluate_model


class MLPredictionAgent:
    """Agent specialized in machine learning model training and evaluation."""
    
    def __init__(self):
        self.llm = llm_wrapper.get_llm_for_agent("ml_prediction")
        self.tools = [
            train_regression_model,
            train_svm_model,
            train_random_forest_model,
            train_knn_model,
            evaluate_model
        ]
        
        # Create the ML prediction prompt
        self.prompt = PromptTemplate(
            template="""You are a Machine Learning Prediction Agent specialized in training and evaluating machine learning models. Your job is to help users build predictive models from their data.

You can:
- Train regression models (linear regression, random forest regression)
- Train SVM models (classification and regression)
- Train Random Forest models (with feature importance analysis)
- Train K-Nearest Neighbors models
- Evaluate trained models on test data

Always explain:
- Which model is appropriate for the problem
- What the performance metrics mean
- Which features are most important (when available)
- Recommendations for model improvement

Available tools: {tool_names}
{tools}

Task: {input}

{agent_scratchpad}

Train the requested model and provide detailed analysis of the results.""",
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        # Create the agent
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5
        )
    
    def train_model(self, file_path: str, ml_request: str) -> Dict[str, Any]:
        """
        Train a machine learning model based on user request.
        
        Args:
            file_path: Path to the data file
            ml_request: Description of the ML task
            
        Returns:
            Dictionary containing the training results
        """
        try:
            task = f"Train a machine learning model on the dataset at {file_path} based on this request: {ml_request}"
            
            result = self.agent_executor.invoke({"input": task})
            
            return {
                "status": "success",
                "file_path": file_path,
                "request": ml_request,
                "result": result.get("output", ""),
                "agent_type": "ml_prediction"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_type": "ml_prediction"
            }
    
    def predict_with_model(self, model_path: str, test_data_path: str, prediction_request: str) -> Dict[str, Any]:
        """
        Make predictions using a trained model.
        
        Args:
            model_path: Path to the saved model
            test_data_path: Path to test data
            prediction_request: Description of what to predict
            
        Returns:
            Dictionary containing the prediction results
        """
        try:
            task = f"Use the model at {model_path} to make predictions on data at {test_data_path}. Request: {prediction_request}"
            
            result = self.agent_executor.invoke({"input": task})
            
            return {
                "status": "success",
                "model_path": model_path,
                "test_data_path": test_data_path,
                "request": prediction_request,
                "result": result.get("output", ""),
                "agent_type": "ml_prediction"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_type": "ml_prediction"
            }
    
    def recommend_model(self, file_path: str, target_column: str, problem_type: str = None) -> Dict[str, Any]:
        """
        Recommend the best model for a given problem.
        
        Args:
            file_path: Path to the data file
            target_column: Target variable column
            problem_type: Type of problem (classification, regression, auto)
            
        Returns:
            Dictionary containing model recommendations
        """
        try:
            if problem_type:
                task = f"Recommend the best machine learning model for a {problem_type} problem using dataset at {file_path} with target column '{target_column}'. Explain your reasoning."
            else:
                task = f"Analyze the dataset at {file_path} with target column '{target_column}' and recommend the most appropriate machine learning approach. Determine if this is a classification or regression problem and suggest the best model."
            
            result = self.agent_executor.invoke({"input": task})
            
            return {
                "status": "success",
                "file_path": file_path,
                "target_column": target_column,
                "problem_type": problem_type,
                "recommendation": result.get("output", ""),
                "agent_type": "ml_prediction"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_type": "ml_prediction"
            }