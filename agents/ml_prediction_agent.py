"""Machine learning prediction agent."""

from typing import List, Dict, Any
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.llm_wrapper import llm_wrapper
# Import JSON wrapper tools that use StructuredTool with single string parameter
# from tools.json_wrapper_tools import train_regression_model, train_svm_model, train_random_forest_model, train_knn_model, evaluate_model
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
        
        # Create the ML prediction prompt with explicit ReAct format
        self.prompt = PromptTemplate(
            template="""You are a Machine Learning Prediction Agent specialized in training and evaluating machine learning models. Your job is to help users build predictive models from their data.

You have access to these tools:
- train_regression_model: Train linear regression models for continuous target variables
- train_svm_model: Train Support Vector Machine for classification or regression
- train_random_forest_model: Train Random Forest with feature importance analysis
- train_knn_model: Train K-Nearest Neighbors classifier
- evaluate_model: Evaluate model performance with metrics (accuracy, precision, recall, F1, R²)

IMPORTANT: You MUST follow the ReAct format exactly. Use this structure:

Thought: [Your reasoning about what to do]
Action: [The tool name to use, must be one of: {tool_names}]
Action Input: [The input string for the tool]

STOP HERE! Do NOT write "Observation:" - the system will provide it automatically.

After the system provides the Observation, you can continue with:
Thought: I now know the final answer
Final Answer: [Your comprehensive ML analysis with model performance, feature importance, and recommendations]

EXAMPLES:

Example 1 - First Step (you generate this):
Task: Train a machine learning model on the dataset at data/titanic.csv based on this request: Predict survival using Random Forest
Thought: I need to train a Random Forest model to predict survival. This is a binary classification problem. I'll use the train_random_forest_model tool with survived as the target column.
Action: train_random_forest_model
Action Input: json_string='{{"file_path": "data/titanic.csv", "target_column": "survived", "feature_columns": "pclass,age,fare,sex", "task_type": "classification"}}'

Example 1 - After Observation (you continue):
Observation: Random Forest model trained: {'model_type': 'Random Forest classification', 'features': ['pclass', 'age', 'fare', 'sex'], 'target': 'survived', 'train_size': 712, 'test_size': 179, 'accuracy': 0.8212, 'precision': 0.8145, 'recall': 0.8212, 'f1_score': 0.8156, 'cv_score_mean': 0.8022, 'cv_score_std': 0.0234, 'feature_importance': {'pclass': 0.2841, 'age': 0.1823, 'fare': 0.2156, 'sex': 0.3180}, 'model_saved': 'data/titanic_rf_classification_model.joblib'}
Thought: I now know the final answer
Final Answer: Successfully trained a Random Forest classifier for survival prediction with 82.12% accuracy. The model shows:

Performance Metrics:
- Accuracy: 82.12% - The model correctly predicts survival 82% of the time
- Precision: 81.45% - When predicting survival, 81% are correct
- Recall: 82.12% - The model identifies 82% of actual survivors
- F1 Score: 81.56% - Balanced performance metric
- Cross-validation: 80.22% ± 2.34%

Most Important Features:
1. Sex (31.80%) - Gender is the strongest predictor
2. Pclass (28.41%) - Passenger class significantly affects survival
3. Fare (21.56%) - Ticket price indicates class and survival chance
4. Age (18.23%) - Age influenced evacuation priority

Recommendations:
- Model performs well with 82% accuracy
- Gender is the most important feature for survival prediction
- Consider feature engineering (family size, title extraction)
- Cross-validation shows consistent performance

Example 2 - First Step (you generate this):
Task: Train a regression model on housing data at data/houses.csv to predict prices
Thought: This is a regression problem for predicting continuous values (prices). I'll use the train_regression_model tool with Price as the target column.
Action: train_regression_model
Action Input: json_string='{{"file_path": "data/houses.csv", "target_column": "Price", "feature_columns": "Bedrooms,Bathrooms,SquareFeet,YearBuilt", "model_type": "linear"}}'

NOW, answer the user's task following the exact format above.

Available tools: {tool_names}
{tools}

Task: {input}

{agent_scratchpad}""",
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        # Create the agent with strict configuration
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)

        # Create agent executor with strict configuration
        # NO handle_parsing_errors - we want strict format compliance
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=False,  # NO fallback - enforce strict format
            # early_stopping_method="generate",  # Stop early if we have an answer
            return_intermediate_steps=True  # For debugging
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