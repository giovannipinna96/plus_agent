"""Data manipulation agent."""

from typing import List, Dict, Any
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.llm_wrapper import llm_wrapper
# Import JSON wrapper tools that use StructuredTool with single string parameter
# from tools.json_wrapper_tools import create_dummy_variables, modify_column_values, handle_missing_values, convert_data_types
# from tools.manipulation_tools import create_dummy_variables, modify_column_values, handle_missing_values, convert_data_types
from tools.tool_react import create_dummy_variables, modify_column_values, handle_missing_values, convert_data_types

class DataManipulationAgent:
    """Agent specialized in data preprocessing and manipulation."""
    
    def __init__(self):
        self.llm = llm_wrapper.get_llm_for_agent("data_manipulation")
        self.tools = [
            create_dummy_variables,
            modify_column_values,
            handle_missing_values,
            convert_data_types
        ]
        
        # Create the data manipulation prompt with explicit ReAct format
        self.prompt = PromptTemplate(
            template="""You are a Data Manipulation Agent specialized in preprocessing and transforming data. Your job is to clean, transform, and prepare data for analysis or machine learning.

You have access to these tools:
- create_dummy_variables: Create one-hot encoded dummy variables for categorical columns
- modify_column_values: Apply transformations (multiply, add, divide, normalize, standardize, replace)
- handle_missing_values: Handle missing data (drop, impute with mean/median/mode, forward/backward fill)
- convert_data_types: Convert column data types (int, float, string, category, datetime)

IMPORTANT: You MUST follow the ReAct format exactly. Use this structure:

Thought: [Your reasoning about what to do]
Action: [The tool name to use, must be one of: {tool_names}]
Action Input: {{"parameter_name": "value", ...}}

CRITICAL: Action Input must be a valid JSON object with the exact parameter names required by the tool.

STOP HERE! Do NOT write "Observation:" - the system will provide it automatically.

After the system provides the Observation, you can continue with:
Thought: I now know the final answer
Final Answer: [Your comprehensive manipulation results and explanation]

EXAMPLES:

Example 1 - First Step (you generate this):
Task: Perform the following data manipulation on the dataset at /tmp/data/titanic.csv: Handle missing values in Age column
Thought: I need to handle missing values in the Age column. I'll use the handle_missing_values tool to impute with median.
Action: handle_missing_values
Action Input: {{"file_path": "/tmp/data/titanic.csv", "column_name": "Age", "method": "median"}}

Example 1 - After Observation (you continue):
Observation: Handled 177 missing values in column 'Age' using method 'median'. Remaining missing: 0. Saved to: /tmp/data/titanic_missing_handled.csv
Thought: I now know the final answer
Final Answer: Successfully handled missing values in the Age column. The median imputation method was used to fill 177 missing values. This method is appropriate because Age is numerical and the median is robust to outliers.

Example 2 - First Step (you generate this):
Task: Create dummy variables for the Sex column in /tmp/data/titanic.csv
Thought: I need to create one-hot encoded dummy variables for the categorical Sex column.
Action: create_dummy_variables
Action Input: {{"file_path": "/tmp/data/titanic.csv", "column_name": "Sex"}}

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
            return_intermediate_steps=True  # For debugging
        )
    
    def manipulate_data(self, file_path: str, manipulation_request: str) -> Dict[str, Any]:
        """
        Perform data manipulation based on user request.
        
        Args:
            file_path: Path to the data file
            manipulation_request: Description of what manipulation to perform
            
        Returns:
            Dictionary containing the manipulation results
        """
        try:
            task = f"Perform the following data manipulation task {manipulation_request}"
            # task = f"Perform the following data manipulation on the dataset at {file_path}: {manipulation_request}"
            
            result = self.agent_executor.invoke({"input": task})
            
            return {
                "status": "success",
                "file_path": file_path,
                "request": manipulation_request,
                "result": result.get("output", ""),
                "agent_type": "data_manipulation"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_type": "data_manipulation"
            }
    
    def prepare_data_for_ml(self, file_path: str, target_column: str) -> Dict[str, Any]:
        """
        Prepare data specifically for machine learning.
        
        Args:
            file_path: Path to the data file
            target_column: Name of the target variable column
            
        Returns:
            Dictionary containing the preparation results
        """
        try:
            task = f"Prepare the dataset at {file_path} for machine learning with '{target_column}' as the target variable. Handle missing values, encode categorical variables, and ensure proper data types."
            
            result = self.agent_executor.invoke({"input": task})
            
            return {
                "status": "success",
                "file_path": file_path,
                "target_column": target_column,
                "preparation": result.get("output", ""),
                "agent_type": "data_manipulation"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_type": "data_manipulation"
            }