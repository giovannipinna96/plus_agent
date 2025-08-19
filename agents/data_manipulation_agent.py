"""Data manipulation agent."""

from typing import List, Dict, Any
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from plus_agent.core.llm_wrapper import llm_wrapper
from plus_agent.tools.manipulation_tools import create_dummy_variables, modify_column_values, handle_missing_values, convert_data_types


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
        
        # Create the data manipulation prompt
        self.prompt = PromptTemplate(
            template="""You are a Data Manipulation Agent specialized in preprocessing and transforming data. Your job is to clean, transform, and prepare data for analysis or machine learning.

You can:
- Create dummy variables for categorical data
- Modify column values (multiply, add, subtract, divide, replace, normalize, standardize)
- Handle missing values (drop, impute with mean/median/mode, forward/backward fill)
- Convert data types (int, float, string, category, datetime)

Always explain what transformations you're applying and why they might be useful for the analysis.

Available tools: {tool_names}
{tools}

Task: {input}

{agent_scratchpad}

Perform the requested data manipulation and explain the results.""",
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
            task = f"Perform the following data manipulation on the dataset at {file_path}: {manipulation_request}"
            
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