"""Data operations agent."""

from typing import List, Dict, Any
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.llm_wrapper import llm_wrapper
# Import JSON wrapper tools that use StructuredTool with single string parameter
# from tools.json_wrapper_tools import filter_data, perform_math_operations, string_operations, aggregate_data
# from tools.operations_tools import filter_data, perform_math_operations, string_operations, aggregate_data
from tools.tool_react import filter_data, perform_math_operations, string_operations, aggregate_data

class DataOperationsAgent:
    """Agent specialized in data operations and analysis."""
    
    def __init__(self):
        self.llm = llm_wrapper.get_llm_for_agent("data_operations")
        self.tools = [
            filter_data,
            perform_math_operations,
            string_operations,
            aggregate_data
        ]
        
        # Create the data operations prompt with explicit ReAct format
        self.prompt = PromptTemplate(
            template="""You are a Data Operations Agent specialized in performing calculations, filtering, and analytical operations on data. Your job is to help users extract insights from their data through various operations.

You have access to these tools:
- filter_data: Filter rows based on conditions (equals, greater_than, less_than, contains, etc.)
- perform_math_operations: Apply math operations (add, subtract, multiply, divide, power, sqrt, log, abs)
- string_operations: String manipulations (upper, lower, length, split, replace, contains_count)
- aggregate_data: Group by columns and calculate statistics (mean, sum, count, min, max, std, median)

IMPORTANT: You MUST follow the ReAct format exactly. Use this structure:

Thought: [Your reasoning about what to do]
Action: [The tool name to use, must be one of: {tool_names}]
Action Input: [The input string for the tool]

STOP HERE! Do NOT write "Observation:" - the system will provide it automatically.

After the system provides the Observation, you can continue with:
Thought: I now know the final answer
Final Answer: [Your comprehensive analysis with insights and business meaning]

EXAMPLES:

Example 1 - First Step (you generate this):
Task: Perform the following data operations on the dataset at data/titanic.csv: Calculate average fare by passenger class
Thought: I need to aggregate the data by passenger class (Pclass) and calculate the mean of the Fare column.
Action: aggregate_data
Action Input: json_string='{{"file_path": "data/titanic.csv", "group_by_columns": "pclass", "agg_column": "fare", "agg_function": "mean"}}'

Example 1 - After Observation (you continue):
Observation: Aggregated data by ['pclass'] using mean on 'fare'. Result shape: (3, 2). Saved to: data/titanic_aggregated.csv
Thought: I now know the final answer
Final Answer: The average fare varies significantly by passenger class:
- First Class: Higher fares for premium accommodations
- Second Class: Moderate fares
- Third Class: Lower fares for basic accommodations
This shows a clear correlation between passenger class and ticket price.

Example 2 - First Step (you generate this):
Task: Filter passengers who survived in data/titanic.csv
Thought: I need to filter the data where the survived column equals 1.
Action: filter_data
Action Input: json_string='{{"file_path": "data/titanic.csv", "column_name": "survived", "condition": "equals", "value": 1}}'

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
    
    def perform_operations(self, file_path: str, operation_request: str) -> Dict[str, Any]:
        """
        Perform data operations based on user request.
        
        Args:
            file_path: Path to the data file
            operation_request: Description of what operations to perform
            
        Returns:
            Dictionary containing the operation results
        """
        try:
            task = f"Perform the following data operations on the dataset at {file_path}: {operation_request}"
            
            result = self.agent_executor.invoke({"input": task})
            
            return {
                "status": "success",
                "file_path": file_path,
                "request": operation_request,
                "result": result.get("output", ""),
                "agent_type": "data_operations"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_type": "data_operations"
            }
    
    def analyze_patterns(self, file_path: str, analysis_focus: str) -> Dict[str, Any]:
        """
        Analyze patterns in the data.
        
        Args:
            file_path: Path to the data file
            analysis_focus: What to focus the analysis on
            
        Returns:
            Dictionary containing the pattern analysis results
        """
        try:
            task = f"Analyze patterns in the dataset at {file_path} focusing on: {analysis_focus}. Use filtering, aggregation, and mathematical operations to uncover insights."
            
            result = self.agent_executor.invoke({"input": task})
            
            return {
                "status": "success",
                "file_path": file_path,
                "focus": analysis_focus,
                "analysis": result.get("output", ""),
                "agent_type": "data_operations"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_type": "data_operations"
            }