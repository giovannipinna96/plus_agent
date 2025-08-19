"""Data operations agent."""

from typing import List, Dict, Any
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from plus_agent.core.llm_wrapper import llm_wrapper
from plus_agent.tools.operations_tools import filter_data, perform_math_operations, string_operations, aggregate_data


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
        
        # Create the data operations prompt
        self.prompt = PromptTemplate(
            template="""You are a Data Operations Agent specialized in performing calculations, filtering, and analytical operations on data. Your job is to help users extract insights from their data through various operations.

You can:
- Filter data based on conditions (equals, greater than, less than, contains, etc.)
- Perform mathematical operations (add, subtract, multiply, divide, power, sqrt, log, abs)
- Execute string operations (upper, lower, length, split, replace, contains count)
- Aggregate data (group by columns and calculate mean, sum, count, min, max, std, median)

Always provide clear explanations of the operations performed and their business meaning.

Available tools: {tool_names}
{tools}

Task: {input}

{agent_scratchpad}

Perform the requested data operations and provide meaningful insights.""",
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