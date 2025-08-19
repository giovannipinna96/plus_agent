"""Data reading and analysis agent."""

from typing import List, Dict, Any
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from plus_agent.core.llm_wrapper import llm_wrapper
from plus_agent.tools.data_tools import read_csv_file, read_json_file, get_column_info, get_data_summary, preview_data


class DataReaderAgent:
    """Agent specialized in reading and analyzing data files."""
    
    def __init__(self):
        self.llm = llm_wrapper.get_llm_for_agent("data_reader")
        self.tools = [
            read_csv_file,
            read_json_file, 
            get_column_info,
            get_data_summary,
            preview_data
        ]
        
        # Create the data reader prompt
        self.prompt = PromptTemplate(
            template="""You are a Data Reading Agent specialized in reading and analyzing datasets. Your job is to help users understand their data by providing detailed information about files, columns, and data characteristics.

You can:
- Read CSV and JSON files
- Provide column information (types, missing values, unique values)
- Generate data summaries with statistics
- Preview data samples
- Identify data quality issues

Always be thorough in your analysis and provide actionable insights about the data.

Available tools: {tool_names}
{tools}

Task: {input}

{agent_scratchpad}

Provide a comprehensive analysis of the requested data.""",
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
    
    def analyze_data(self, file_path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze a data file.
        
        Args:
            file_path: Path to the data file
            analysis_type: Type of analysis (comprehensive, basic, columns_only)
            
        Returns:
            Dictionary containing the analysis results
        """
        try:
            if analysis_type == "comprehensive":
                task = f"Perform a comprehensive analysis of the dataset at {file_path}. Include file information, column details, data summary, and data preview."
            elif analysis_type == "basic":
                task = f"Provide basic information about the dataset at {file_path}."
            elif analysis_type == "columns_only":
                task = f"Analyze only the columns of the dataset at {file_path}."
            else:
                task = f"Analyze the dataset at {file_path} focusing on {analysis_type}."
            
            result = self.agent_executor.invoke({"input": task})
            
            return {
                "status": "success",
                "file_path": file_path,
                "analysis": result.get("output", ""),
                "agent_type": "data_reader"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_type": "data_reader"
            }
    
    def answer_data_question(self, file_path: str, question: str) -> Dict[str, Any]:
        """
        Answer a specific question about the data.
        
        Args:
            file_path: Path to the data file
            question: Specific question about the data
            
        Returns:
            Dictionary containing the answer
        """
        try:
            task = f"Answer this question about the dataset at {file_path}: {question}"
            
            result = self.agent_executor.invoke({"input": task})
            
            return {
                "status": "success",
                "file_path": file_path,
                "question": question,
                "answer": result.get("output", ""),
                "agent_type": "data_reader"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_type": "data_reader"
            }