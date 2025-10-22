"""Data reading and analysis agent."""

from typing import List, Dict, Any
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.llm_wrapper import llm_wrapper
from tools.data_tools import read_csv_file, read_json_file, get_column_info, get_data_summary, preview_data


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
        
        # Create the data reader prompt with explicit ReAct format
        self.prompt = PromptTemplate(
            template="""You are a Data Reading Agent specialized in reading and analyzing datasets. Your job is to help users understand their data by providing detailed information about files, columns, and data characteristics.

You have access to these tools:
- read_csv_file: Read CSV files and get basic information
- read_json_file: Read JSON files and convert to DataFrame
- get_column_info: Get detailed column information (types, missing values, unique values)
- get_data_summary: Generate statistical summaries (mean, median, std, etc.)
- preview_data: Show sample rows from the dataset

IMPORTANT: You MUST follow the ReAct format exactly. Use this structure:

Thought: [Your reasoning about what to do]
Action: [The tool name to use, must be one of: {tool_names}]
Action Input: [The input string for the tool]

STOP HERE! Do NOT write "Observation:" - the system will provide it automatically.

After the system provides the Observation, you can continue with:
Thought: I now know the final answer
Final Answer: [Your comprehensive data analysis]

EXAMPLES:

Example 1 - First Step (you generate this):
Task: Perform a comprehensive analysis of the dataset at data/titanic.csv
Thought: I need to start by reading the CSV file to understand the basic structure of the dataset.
Action: read_csv_file
Action Input: data/titanic.csv

(Then the system will run the tool and provide the Observation)

Example 1 - After Observation (you continue):
Observation: Successfully read CSV file: data/titanic.csv
Shape: (891, 12) - 891 rows and 12 columns
Columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
Thought: Now I have the basic file information. I should get detailed column information to understand data types and missing values.
Action: get_column_info
Action Input: data/titanic.csv

(Then the system will provide another Observation, and you continue this process)

Example 2 - First Step (you generate this):
Task: Provide basic information about the dataset at uploads/mydata.csv
Thought: I'll start by reading the CSV file to get an overview of the dataset structure.
Action: read_csv_file
Action Input: uploads/mydata.csv

(Then the system will run the tool and provide the Observation)

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