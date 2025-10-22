"""Planning agent that breaks down complex prompts into steps."""

from typing import List, Dict, Any
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.llm_wrapper import llm_wrapper


class StrictReActOutputParser(ReActSingleInputOutputParser):
    """
    Strict ReAct output parser that enforces exact format compliance.
    NO FALLBACK - will raise errors if format is incorrect.
    """

    def parse(self, text: str) -> AgentAction | AgentFinish:
        """
        Parse LLM output strictly according to ReAct format.

        Expected format:
        Thought: <reasoning>
        Action: <action_name>
        Action Input: <action_input>

        OR

        Thought: I now know the final answer
        Final Answer: <final_answer>
        """
        # Check for Final Answer first
        if "Final Answer:" in text:
            # Extract the final answer
            final_answer_pattern = r"Final Answer:\s*(.*)"
            match = re.search(final_answer_pattern, text, re.DOTALL)
            if match:
                return AgentFinish(
                    return_values={"output": match.group(1).strip()},
                    log=text
                )
            else:
                raise OutputParserException(
                    f"Could not parse Final Answer from text: {text}\n"
                    f"Expected format: 'Final Answer: <answer>'"
                )

        # Check for Thought/Action/Action Input format
        thought_pattern = r"Thought:\s*(.*?)\nAction:\s*(.*?)\nAction Input:\s*(.*?)(?:\n|$)"
        match = re.search(thought_pattern, text, re.DOTALL)

        if not match:
            # Be very explicit about what's wrong
            error_msg = f"Output does not match required ReAct format.\n\n"
            error_msg += f"Expected format:\n"
            error_msg += f"Thought: <your reasoning>\n"
            error_msg += f"Action: <action_name>\n"
            error_msg += f"Action Input: <action_input>\n\n"
            error_msg += f"Received text:\n{text}\n\n"

            # Check what parts are missing
            if "Thought:" not in text:
                error_msg += "ERROR: Missing 'Thought:' at the beginning\n"
            if "Action:" not in text:
                error_msg += "ERROR: Missing 'Action:' after Thought\n"
            if "Action Input:" not in text:
                error_msg += "ERROR: Missing 'Action Input:' after Action\n"

            raise OutputParserException(error_msg)

        thought = match.group(1).strip()
        action = match.group(2).strip()
        action_input = match.group(3).strip()

        # Validate that action and action_input are not empty
        if not action:
            raise OutputParserException(
                f"Action cannot be empty. Text: {text}"
            )
        if not action_input:
            raise OutputParserException(
                f"Action Input cannot be empty. Text: {text}"
            )

        return AgentAction(tool=action, tool_input=action_input, log=text)


@tool
def create_execution_plan(user_prompt: str) -> str:
    """
    Analyze user prompt and create a step-by-step execution plan.

    Args:
        user_prompt: The user's request for data analysis

    Returns:
        String containing the execution plan with numbered steps
    """
    try:
        # Simple planning logic - in a real implementation, this could use an LLM
        plan_steps = []
        prompt_lower = user_prompt.lower()

        # Check if data needs to be read first
        if any(keyword in prompt_lower for keyword in ['dataset', 'data', 'file', 'csv', 'load']):
            plan_steps.append("1. Read and examine the dataset using DataReaderAgent")

        # Check for data manipulation needs
        if any(keyword in prompt_lower for keyword in ['clean', 'missing', 'null', 'dummy', 'encode', 'transform']):
            plan_steps.append(f"{len(plan_steps)+1}. Handle data preprocessing using DataManipulationAgent")

        # Check for data operations
        if any(keyword in prompt_lower for keyword in ['filter', 'group', 'aggregate', 'calculate', 'sum', 'mean', 'count']):
            plan_steps.append(f"{len(plan_steps)+1}. Perform data operations using DataOperationsAgent")

        # Check for ML tasks
        if any(keyword in prompt_lower for keyword in ['model', 'predict', 'train', 'classification', 'regression', 'machine learning', 'ml']):
            plan_steps.append(f"{len(plan_steps)+1}. Train and evaluate ML model using MLPredictionAgent")

        # If no specific tasks identified, default to data exploration
        if not plan_steps:
            plan_steps = [
                "1. Read and examine the dataset using DataReaderAgent",
                "2. Analyze data characteristics and patterns"
            ]

        plan = "\n".join(plan_steps)
        return f"Execution Plan for: '{user_prompt}'\n{plan}"

    except Exception as e:
        return f"Error creating execution plan: {str(e)}"


class PlannerAgent:
    """Agent responsible for planning and coordinating the overall workflow."""
    
    def __init__(self):
        self.llm = llm_wrapper.get_llm_for_agent("planner")
        # self.tools = [create_execution_plan]
        self.tools = []
        
        # Create the planning prompt with explicit ReAct format
        self.prompt = PromptTemplate(
            template="""You are a Data Analysis Planning Agent. Your job is to understand user requests for data analysis and break them down into clear, executable steps.

You have access to the following specialized agents:
- DataReaderAgent: Reads CSV/JSON files, provides column information, data summaries, and previews
- DataManipulationAgent: Handles missing values, creates dummy variables, transforms data types
- DataOperationsAgent: Filters data, performs mathematical operations, aggregates data
- MLPredictionAgent: Trains machine learning models (regression, SVM, random forest, KNN)

IMPORTANT: You MUST follow the ReAct format exactly. Use this structure:

Thought: [Your reasoning about what to do]
Action: [The tool name to use, must be one of: {tool_names}]
Action Input: [The input string for the tool]

STOP HERE! Do NOT write "Observation:" - the system will provide it automatically.

After the system provides the Observation, you can continue with:
Thought: I now know the final answer
Final Answer: [Your final execution plan]

EXAMPLES:

Example 1 - First Step (you generate this):
User Request: Show me basic information about the dataset
Thought: The user wants basic information about a dataset. I should use the create_execution_plan tool to analyze this request.
Action: create_execution_plan
Action Input: Show me basic information about the dataset

(Then the system will run the tool and provide the Observation)

Example 1 - After Observation (you continue):
Observation: Execution Plan for: 'Show me basic information about the dataset'
1. Read and examine the dataset using DataReaderAgent
Thought: I now know the final answer
Final Answer: Execution Plan:
1. Read and examine the dataset using DataReaderAgent

Example 2 - First Step (you generate this):
User Request: Clean the data and train a model
Thought: This request involves multiple steps: data cleaning and model training. I need to create a plan.
Action: create_execution_plan
Action Input: Clean the data and train a model

(Then the system will run the tool and provide the Observation)

NOW, answer the user's request following the exact format above.

Available tools: {tool_names}
{tools}

User Request: {input}

{agent_scratchpad}""",
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        # Create the agent with custom strict parser
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
    
    def plan(self, user_prompt: str) -> Dict[str, Any]:
        """
        Create an execution plan for the user's request.
        
        Args:
            user_prompt: User's data analysis request
            
        Returns:
            Dictionary containing the execution plan
        """
        try:
            result = self.agent_executor.invoke({"input": user_prompt})
            
            return {
                "status": "success",
                "user_prompt": user_prompt,
                "plan": result.get("output", ""),
                "agent_type": "planner"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_type": "planner"
            }