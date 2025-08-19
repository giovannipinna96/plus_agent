"""Planning agent that breaks down complex prompts into steps."""

from typing import List, Dict, Any
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from plus_agent.core.llm_wrapper import llm_wrapper


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
        self.tools = [create_execution_plan]
        
        # Create the planning prompt
        self.prompt = PromptTemplate(
            template="""You are a Data Analysis Planning Agent. Your job is to understand user requests for data analysis and break them down into clear, executable steps.

You have access to the following specialized agents:
- DataReaderAgent: Reads CSV/JSON files, provides column information, data summaries, and previews
- DataManipulationAgent: Handles missing values, creates dummy variables, transforms data types
- DataOperationsAgent: Filters data, performs mathematical operations, aggregates data
- MLPredictionAgent: Trains machine learning models (regression, SVM, random forest, KNN)

Given a user request, analyze what needs to be done and create a step-by-step plan.

Available tools: {tool_names}
{tools}

User Request: {input}

{agent_scratchpad}

Think step by step about what the user wants to accomplish and create a clear execution plan.""",
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        # Create the agent
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=3
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