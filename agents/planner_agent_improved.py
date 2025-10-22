"""Improved Planning agent with LLM-based JSON structured planning."""

from typing import List, Dict, Any
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import tool
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.llm_wrapper import llm_wrapper


@tool
def create_structured_plan(user_prompt: str) -> str:
    """
    Create a structured JSON execution plan based on user request.
    This tool helps organize the analysis into clear steps.

    Args:
        user_prompt: The user's request for data analysis

    Returns:
        JSON string containing structured execution plan
    """
    # This is a placeholder - the actual LLM will generate the plan
    # The LLM will be instructed to output JSON directly in its Final Answer
    return f"Planning for: {user_prompt}"


class ImprovedPlannerAgent:
    """
    Improved Planning Agent that uses LLM to generate structured JSON plans.

    The agent analyzes user requests and creates a detailed execution plan with:
    - Specific agent assignments
    - Input parameters for each step
    - Clear task descriptions
    - Proper sequencing
    """

    def __init__(self):
        self.llm = llm_wrapper.get_llm_for_agent("planner")
        self.tools = []  # No tools needed - LLM generates plan directly

        # Create enhanced planning prompt
        self.prompt = PromptTemplate(
            template="""You are an Intelligent Data Analysis Planning Agent. Your job is to analyze user requests and create a detailed, structured execution plan in JSON format.

Available Specialized Agents:
1. **DataReaderAgent**: Reads CSV/JSON files, provides column info, data summaries, and previews
   - Use when: Need to understand dataset structure, explore data, get statistics
   - Input: file_path, analysis_type (comprehensive/basic/columns_only)

2. **DataManipulationAgent**: Handles missing values, creates dummy variables, transforms data types
   - Use when: Need data cleaning, encoding, type conversion, imputation
   - Input: file_path, manipulation_request (description of what to do)

3. **DataOperationsAgent**: Filters data, performs math operations, aggregates data
   - Use when: Need calculations, filtering, grouping, statistical operations
   - Input: file_path, operation_request (description of operations)

4. **MLPredictionAgent**: Trains ML models (regression, SVM, random forest, KNN)
   - Use when: Need to build predictive models, train algorithms, evaluate performance
   - Input: file_path, ml_request (description of ML task)

Your task is to create a JSON plan with the following structure:

{{
  "plan_description": "Brief overall description of what will be done",
  "steps": [
    {{
      "step_number": 1,
      "agent_name": "DataReaderAgent",
      "input_parameters": {{
        "file_path": "data/titanic.csv",
        "analysis_type": "comprehensive"
      }},
      "task_description": "Load and analyze the dataset structure",
      "reasoning": "First, we need to understand the data before proceeding"
    }},
    {{
      "step_number": 2,
      "agent_name": "DataManipulationAgent",
      "input_parameters": {{
        "file_path": "data/titanic.csv",
        "manipulation_request": "Handle missing values in Age column using median imputation"
      }},
      "task_description": "Clean the data by handling missing values",
      "reasoning": "Missing values must be handled before analysis"
    }}
  ]
}}

IMPORTANT INSTRUCTIONS:
1. Analyze the user's request carefully to understand their goals
2. Break down complex requests into logical sequential steps
3. Choose the most appropriate agents for each task
4. Provide specific input parameters for each agent
5. Ensure proper sequencing (e.g., read data before manipulating it)
6. Include clear reasoning for each step

CRITICAL: Your Final Answer MUST be ONLY the valid JSON plan. Do NOT include any other text, explanations, or markdown formatting. Just the pure JSON object.

EXAMPLES:

Example 1:
User Request: "Show me basic information about the Titanic dataset"
Thought: This is a simple data exploration request. I need to use the DataReaderAgent to analyze the dataset.
Final Answer: {{"plan_description": "Explore the Titanic dataset structure and basic information", "steps": [{{"step_number": 1, "agent_name": "DataReaderAgent", "input_parameters": {{"file_path": "data/titanic.csv", "analysis_type": "comprehensive"}}, "task_description": "Analyze dataset structure, columns, data types, and basic statistics", "reasoning": "User wants comprehensive overview of the dataset"}}]}}

Example 2:
User Request: "Clean the data and train a classification model to predict survival"
Thought: This requires multiple steps: read data, clean it, and train a model. I'll create a comprehensive plan.
Final Answer: {{"plan_description": "Complete ML workflow for survival prediction", "steps": [{{"step_number": 1, "agent_name": "DataReaderAgent", "input_parameters": {{"file_path": "data/titanic.csv", "analysis_type": "comprehensive"}}, "task_description": "Analyze dataset and identify data quality issues", "reasoning": "Need to understand data before cleaning"}}, {{"step_number": 2, "agent_name": "DataManipulationAgent", "input_parameters": {{"file_path": "data/titanic.csv", "manipulation_request": "Handle missing values in Age and Embarked columns, create dummy variables for Sex and Embarked"}}, "task_description": "Clean data and encode categorical variables", "reasoning": "ML models require clean numeric data"}}, {{"step_number": 3, "agent_name": "MLPredictionAgent", "input_parameters": {{"file_path": "data/titanic.csv", "ml_request": "Train a Random Forest classification model to predict Survived column"}}, "task_description": "Train and evaluate survival prediction model", "reasoning": "User wants a classification model for survival prediction"}}]}}

NOW, create a structured JSON plan for the following user request. Remember: Output ONLY the JSON, nothing else.

User Request: {input}

{agent_scratchpad}""",
            input_variables=["input", "agent_scratchpad"]
        )

        # Create the agent
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

    def plan(self, user_prompt: str) -> Dict[str, Any]:
        """
        Create a structured execution plan for the user's request.

        Args:
            user_prompt: User's data analysis request

        Returns:
            Dictionary containing structured plan with steps
        """
        try:
            result = self.agent_executor.invoke({"input": user_prompt})
            plan_output = result.get("output", "")

            # Try to parse the JSON plan
            try:
                # Clean the output - sometimes LLM adds extra text
                plan_output = plan_output.strip()

                # Find JSON object in the output
                start_idx = plan_output.find('{')
                end_idx = plan_output.rfind('}') + 1

                if start_idx != -1 and end_idx > start_idx:
                    json_str = plan_output[start_idx:end_idx]
                    structured_plan = json.loads(json_str)

                    # Validate plan structure
                    if not isinstance(structured_plan, dict):
                        raise ValueError("Plan must be a dictionary")
                    if "steps" not in structured_plan:
                        raise ValueError("Plan must contain 'steps' field")

                    return {
                        "status": "success",
                        "user_prompt": user_prompt,
                        "plan": json.dumps(structured_plan, indent=2),
                        "structured_plan": structured_plan,
                        "agent_type": "planner"
                    }
                else:
                    raise ValueError("No JSON object found in output")

            except (json.JSONDecodeError, ValueError) as e:
                # Fallback: create a simple plan if JSON parsing fails
                print(f"⚠️ Warning: Could not parse JSON plan: {e}")
                print(f"Output was: {plan_output}")

                # Create a basic fallback plan
                fallback_plan = {
                    "plan_description": f"Analysis for: {user_prompt}",
                    "steps": [
                        {
                            "step_number": 1,
                            "agent_name": "DataReaderAgent",
                            "input_parameters": {
                                "file_path": "data/titanic.csv",
                                "analysis_type": "comprehensive"
                            },
                            "task_description": "Analyze the dataset",
                            "reasoning": "Start with data exploration"
                        }
                    ]
                }

                return {
                    "status": "success",
                    "user_prompt": user_prompt,
                    "plan": json.dumps(fallback_plan, indent=2),
                    "structured_plan": fallback_plan,
                    "agent_type": "planner",
                    "warning": f"Used fallback plan due to JSON parsing error: {e}"
                }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_type": "planner"
            }

    def replan_with_error_context(self, user_prompt: str, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new plan after an agent failure, incorporating error context.

        Args:
            user_prompt: Original user request
            error_context: Information about what went wrong

        Returns:
            Dictionary containing new structured plan
        """
        try:
            # Create enhanced prompt with error context
            enhanced_prompt = f"""Original Request: {user_prompt}

PREVIOUS ATTEMPT FAILED:
- Failed Agent: {error_context.get('failed_agent', 'unknown')}
- Error: {error_context.get('error', 'unknown error')}
- Failed Step: {error_context.get('failed_step', 'unknown')}

Please create a NEW plan that addresses this error. Consider:
1. What went wrong and why
2. Alternative approaches to accomplish the goal
3. Steps to verify data before proceeding
4. Fallback strategies if needed

Create a revised JSON plan that avoids the previous error."""

            return self.plan(enhanced_prompt)

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_type": "planner"
            }
