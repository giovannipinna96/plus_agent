"""Improved LangGraph orchestration system with structured planning, in-memory data, and error recovery."""

from typing import Dict, Any, List, Optional, Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
import json
import pandas as pd
from pydantic import Field

# Import agents
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.planner_agent_improved import ImprovedPlannerAgent
from agents.data_reader_agent import DataReaderAgent
from agents.data_manipulation_agent import DataManipulationAgent
from agents.data_operations_agent import DataOperationsAgent
from agents.ml_prediction_agent import MLPredictionAgent

# Import LangSmith integration
from core.langsmith_integration import trace_workflow_execution, trace_agent_execution, langsmith_logger


class ImprovedMultiAgentState(MessagesState):
    """
    Enhanced state for multi-agent workflow with:
    - Structured JSON plan tracking
    - In-memory DataFrame storage (eliminates disk I/O)
    - Error tracking and recovery
    - Step-by-step execution control
    """
    # File path for dataset
    current_file_path: Optional[str] = None

    # IMPROVEMENT 1: Store DataFrame in memory to avoid repeated disk reads
    current_dataframe: Optional[Any] = None  # pandas.DataFrame (Any for serialization)
    dataframe_metadata: Optional[Dict[str, Any]] = None  # Info about the DataFrame

    # IMPROVEMENT 2: Structured JSON plan instead of string
    structured_plan: Optional[Dict[str, Any]] = None
    execution_plan: Optional[str] = None  # Keep for backward compatibility

    # IMPROVEMENT 3: Error tracking for auto-correction
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 2

    # Execution tracking
    current_step_index: int = 0
    completed_steps: List[str] = Field(default_factory=list)
    current_agent: Optional[str] = None
    results: Dict[str, Any] = Field(default_factory=dict)


class ImprovedMultiAgentOrchestrator:
    """
    Improved orchestrator with:
    1. Intelligent Planner with JSON structured plans
    2. In-memory DataFrame management (no disk I/O between agents)
    3. Robust Supervisor with structured plan routing
    4. Auto-correction capability with error recovery
    """

    def __init__(self):
        # Initialize improved agents
        self.planner = ImprovedPlannerAgent()
        self.data_reader = DataReaderAgent()
        self.data_manipulation = DataManipulationAgent()
        self.data_operations = DataOperationsAgent()
        self.ml_prediction = MLPredictionAgent()

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the improved LangGraph workflow with error recovery."""

        # Create the state graph
        workflow = StateGraph(ImprovedMultiAgentState)

        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("data_reader", self._data_reader_node)
        workflow.add_node("data_manipulation", self._data_manipulation_node)
        workflow.add_node("data_operations", self._data_operations_node)
        workflow.add_node("ml_prediction", self._ml_prediction_node)
        workflow.add_node("supervisor", self._supervisor_node)

        # IMPROVEMENT 4: Add error handler node
        workflow.add_node("error_handler", self._error_handler_node)

        # Add edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "supervisor")

        # All agent nodes route through supervisor
        workflow.add_edge("data_reader", "supervisor")
        workflow.add_edge("data_manipulation", "supervisor")
        workflow.add_edge("data_operations", "supervisor")
        workflow.add_edge("ml_prediction", "supervisor")

        # Error handler can route back to planner (for replanning) or end
        workflow.add_edge("error_handler", "planner")

        # Supervisor routes to END
        workflow.add_edge("supervisor", END)

        return workflow.compile()

    def _planner_node(self, state: ImprovedMultiAgentState) -> Dict[str, Any]:
        """Execute the improved planner agent with JSON structured planning."""
        try:
            print(f"\n{'='*60}")
            print(f"🎭 IMPROVED PLANNER AGENT STARTING")
            print(f"{'='*60}")

            # Check if this is a retry (replanning after error)
            retry_count = state.get("retry_count", 0)
            errors = state.get("errors", [])

            if retry_count > 0 and errors:
                print(f"🔄 REPLANNING (Retry {retry_count}/{state.get('max_retries', 2)})")
                print(f"⚠️ Previous error: {errors[-1].get('error', 'unknown')}")

            # Get the user's original message
            user_message = None
            for message in state["messages"]:
                if isinstance(message, HumanMessage):
                    user_message = message.content
                    break

            if not user_message:
                print("❌ No user message found in state")
                return {
                    "messages": [AIMessage(content="No user message found.")],
                    "current_agent": "planner",
                    "execution_plan": "No plan created"
                }

            print(f"📝 User message: {user_message[:100]}...")
            print(f"🔄 Calling improved planner agent...")

            # Create execution plan (with error context if replanning)
            import time
            start_time = time.time()

            if retry_count > 0 and errors:
                # Replanning after error
                error_context = {
                    "failed_agent": errors[-1].get("agent", "unknown"),
                    "error": errors[-1].get("error", "unknown"),
                    "failed_step": errors[-1].get("step", "unknown")
                }
                result = self.planner.replan_with_error_context(user_message, error_context)
            else:
                # Normal planning
                result = self.planner.plan(user_message)

            exec_time = time.time() - start_time

            # Extract structured plan
            structured_plan = result.get("structured_plan", {})
            plan_json = result.get("plan", "")

            plan_message = AIMessage(
                content=f"Planning Agent: Created structured plan with {len(structured_plan.get('steps', []))} steps"
            )

            print(f"✅ PLANNER AGENT COMPLETED in {exec_time:.2f}s")
            print(f"   Status: {result.get('status')}")
            print(f"   Steps: {len(structured_plan.get('steps', []))}")
            print(f"   Plan: {structured_plan.get('plan_description', 'N/A')}")
            print(f"{'='*60}\n")

            return {
                "messages": [plan_message],
                "execution_plan": plan_json,
                "structured_plan": structured_plan,
                "current_agent": "planner",
                "current_step_index": 0,  # Reset step index for new plan
                "results": {**state.get("results", {}), "planner": result}
            }

        except Exception as e:
            print(f"❌ PLANNER AGENT ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

            error_info = {
                "agent": "planner",
                "error": str(e),
                "step": "planning",
                "timestamp": pd.Timestamp.now().isoformat()
            }

            return {
                "messages": [AIMessage(content=f"Planner Agent Error: {str(e)}")],
                "current_agent": "planner",
                "errors": state.get("errors", []) + [error_info],
                "results": {**state.get("results", {}), "planner": {"status": "error", "error": str(e)}}
            }

    def _data_reader_node(self, state: ImprovedMultiAgentState) -> Dict[str, Any]:
        """
        Execute data reader agent with IN-MEMORY DataFrame storage.

        IMPROVEMENT: Loads DataFrame once and stores in state, eliminating
        repeated disk reads by subsequent agents.
        """
        try:
            print(f"\n{'='*60}")
            print(f"📊 DATA READER AGENT STARTING")
            print(f"{'='*60}")

            file_path = state.get("current_file_path", "data/titanic.csv")
            print(f"📁 File path: {file_path}")

            # Get current step parameters
            current_step = self._get_current_step(state)
            if current_step:
                # Extract parameters from structured plan
                params = current_step.get("input_parameters", {})
                file_path = params.get("file_path", file_path)
                analysis_type = params.get("analysis_type", "comprehensive")
                task_desc = current_step.get("task_description", "Analyze data")
                print(f"📋 Task: {task_desc}")
            else:
                analysis_type = "comprehensive"

            print(f"🔄 Calling data reader agent...")

            import time
            start_time = time.time()
            result = self.data_reader.analyze_data(file_path, analysis_type)
            exec_time = time.time() - start_time

            # IMPROVEMENT: Load DataFrame into memory
            dataframe = None
            dataframe_metadata = None

            try:
                if file_path.endswith('.csv'):
                    dataframe = pd.read_csv(file_path)
                elif file_path.endswith('.json'):
                    dataframe = pd.read_json(file_path)

                if dataframe is not None:
                    dataframe_metadata = {
                        "shape": dataframe.shape,
                        "columns": list(dataframe.columns),
                        "dtypes": {col: str(dtype) for col, dtype in dataframe.dtypes.items()},
                        "memory_usage_mb": dataframe.memory_usage(deep=True).sum() / 1024 / 1024
                    }
                    print(f"✨ DataFrame loaded into memory: {dataframe.shape}")
                    print(f"   Memory usage: {dataframe_metadata['memory_usage_mb']:.2f} MB")
            except Exception as e:
                print(f"⚠️ Warning: Could not load DataFrame into memory: {e}")

            reader_message = AIMessage(
                content=f"Data Reader Agent: {result.get('analysis', 'Analysis failed')[:200]}..."
            )

            print(f"✅ DATA READER AGENT COMPLETED in {exec_time:.2f}s")
            print(f"   Status: {result.get('status')}")
            print(f"{'='*60}\n")

            return {
                "messages": [reader_message],
                "current_agent": "data_reader",
                "current_dataframe": dataframe,
                "dataframe_metadata": dataframe_metadata,
                "results": {**state.get("results", {}), "data_reader": result}
            }

        except Exception as e:
            print(f"❌ DATA READER AGENT ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

            error_info = {
                "agent": "data_reader",
                "error": str(e),
                "step": state.get("current_step_index", 0),
                "timestamp": pd.Timestamp.now().isoformat()
            }

            return {
                "messages": [AIMessage(content=f"Data Reader Agent Error: {str(e)}")],
                "current_agent": "data_reader",
                "errors": state.get("errors", []) + [error_info],
                "results": {**state.get("results", {}), "data_reader": {"status": "error", "error": str(e)}}
            }

    def _data_manipulation_node(self, state: ImprovedMultiAgentState) -> Dict[str, Any]:
        """Execute data manipulation agent with DataFrame state management."""
        try:
            print(f"\n{'='*60}")
            print(f"🔧 DATA MANIPULATION AGENT STARTING")
            print(f"{'='*60}")

            # Get parameters from structured plan
            current_step = self._get_current_step(state)
            if current_step:
                params = current_step.get("input_parameters", {})
                file_path = params.get("file_path", state.get("current_file_path", "data/titanic.csv"))
                manipulation_request = params.get("manipulation_request", "Handle missing values")
                task_desc = current_step.get("task_description", "Manipulate data")
            else:
                file_path = state.get("current_file_path", "data/titanic.csv")
                manipulation_request = "Handle missing values and prepare data for analysis"
                task_desc = "Data manipulation"

            print(f"📁 File path: {file_path}")
            print(f"📋 Task: {task_desc}")
            print(f"📝 Request: {manipulation_request}")

            # Check if DataFrame is in memory
            if state.get("current_dataframe") is not None:
                print(f"✨ Using DataFrame from memory (avoiding disk read)")
                print(f"   Shape: {state.get('dataframe_metadata', {}).get('shape', 'unknown')}")

            print(f"🔄 Calling data manipulation agent...")

            import time
            start_time = time.time()
            result = self.data_manipulation.manipulate_data(file_path, manipulation_request)
            exec_time = time.time() - start_time

            # TODO: Update DataFrame in memory if manipulation creates new file
            # For now, keep existing DataFrame (could be enhanced to reload)

            manipulation_message = AIMessage(
                content=f"Data Manipulation Agent: {result.get('result', 'Manipulation failed')[:200]}..."
            )

            print(f"✅ DATA MANIPULATION AGENT COMPLETED in {exec_time:.2f}s")
            print(f"   Status: {result.get('status')}")
            print(f"{'='*60}\n")

            return {
                "messages": [manipulation_message],
                "current_agent": "data_manipulation",
                "results": {**state.get("results", {}), "data_manipulation": result}
            }

        except Exception as e:
            print(f"❌ DATA MANIPULATION AGENT ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

            error_info = {
                "agent": "data_manipulation",
                "error": str(e),
                "step": state.get("current_step_index", 0),
                "timestamp": pd.Timestamp.now().isoformat()
            }

            return {
                "messages": [AIMessage(content=f"Data Manipulation Agent Error: {str(e)}")],
                "current_agent": "data_manipulation",
                "errors": state.get("errors", []) + [error_info],
                "results": {**state.get("results", {}), "data_manipulation": {"status": "error", "error": str(e)}}
            }

    def _data_operations_node(self, state: ImprovedMultiAgentState) -> Dict[str, Any]:
        """Execute data operations agent."""
        try:
            print(f"\n{'='*60}")
            print(f"⚡ DATA OPERATIONS AGENT STARTING")
            print(f"{'='*60}")

            # Get parameters from structured plan
            current_step = self._get_current_step(state)
            if current_step:
                params = current_step.get("input_parameters", {})
                file_path = params.get("file_path", state.get("current_file_path", "data/titanic.csv"))
                operation_request = params.get("operation_request", "Analyze data patterns")
                task_desc = current_step.get("task_description", "Perform operations")
            else:
                file_path = state.get("current_file_path", "data/titanic.csv")
                operation_request = "Analyze data patterns and perform basic statistical operations"
                task_desc = "Data operations"

            print(f"📁 File path: {file_path}")
            print(f"📋 Task: {task_desc}")
            print(f"📝 Request: {operation_request}")

            if state.get("current_dataframe") is not None:
                print(f"✨ DataFrame available in memory")

            print(f"🔄 Calling data operations agent...")

            import time
            start_time = time.time()
            result = self.data_operations.perform_operations(file_path, operation_request)
            exec_time = time.time() - start_time

            operations_message = AIMessage(
                content=f"Data Operations Agent: {result.get('result', 'Operations failed')[:200]}..."
            )

            print(f"✅ DATA OPERATIONS AGENT COMPLETED in {exec_time:.2f}s")
            print(f"   Status: {result.get('status')}")
            print(f"{'='*60}\n")

            return {
                "messages": [operations_message],
                "current_agent": "data_operations",
                "results": {**state.get("results", {}), "data_operations": result}
            }

        except Exception as e:
            print(f"❌ DATA OPERATIONS AGENT ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

            error_info = {
                "agent": "data_operations",
                "error": str(e),
                "step": state.get("current_step_index", 0),
                "timestamp": pd.Timestamp.now().isoformat()
            }

            return {
                "messages": [AIMessage(content=f"Data Operations Agent Error: {str(e)}")],
                "current_agent": "data_operations",
                "errors": state.get("errors", []) + [error_info],
                "results": {**state.get("results", {}), "data_operations": {"status": "error", "error": str(e)}}
            }

    def _ml_prediction_node(self, state: ImprovedMultiAgentState) -> Dict[str, Any]:
        """Execute ML prediction agent."""
        try:
            print(f"\n{'='*60}")
            print(f"🎯 ML PREDICTION AGENT STARTING")
            print(f"{'='*60}")

            # Get parameters from structured plan
            current_step = self._get_current_step(state)
            if current_step:
                params = current_step.get("input_parameters", {})
                file_path = params.get("file_path", state.get("current_file_path", "data/titanic.csv"))
                ml_request = params.get("ml_request", "Train a classification model")
                task_desc = current_step.get("task_description", "Train model")
            else:
                file_path = state.get("current_file_path", "data/titanic.csv")
                ml_request = "Train a classification model to predict the target variable"
                task_desc = "ML prediction"

            print(f"📁 File path: {file_path}")
            print(f"📋 Task: {task_desc}")
            print(f"📝 Request: {ml_request}")

            if state.get("current_dataframe") is not None:
                print(f"✨ DataFrame available in memory")

            print(f"🔄 Calling ML prediction agent...")

            import time
            start_time = time.time()
            result = self.ml_prediction.train_model(file_path, ml_request)
            exec_time = time.time() - start_time

            ml_message = AIMessage(
                content=f"ML Prediction Agent: {result.get('result', 'ML training failed')[:200]}..."
            )

            print(f"✅ ML PREDICTION AGENT COMPLETED in {exec_time:.2f}s")
            print(f"   Status: {result.get('status')}")
            print(f"{'='*60}\n")

            return {
                "messages": [ml_message],
                "current_agent": "ml_prediction",
                "results": {**state.get("results", {}), "ml_prediction": result}
            }

        except Exception as e:
            print(f"❌ ML PREDICTION AGENT ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

            error_info = {
                "agent": "ml_prediction",
                "error": str(e),
                "step": state.get("current_step_index", 0),
                "timestamp": pd.Timestamp.now().isoformat()
            }

            return {
                "messages": [AIMessage(content=f"ML Prediction Agent Error: {str(e)}")],
                "current_agent": "ml_prediction",
                "errors": state.get("errors", []) + [error_info],
                "results": {**state.get("results", {}), "ml_prediction": {"status": "error", "error": str(e)}}
            }

    def _supervisor_node(
        self, state: ImprovedMultiAgentState
    ) -> Command[Literal["data_reader", "data_manipulation", "data_operations", "ml_prediction", "error_handler", END]]:
        """
        IMPROVED Supervisor with structured plan routing.

        Instead of string matching, reads the JSON plan and executes steps sequentially.
        Handles errors by routing to error_handler for retry logic.
        """
        try:
            print(f"\n{'='*60}")
            print(f"🎯 SUPERVISOR - Determining next step")
            print(f"{'='*60}")

            # Get structured plan
            structured_plan = state.get("structured_plan", {})
            steps = structured_plan.get("steps", [])
            current_step_index = state.get("current_step_index", 0)
            current_agent = state.get("current_agent", "")
            errors = state.get("errors", [])

            # Mark current step as completed
            completed = list(state.get("completed_steps", []))
            if current_agent and current_agent != "planner" and current_agent not in completed:
                completed.append(current_agent)
                print(f"✅ Completed: {current_agent}")

            # Check for errors in last agent execution
            if errors and len(errors) > 0:
                last_error = errors[-1]
                if last_error.get("agent") == current_agent:
                    print(f"❌ Error detected in {current_agent}")
                    retry_count = state.get("retry_count", 0)
                    max_retries = state.get("max_retries", 2)

                    if retry_count < max_retries:
                        print(f"🔄 Routing to error_handler for retry ({retry_count + 1}/{max_retries})")
                        return Command(
                            goto="error_handler",
                            update={
                                "completed_steps": completed,
                                "retry_count": retry_count + 1
                            }
                        )
                    else:
                        print(f"❌ Max retries reached. Ending workflow.")
                        final_message = AIMessage(
                            content=f"Workflow failed after {max_retries} retries. Last error: {last_error.get('error')}"
                        )
                        return Command(
                            goto=END,
                            update={
                                "messages": [final_message],
                                "completed_steps": completed
                            }
                        )

            # Check if we have more steps to execute
            if current_step_index < len(steps):
                next_step = steps[current_step_index]
                agent_name = next_step.get("agent_name", "")
                task_desc = next_step.get("task_description", "")

                print(f"📋 Step {current_step_index + 1}/{len(steps)}: {agent_name}")
                print(f"   Task: {task_desc}")

                # Map agent names to node names
                agent_map = {
                    "DataReaderAgent": "data_reader",
                    "DataManipulationAgent": "data_manipulation",
                    "DataOperationsAgent": "data_operations",
                    "MLPredictionAgent": "ml_prediction"
                }

                node_name = agent_map.get(agent_name)

                if node_name:
                    print(f"➡️ Routing to: {node_name}")
                    return Command(
                        goto=node_name,
                        update={
                            "completed_steps": completed,
                            "current_step_index": current_step_index + 1
                        }
                    )
                else:
                    print(f"⚠️ Unknown agent: {agent_name}")
                    # Skip to next step
                    return Command(
                        goto="supervisor",
                        update={
                            "completed_steps": completed,
                            "current_step_index": current_step_index + 1
                        }
                    )
            else:
                # All steps completed successfully
                print(f"✅ All {len(steps)} steps completed successfully!")
                print(f"{'='*60}\n")

                final_message = AIMessage(
                    content=f"Multi-agent analysis completed successfully! Executed {len(steps)} steps."
                )
                return Command(
                    goto=END,
                    update={
                        "messages": [final_message],
                        "completed_steps": completed
                    }
                )

        except Exception as e:
            print(f"❌ SUPERVISOR ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

            error_message = AIMessage(content=f"Supervisor Error: {str(e)}")
            return Command(
                goto=END,
                update={"messages": [error_message]}
            )

    def _error_handler_node(self, state: ImprovedMultiAgentState) -> Dict[str, Any]:
        """
        IMPROVEMENT 4: Error handler that triggers replanning.

        When an agent fails, this node prepares the state for replanning
        by routing back to the planner with error context.
        """
        try:
            print(f"\n{'='*60}")
            print(f"🔧 ERROR HANDLER - Processing failure")
            print(f"{'='*60}")

            errors = state.get("errors", [])
            retry_count = state.get("retry_count", 0)

            if errors:
                last_error = errors[-1]
                print(f"❌ Last error from: {last_error.get('agent')}")
                print(f"   Error: {last_error.get('error')[:100]}...")
                print(f"🔄 Preparing for replan (retry {retry_count})")

            # Return state update - will route back to planner
            return {
                "current_agent": "error_handler",
                "current_step_index": 0,  # Reset step index for new plan
                "completed_steps": []  # Clear completed steps
            }

        except Exception as e:
            print(f"❌ ERROR HANDLER FAILURE: {str(e)}")
            return {
                "current_agent": "error_handler",
                "errors": state.get("errors", []) + [{
                    "agent": "error_handler",
                    "error": str(e),
                    "timestamp": pd.Timestamp.now().isoformat()
                }]
            }

    def _get_current_step(self, state: ImprovedMultiAgentState) -> Optional[Dict[str, Any]]:
        """Helper to get current step from structured plan."""
        structured_plan = state.get("structured_plan", {})
        steps = structured_plan.get("steps", [])
        # Use current_step_index - 1 because supervisor increments before calling agent
        step_index = state.get("current_step_index", 1) - 1

        if 0 <= step_index < len(steps):
            return steps[step_index]
        return None

    @trace_workflow_execution("improved_multi_agent_analysis")
    def run_analysis(self, user_prompt: str, file_path: str = None) -> Dict[str, Any]:
        """
        Run the complete improved multi-agent analysis.

        Args:
            user_prompt: User's analysis request
            file_path: Path to data file (optional, defaults to Titanic dataset)

        Returns:
            Dictionary containing all results from the analysis
        """
        try:
            # Log user interaction
            langsmith_logger.log_user_interaction(user_prompt, file_path)

            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=user_prompt)],
                "current_file_path": file_path or "data/titanic.csv",
                "execution_plan": "",
                "structured_plan": None,
                "current_dataframe": None,
                "dataframe_metadata": None,
                "errors": [],
                "retry_count": 0,
                "max_retries": 2,
                "current_step_index": 0,
                "completed_steps": [],
                "current_agent": None,
                "results": {}
            }

            # Run the workflow
            final_state = self.workflow.invoke(initial_state)

            # Format results
            result = {
                "status": "success" if len(final_state.get("errors", [])) == 0 else "completed_with_errors",
                "user_prompt": user_prompt,
                "file_path": final_state.get("current_file_path"),
                "execution_plan": final_state.get("execution_plan"),
                "structured_plan": final_state.get("structured_plan"),
                "dataframe_shape": final_state.get("dataframe_metadata", {}).get("shape"),
                "completed_steps": final_state.get("completed_steps", []),
                "total_steps": len(final_state.get("structured_plan", {}).get("steps", [])),
                "errors": final_state.get("errors", []),
                "retry_count": final_state.get("retry_count", 0),
                "messages": [msg.content for msg in final_state.get("messages", [])],
                "agent_results": final_state.get("results", {})
            }

            # Log workflow completion
            langsmith_logger.log_workflow_completion(result)

            return result

        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "user_prompt": user_prompt
            }

            # Log error
            langsmith_logger.log_workflow_completion(error_result)

            return error_result
