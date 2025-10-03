"""Enhanced LangGraph orchestration system with detailed tool and agent tracking."""

from typing import Dict, Any, List, Optional, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
import json
import time

# Import agents
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.planner_agent import PlannerAgent
from agents.data_reader_agent import DataReaderAgent
from agents.data_manipulation_agent import DataManipulationAgent
from agents.data_operations_agent import DataOperationsAgent
from agents.ml_prediction_agent import MLPredictionAgent

# Import LangSmith integration and tracking
from core.langsmith_integration import trace_workflow_execution, trace_agent_execution, langsmith_logger
from core.tool_tracking_wrapper import global_tracker, track_agent_execution


class MultiAgentState(MessagesState):
    """Extended state for multi-agent workflow with tracking."""
    current_file_path: Optional[str] = None
    execution_plan: Optional[str] = None
    completed_steps: List[str] = []
    current_agent: Optional[str] = None
    results: Dict[str, Any] = {}
    # New fields for tracking
    tools_used: List[Dict[str, Any]] = []
    agents_activated: List[Dict[str, Any]] = []
    execution_metadata: Dict[str, Any] = {}


class EnhancedMultiAgentOrchestrator:
    """Enhanced orchestrator with detailed tracking capabilities."""

    def __init__(self):
        # Initialize agents
        self.planner = PlannerAgent()
        self.data_reader = DataReaderAgent()
        self.data_manipulation = DataManipulationAgent()
        self.data_operations = DataOperationsAgent()
        self.ml_prediction = MLPredictionAgent()

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with enhanced tracking."""

        # Create the state graph
        workflow = StateGraph(MultiAgentState)

        # Add nodes with tracking wrappers
        workflow.add_node("planner", self._tracked_planner_node)
        workflow.add_node("data_reader", self._tracked_data_reader_node)
        workflow.add_node("data_manipulation", self._tracked_data_manipulation_node)
        workflow.add_node("data_operations", self._tracked_data_operations_node)
        workflow.add_node("ml_prediction", self._tracked_ml_prediction_node)
        workflow.add_node("supervisor", self._tracked_supervisor_node)

        # Add edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "supervisor")
        workflow.add_edge("data_reader", "supervisor")
        workflow.add_edge("data_manipulation", "supervisor")
        workflow.add_edge("data_operations", "supervisor")
        workflow.add_edge("ml_prediction", "supervisor")

        return workflow.compile()

    @track_agent_execution("PlannerAgent")
    def _tracked_planner_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute the planner agent with tracking."""
        start_time = time.time()

        try:
            # Get the user's original message
            user_message = None
            for message in state["messages"]:
                if isinstance(message, HumanMessage):
                    user_message = message.content
                    break

            if not user_message:
                return {
                    "messages": [AIMessage(content="No user message found.")],
                    "current_agent": "planner",
                    "execution_plan": "No plan created"
                }

            # Create execution plan
            result = self.planner.plan(user_message)

            plan_message = AIMessage(content=f"Planning Agent: {result.get('plan', 'Planning failed')}")

            execution_time = time.time() - start_time

            # Add to execution metadata
            execution_metadata = state.get("execution_metadata", {})
            execution_metadata["planner_execution_time"] = execution_time
            execution_metadata["plan_created"] = result.get("plan", "")

            return {
                "messages": [plan_message],
                "execution_plan": result.get("plan", ""),
                "current_agent": "planner",
                "results": {"planner": result},
                "execution_metadata": execution_metadata,
                "agents_activated": [{"agent": "planner", "time": execution_time, "timestamp": time.time()}]
            }

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Error in planner node: {str(e)}")
            return {
                "messages": [AIMessage(content=f"Planning failed: {str(e)}")],
                "current_agent": "planner",
                "execution_plan": "Planning failed due to error",
                "execution_metadata": {"planner_error": str(e), "planner_execution_time": execution_time}
            }

    @track_agent_execution("DataReaderAgent")
    def _tracked_data_reader_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute the data reader agent with tracking."""
        start_time = time.time()

        try:
            file_path = state.get("current_file_path", "data/titanic.csv")

            # Get the latest message for context
            user_message = ""
            for message in reversed(state["messages"]):
                if isinstance(message, HumanMessage):
                    user_message = message.content
                    break

            # Execute data reading
            result = self.data_reader.read_and_analyze(file_path, user_message)

            response_message = AIMessage(content=f"Data Reader Agent: {result.get('summary', 'Analysis completed')}")

            execution_time = time.time() - start_time

            # Update execution metadata
            execution_metadata = state.get("execution_metadata", {})
            execution_metadata["data_reader_execution_time"] = execution_time
            execution_metadata["file_analyzed"] = file_path

            # Track tools used (this would be enhanced by modifying the agent to report tool usage)
            tools_used = state.get("tools_used", [])
            tools_used.append({
                "agent": "data_reader",
                "tools": ["read_csv_file", "get_data_summary"],  # This should come from actual agent execution
                "execution_time": execution_time,
                "timestamp": time.time()
            })

            return {
                "messages": [response_message],
                "current_agent": "data_reader",
                "results": {**state.get("results", {}), "data_reader": result},
                "execution_metadata": execution_metadata,
                "tools_used": tools_used,
                "completed_steps": state.get("completed_steps", []) + ["data_reading"]
            }

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Error in data reader node: {str(e)}")
            return {
                "messages": [AIMessage(content=f"Data reading failed: {str(e)}")],
                "current_agent": "data_reader",
                "execution_metadata": {**state.get("execution_metadata", {}), "data_reader_error": str(e)}
            }

    @track_agent_execution("DataManipulationAgent")
    def _tracked_data_manipulation_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute the data manipulation agent with tracking."""
        start_time = time.time()

        try:
            file_path = state.get("current_file_path", "data/titanic.csv")
            execution_plan = state.get("execution_plan", "")

            # Execute data manipulation
            result = self.data_manipulation.manipulate_data(file_path, execution_plan)

            response_message = AIMessage(content=f"Data Manipulation Agent: {result.get('summary', 'Manipulation completed')}")

            execution_time = time.time() - start_time

            # Update execution metadata
            execution_metadata = state.get("execution_metadata", {})
            execution_metadata["data_manipulation_execution_time"] = execution_time

            # Track tools used
            tools_used = state.get("tools_used", [])
            tools_used.append({
                "agent": "data_manipulation",
                "tools": ["create_dummy_variables", "handle_missing_values"],  # This should come from actual agent execution
                "execution_time": execution_time,
                "timestamp": time.time()
            })

            return {
                "messages": [response_message],
                "current_agent": "data_manipulation",
                "results": {**state.get("results", {}), "data_manipulation": result},
                "execution_metadata": execution_metadata,
                "tools_used": tools_used,
                "completed_steps": state.get("completed_steps", []) + ["data_manipulation"]
            }

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Error in data manipulation node: {str(e)}")
            return {
                "messages": [AIMessage(content=f"Data manipulation failed: {str(e)}")],
                "current_agent": "data_manipulation",
                "execution_metadata": {**state.get("execution_metadata", {}), "data_manipulation_error": str(e)}
            }

    @track_agent_execution("DataOperationsAgent")
    def _tracked_data_operations_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute the data operations agent with tracking."""
        start_time = time.time()

        try:
            file_path = state.get("current_file_path", "data/titanic.csv")
            execution_plan = state.get("execution_plan", "")

            # Execute data operations
            result = self.data_operations.perform_operations(file_path, execution_plan)

            response_message = AIMessage(content=f"Data Operations Agent: {result.get('summary', 'Operations completed')}")

            execution_time = time.time() - start_time

            # Update execution metadata
            execution_metadata = state.get("execution_metadata", {})
            execution_metadata["data_operations_execution_time"] = execution_time

            # Track tools used
            tools_used = state.get("tools_used", [])
            tools_used.append({
                "agent": "data_operations",
                "tools": ["filter_data", "aggregate_data", "perform_math_operations"],  # This should come from actual agent execution
                "execution_time": execution_time,
                "timestamp": time.time()
            })

            return {
                "messages": [response_message],
                "current_agent": "data_operations",
                "results": {**state.get("results", {}), "data_operations": result},
                "execution_metadata": execution_metadata,
                "tools_used": tools_used,
                "completed_steps": state.get("completed_steps", []) + ["data_operations"]
            }

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Error in data operations node: {str(e)}")
            return {
                "messages": [AIMessage(content=f"Data operations failed: {str(e)}")],
                "current_agent": "data_operations",
                "execution_metadata": {**state.get("execution_metadata", {}), "data_operations_error": str(e)}
            }

    @track_agent_execution("MLPredictionAgent")
    def _tracked_ml_prediction_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute the ML prediction agent with tracking."""
        start_time = time.time()

        try:
            file_path = state.get("current_file_path", "data/titanic.csv")
            execution_plan = state.get("execution_plan", "")

            # Execute ML prediction
            result = self.ml_prediction.train_and_predict(file_path, execution_plan)

            response_message = AIMessage(content=f"ML Prediction Agent: {result.get('summary', 'ML completed')}")

            execution_time = time.time() - start_time

            # Update execution metadata
            execution_metadata = state.get("execution_metadata", {})
            execution_metadata["ml_prediction_execution_time"] = execution_time

            # Track tools used
            tools_used = state.get("tools_used", [])
            tools_used.append({
                "agent": "ml_prediction",
                "tools": ["train_random_forest_model", "train_regression_model", "evaluate_model"],  # This should come from actual agent execution
                "execution_time": execution_time,
                "timestamp": time.time()
            })

            return {
                "messages": [response_message],
                "current_agent": "ml_prediction",
                "results": {**state.get("results", {}), "ml_prediction": result},
                "execution_metadata": execution_metadata,
                "tools_used": tools_used,
                "completed_steps": state.get("completed_steps", []) + ["ml_prediction"]
            }

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Error in ML prediction node: {str(e)}")
            return {
                "messages": [AIMessage(content=f"ML prediction failed: {str(e)}")],
                "current_agent": "ml_prediction",
                "execution_metadata": {**state.get("execution_metadata", {}), "ml_prediction_error": str(e)}
            }

    @track_agent_execution("SupervisorAgent")
    def _tracked_supervisor_node(self, state: MultiAgentState) -> Command[Literal["data_reader", "data_manipulation", "data_operations", "ml_prediction", END]]:
        """Supervisor node with enhanced tracking."""
        start_time = time.time()

        try:
            execution_plan = state.get("execution_plan", "").lower()
            completed_steps = state.get("completed_steps", [])

            # Update execution metadata for supervisor decisions
            execution_metadata = state.get("execution_metadata", {})
            execution_metadata["supervisor_decisions"] = execution_metadata.get("supervisor_decisions", [])

            # Determine next agent based on plan and completed steps
            if "datareaderagent" in execution_plan and "data_reading" not in completed_steps:
                decision = "data_reader"
            elif ("datamanipulationagent" in execution_plan or "dummy" in execution_plan or "encode" in execution_plan) and "data_manipulation" not in completed_steps:
                decision = "data_manipulation"
            elif ("dataoperationsagent" in execution_plan or "filter" in execution_plan or "aggregate" in execution_plan or "calculate" in execution_plan) and "data_operations" not in completed_steps:
                decision = "data_operations"
            elif ("mlpredictionagent" in execution_plan or "model" in execution_plan or "predict" in execution_plan or "train" in execution_plan) and "ml_prediction" not in completed_steps:
                decision = "ml_prediction"
            else:
                decision = END

            execution_time = time.time() - start_time

            # Log supervisor decision
            execution_metadata["supervisor_decisions"].append({
                "decision": decision,
                "reasoning": f"Based on plan: {execution_plan}, completed: {completed_steps}",
                "execution_time": execution_time,
                "timestamp": time.time()
            })

            # Update state with tracking info
            return Command(
                goto=decision,
                update={"execution_metadata": execution_metadata}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Error in supervisor node: {str(e)}")
            return Command(
                goto=END,
                update={"execution_metadata": {**state.get("execution_metadata", {}), "supervisor_error": str(e)}}
            )

    @trace_workflow_execution("enhanced_multi_agent_analysis")
    def run_enhanced_analysis(self, user_prompt: str, file_path: str = None, session_id: str = None) -> Dict[str, Any]:
        """
        Run the complete multi-agent analysis with detailed tracking.

        Args:
            user_prompt: User's analysis request
            file_path: Path to data file (optional, defaults to Titanic dataset)
            session_id: Unique session identifier for tracking

        Returns:
            Dictionary containing all results and detailed statistics
        """
        # Start tracking session
        if session_id:
            global_tracker.start_session(session_id, user_prompt)

        workflow_start_time = time.time()

        try:
            # Log user interaction
            langsmith_logger.log_user_interaction(user_prompt, file_path)

            # Initialize state with tracking fields
            initial_state = {
                "messages": [HumanMessage(content=user_prompt)],
                "current_file_path": file_path or "data/titanic.csv",
                "execution_plan": "",
                "completed_steps": [],
                "current_agent": None,
                "results": {},
                "tools_used": [],
                "agents_activated": [],
                "execution_metadata": {
                    "workflow_start_time": workflow_start_time,
                    "user_prompt": user_prompt,
                    "file_path": file_path or "data/titanic.csv"
                }
            }

            # Run the workflow
            final_state = self.workflow.invoke(initial_state)

            workflow_end_time = time.time()
            total_workflow_time = workflow_end_time - workflow_start_time

            # Extract tracking information
            execution_metadata = final_state.get("execution_metadata", {})
            execution_metadata["workflow_end_time"] = workflow_end_time
            execution_metadata["total_workflow_time"] = total_workflow_time

            # Format results with enhanced tracking
            result = {
                "status": "success",
                "user_prompt": user_prompt,
                "file_path": final_state.get("current_file_path"),
                "execution_plan": final_state.get("execution_plan"),
                "completed_steps": final_state.get("completed_steps", []),
                "messages": [msg.content for msg in final_state.get("messages", [])],
                "agent_results": final_state.get("results", {}),

                # Enhanced tracking information
                "execution_metadata": execution_metadata,
                "tools_used": final_state.get("tools_used", []),
                "agents_activated": final_state.get("agents_activated", []),
                "total_execution_time": total_workflow_time,

                # Summary statistics
                "summary_stats": {
                    "total_agents_used": len(final_state.get("agents_activated", [])),
                    "total_tools_used": len(final_state.get("tools_used", [])),
                    "total_steps_completed": len(final_state.get("completed_steps", [])),
                    "workflow_duration": total_workflow_time
                }
            }

            # End tracking session
            if session_id:
                session_data = global_tracker.end_session()
                result["tracking_session_data"] = session_data

            # Log workflow completion
            langsmith_logger.log_workflow_completion(result)

            return result

        except Exception as e:
            workflow_end_time = time.time()
            total_workflow_time = workflow_end_time - workflow_start_time

            # End tracking session even on error
            if session_id:
                session_data = global_tracker.end_session()

            error_result = {
                "status": "error",
                "error": str(e),
                "user_prompt": user_prompt,
                "total_execution_time": total_workflow_time,
                "tracking_session_data": session_data if session_id else None
            }

            langsmith_logger.log_workflow_error(error_result)
            return error_result