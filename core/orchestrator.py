"""LangGraph orchestration system for multi-agent workflow."""

from typing import Dict, Any, List, Optional, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
import json

from pydantic import Field

# Import agents
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.planner_agent import PlannerAgent
from agents.data_reader_agent import DataReaderAgent
from agents.data_manipulation_agent import DataManipulationAgent
from agents.data_operations_agent import DataOperationsAgent
from agents.ml_prediction_agent import MLPredictionAgent

# Import LangSmith integration
from core.langsmith_integration import trace_workflow_execution, trace_agent_execution, langsmith_logger

class MultiAgentState(MessagesState):
    """Extended state for multi-agent workflow."""
    current_file_path: Optional[str] = None
    execution_plan: Optional[str] = None
    completed_steps: List[str] = Field(default_factory=list)
    current_agent: Optional[str] = None
    results: Dict[str, Any] = Field(default_factory=dict)


class MultiAgentOrchestrator:
    """Orchestrates the multi-agent workflow using LangGraph."""
    
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
        """Build the LangGraph workflow."""
        
        # Create the state graph
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("data_reader", self._data_reader_node)
        workflow.add_node("data_manipulation", self._data_manipulation_node)
        workflow.add_node("data_operations", self._data_operations_node)
        workflow.add_node("ml_prediction", self._ml_prediction_node)
        workflow.add_node("supervisor", self._supervisor_node)
        
        # Add edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "supervisor")
        workflow.add_edge("data_reader", "supervisor")
        workflow.add_edge("data_manipulation", "supervisor")
        workflow.add_edge("data_operations", "supervisor")
        workflow.add_edge("ml_prediction", "supervisor")
        workflow.add_edge("supervisor", END)
        
        return workflow.compile()
    
    def _planner_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute the planner agent."""
        try:
            print(f"\n{'='*60}")
            print(f"🎭 PLANNER AGENT STARTING")
            print(f"{'='*60}")

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
            print(f"🔄 Calling planner agent...")

            # Create execution plan
            import time
            start_time = time.time()
            result = self.planner.plan(user_message)
            exec_time = time.time() - start_time

            plan_message = AIMessage(content=f"Planning Agent: {result.get('plan', 'Planning failed')}")

            print(f"✅ PLANNER AGENT COMPLETED in {exec_time:.2f}s")
            print(f"   Status: {result.get('status')}")
            print(f"   Plan length: {len(result.get('plan', ''))} characters")
            print(f"{'='*60}\n")

            return {
                "messages": [plan_message],
                "execution_plan": result.get("plan", ""),
                "current_agent": "planner",
                "results": {"planner": result}
            }

        except Exception as e:
            print(f"❌ PLANNER AGENT ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            error_message = AIMessage(content=f"Planner Agent Error: {str(e)}")
            return {
                "messages": [error_message],
                "current_agent": "planner",
                "results": {"planner": {"status": "error", "error": str(e)}}
            }
    
    def _data_reader_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute the data reader agent."""
        try:
            print(f"\n{'='*60}")
            print(f"📊 DATA READER AGENT STARTING")
            print(f"{'='*60}")

            file_path = state.get("current_file_path", "data/titanic.csv")
            print(f"📁 File path: {file_path}")
            print(f"🔄 Calling data reader agent...")

            import time
            start_time = time.time()
            result = self.data_reader.analyze_data(file_path, "comprehensive")
            exec_time = time.time() - start_time

            reader_message = AIMessage(content=f"Data Reader Agent: {result.get('analysis', 'Analysis failed')}")

            print(f"✅ DATA READER AGENT COMPLETED in {exec_time:.2f}s")
            print(f"   Status: {result.get('status')}")
            if result.get('status') == 'success':
                print(f"   Analysis length: {len(result.get('analysis', ''))} characters")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
            print(f"{'='*60}\n")

            return {
                "messages": [reader_message],
                "current_agent": "data_reader",
                "results": {**state.get("results", {}), "data_reader": result}
            }

        except Exception as e:
            print(f"❌ DATA READER AGENT ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            error_message = AIMessage(content=f"Data Reader Agent Error: {str(e)}")
            return {
                "messages": [error_message],
                "current_agent": "data_reader",
                "results": {**state.get("results", {}), "data_reader": {"status": "error", "error": str(e)}}
            }
    
    def _data_manipulation_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute the data manipulation agent."""
        try:
            print(f"\n{'='*60}")
            print(f"🔧 DATA MANIPULATION AGENT STARTING")
            print(f"{'='*60}")

            file_path = state.get("current_file_path", "data/titanic.csv")

            # Extract manipulation request from messages or use default
            manipulation_request = "Handle missing values and prepare data for analysis"
            print(f"📁 File path: {file_path}")
            print(f"📋 Request: {manipulation_request}")
            print(f"🔄 Calling data manipulation agent...")

            import time
            start_time = time.time()
            result = self.data_manipulation.manipulate_data(file_path, manipulation_request)
            exec_time = time.time() - start_time

            manipulation_message = AIMessage(content=f"Data Manipulation Agent: {result.get('result', 'Manipulation failed')}")

            print(f"✅ DATA MANIPULATION AGENT COMPLETED in {exec_time:.2f}s")
            print(f"   Status: {result.get('status')}")
            if result.get('status') == 'success':
                print(f"   Result length: {len(result.get('result', ''))} characters")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
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
            error_message = AIMessage(content=f"Data Manipulation Agent Error: {str(e)}")
            return {
                "messages": [error_message],
                "current_agent": "data_manipulation",
                "results": {**state.get("results", {}), "data_manipulation": {"status": "error", "error": str(e)}}
            }
    
    def _data_operations_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute the data operations agent."""
        try:
            print(f"\n{'='*60}")
            print(f"⚡ DATA OPERATIONS AGENT STARTING")
            print(f"{'='*60}")

            file_path = state.get("current_file_path", "data/titanic.csv")

            # Extract operation request from messages or use default
            operation_request = "Analyze data patterns and perform basic statistical operations"
            print(f"📁 File path: {file_path}")
            print(f"📋 Request: {operation_request}")
            print(f"🔄 Calling data operations agent...")

            import time
            start_time = time.time()
            result = self.data_operations.perform_operations(file_path, operation_request)
            exec_time = time.time() - start_time

            operations_message = AIMessage(content=f"Data Operations Agent: {result.get('result', 'Operations failed')}")

            print(f"✅ DATA OPERATIONS AGENT COMPLETED in {exec_time:.2f}s")
            print(f"   Status: {result.get('status')}")
            if result.get('status') == 'success':
                print(f"   Result length: {len(result.get('result', ''))} characters")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
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
            error_message = AIMessage(content=f"Data Operations Agent Error: {str(e)}")
            return {
                "messages": [error_message],
                "current_agent": "data_operations",
                "results": {**state.get("results", {}), "data_operations": {"status": "error", "error": str(e)}}
            }
    
    def _ml_prediction_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """Execute the ML prediction agent."""
        try:
            print(f"\n{'='*60}")
            print(f"🎯 ML PREDICTION AGENT STARTING")
            print(f"{'='*60}")

            file_path = state.get("current_file_path", "data/titanic.csv")

            # Extract ML request from messages or use default
            ml_request = "Train a classification model to predict the target variable"
            print(f"📁 File path: {file_path}")
            print(f"📋 Request: {ml_request}")
            print(f"🔄 Calling ML prediction agent...")

            import time
            start_time = time.time()
            result = self.ml_prediction.train_model(file_path, ml_request)
            exec_time = time.time() - start_time

            ml_message = AIMessage(content=f"ML Prediction Agent: {result.get('result', 'ML training failed')}")

            print(f"✅ ML PREDICTION AGENT COMPLETED in {exec_time:.2f}s")
            print(f"   Status: {result.get('status')}")
            if result.get('status') == 'success':
                print(f"   Result length: {len(result.get('result', ''))} characters")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
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
            error_message = AIMessage(content=f"ML Prediction Agent Error: {str(e)}")
            return {
                "messages": [error_message],
                "current_agent": "ml_prediction",
                "results": {**state.get("results", {}), "ml_prediction": {"status": "error", "error": str(e)}}
            }
    
    def _supervisor_node(self, state: MultiAgentState) -> Command[Literal["data_reader", "data_manipulation", "data_operations", "ml_prediction", END]]:
        """Supervisor node that decides which agent to call next."""
        try:
            # Get the execution plan and determine next step
            plan = state.get("execution_plan", "")
            completed = state.get("completed_steps", [])
            current_agent = state.get("current_agent", "")
            
            # Mark current step as completed
            if current_agent and current_agent not in completed:
                completed.append(current_agent)
            
            # Simple routing logic based on plan keywords
            if "DataReaderAgent" in plan and "data_reader" not in completed:
                return Command(
                    goto="data_reader",
                    update={"completed_steps": completed}
                )
            elif "DataManipulationAgent" in plan and "data_manipulation" not in completed:
                return Command(
                    goto="data_manipulation", 
                    update={"completed_steps": completed}
                )
            elif "DataOperationsAgent" in plan and "data_operations" not in completed:
                return Command(
                    goto="data_operations",
                    update={"completed_steps": completed}
                )
            elif "MLPredictionAgent" in plan and "ml_prediction" not in completed:
                return Command(
                    goto="ml_prediction",
                    update={"completed_steps": completed}
                )
            else:
                # All planned steps completed
                final_message = AIMessage(content="Multi-agent analysis completed successfully!")
                return Command(
                    goto=END,
                    update={
                        "messages": [final_message],
                        "completed_steps": completed
                    }
                )
                
        except Exception as e:
            error_message = AIMessage(content=f"Supervisor Error: {str(e)}")
            return Command(
                goto=END,
                update={"messages": [error_message]}
            )
    
    @trace_workflow_execution("multi_agent_analysis")
    def run_analysis(self, user_prompt: str, file_path: str = None) -> Dict[str, Any]:
        """
        Run the complete multi-agent analysis.
        
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
                "completed_steps": [],
                "current_agent": None,
                "results": {}
            }
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Format results
            result = {
                "status": "success",
                "user_prompt": user_prompt,
                "file_path": final_state.get("current_file_path"),
                "execution_plan": final_state.get("execution_plan"),
                "completed_steps": final_state.get("completed_steps", []),
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