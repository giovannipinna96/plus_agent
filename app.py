"""Gradio interface for the multi-agent data analysis system."""

import gradio as gr
import pandas as pd
import os
import json
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import core components
from core.config import config
from core.orchestrator import MultiAgentOrchestrator

# Initialize the orchestrator
orchestrator = None


def initialize_system():
    """Initialize the multi-agent system."""
    global orchestrator
    try:
        print("Initializing multi-agent system...")
        
        # Setup LangSmith if configured
        config.setup_langsmith()
        
        # Initialize the orchestrator
        orchestrator = MultiAgentOrchestrator()
        
        print("Multi-agent system initialized successfully!")
        return True
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        return False


def process_user_request(message: str, chat_history: List, uploaded_file) -> tuple:
    """
    Process user request through the multi-agent system.
    
    Args:
        message: User's message/request
        chat_history: Previous chat history
        uploaded_file: Uploaded file (if any)
        
    Returns:
        Tuple of (updated_chat_history, empty_message, file_info)
    """
    global orchestrator
    
    try:
        # Determine file path
        file_path = None
        file_info = "Using default Titanic dataset"
        
        if uploaded_file is not None:
            # Use uploaded file
            file_path = uploaded_file.name
            file_info = f"Using uploaded file: {os.path.basename(file_path)}"
            
            # Check file format
            if not file_path.endswith(('.csv', '.json')):
                error_msg = "Please upload a CSV or JSON file."
                chat_history.append([message, error_msg])
                return chat_history, "", f"Error: {error_msg}"
        else:
            # Use default Titanic dataset
            file_path = config.default_dataset_path
            if not os.path.exists(file_path):
                error_msg = "Default dataset not found. Please upload a file."
                chat_history.append([message, error_msg])
                return chat_history, "", f"Error: {error_msg}"
        
        # Add user message to chat history
        chat_history.append([message, "Processing your request..."])
        
        # Run the multi-agent analysis
        if orchestrator is None:
            error_msg = "System not initialized. Please refresh the page."
            chat_history[-1][1] = error_msg
            return chat_history, "", f"Error: {error_msg}"
        
        result = orchestrator.run_analysis(message, file_path)
        
        if result["status"] == "success":
            # Format the response
            response_parts = []
            
            # Add execution plan
            if result.get("execution_plan"):
                response_parts.append(f"**Execution Plan:**\n{result['execution_plan']}")
            
            # Add agent results
            agent_results = result.get("agent_results", {})
            for agent_name, agent_result in agent_results.items():
                if agent_result.get("status") == "success":
                    response_parts.append(f"**{agent_name.replace('_', ' ').title()} Results:**")
                    
                    if agent_name == "data_reader" and "analysis" in agent_result:
                        response_parts.append(agent_result["analysis"])
                    elif agent_name == "data_manipulation" and "result" in agent_result:
                        response_parts.append(agent_result["result"])
                    elif agent_name == "data_operations" and "result" in agent_result:
                        response_parts.append(agent_result["result"])
                    elif agent_name == "ml_prediction" and "result" in agent_result:
                        response_parts.append(agent_result["result"])
                    elif agent_name == "planner" and "plan" in agent_result:
                        response_parts.append(agent_result["plan"])
            
            # Combine all parts
            if response_parts:
                response = "\n\n".join(response_parts)
            else:
                response = "Analysis completed successfully!"
            
            # Update chat history
            chat_history[-1][1] = response
            
        else:
            # Handle error
            error_msg = f"Error: {result.get('error', 'Unknown error occurred')}"
            chat_history[-1][1] = error_msg
            file_info = f"Error: {error_msg}"
        
        return chat_history, "", file_info
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        if chat_history and len(chat_history) > 0:
            chat_history[-1][1] = error_msg
        else:
            chat_history.append([message, error_msg])
        return chat_history, "", f"Error: {error_msg}"


def handle_file_upload(file) -> str:
    """Handle file upload and return file information."""
    if file is None:
        return "No file uploaded. Using default Titanic dataset."
    
    try:
        # Get file info
        file_name = os.path.basename(file.name)
        file_size = os.path.getsize(file.name)
        
        # Check file format
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
            file_type = "CSV"
        elif file.name.endswith('.json'):
            with open(file.name, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                    file_type = "JSON (array)"
                else:
                    return f"JSON file format not supported. Please upload a JSON array."
        else:
            return "Unsupported file format. Please upload CSV or JSON files."
        
        return f"📁 **File Uploaded Successfully!**\n\n" \
               f"**Name:** {file_name}\n" \
               f"**Type:** {file_type}\n" \
               f"**Size:** {file_size:,} bytes\n" \
               f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns\n" \
               f"**Columns:** {', '.join(list(df.columns)[:10])}{'...' if len(df.columns) > 10 else ''}"
               
    except Exception as e:
        return f"Error reading file: {str(e)}"


def get_sample_prompts() -> List[List[str]]:
    """Get sample prompts for different complexity levels."""
    return [
        ["Show me the basic information about this dataset"],
        ["What are the data types and missing values in each column?"],
        ["Calculate the average age by gender"],
        ["Create dummy variables for categorical columns and handle missing values"],
        ["Filter passengers who survived and analyze their characteristics"],
        ["Train a random forest model to predict survival and show feature importance"],
        ["Compare the performance of different ML models for classification"],
        ["Perform a complete analysis: data exploration, preprocessing, and predictive modeling"]
    ]


def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="Multi-Agent Data Analysis System",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as interface:
        
        # Header
        with gr.Row():
            gr.HTML("""
                <div class="header">
                    <h1>🤖 Multi-Agent Data Analysis System</h1>
                    <p>Powered by LangChain, LangGraph, and Small LLMs</p>
                </div>
            """)
        
        # Main interface
        with gr.Row():
            # Left column - File upload and info
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Data Upload")
                
                file_upload = gr.File(
                    label="Upload CSV or JSON file",
                    file_types=[".csv", ".json"],
                    type="filepath"
                )
                
                file_info = gr.Markdown(
                    "**Default Dataset:** Titanic dataset (891 rows × 15 columns)\n\n"
                    "Upload your own CSV or JSON file, or use the default Titanic dataset for testing."
                )
                
                gr.Markdown("### 💡 Sample Prompts")
                
                sample_prompts = get_sample_prompts()
                
                gr.Markdown("**Simple:**")
                for prompt in sample_prompts[:2]:
                    gr.Button(prompt[0], size="sm")
                
                gr.Markdown("**Medium:**")  
                for prompt in sample_prompts[2:5]:
                    gr.Button(prompt[0], size="sm")
                    
                gr.Markdown("**Complex:**")
                for prompt in sample_prompts[5:]:
                    gr.Button(prompt[0], size="sm")
            
            # Right column - Chat interface
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat with the Multi-Agent System")
                
                chatbot = gr.Chatbot(
                    height=500,
                    placeholder="Multi-agent responses will appear here...",
                    show_label=False
                )
                
                with gr.Row():
                    message_input = gr.Textbox(
                        placeholder="Describe what you want to analyze (e.g., 'Show me survival rates by passenger class')",
                        show_label=False,
                        scale=4
                    )
                    submit_btn = gr.Button("🚀 Analyze", scale=1, variant="primary")
                
                gr.Markdown("""
                **Available Agents:**
                - 📊 **Data Reader**: Analyzes datasets, provides column info, summaries
                - 🔧 **Data Manipulation**: Handles missing values, creates dummy variables, transforms data
                - ⚡ **Data Operations**: Filters, aggregates, performs mathematical operations
                - 🎯 **ML Prediction**: Trains and evaluates machine learning models
                - 🎭 **Planner**: Orchestrates the overall workflow
                """)
        
        # Event handlers
        file_upload.change(
            fn=handle_file_upload,
            inputs=file_upload,
            outputs=file_info
        )
        
        submit_btn.click(
            fn=process_user_request,
            inputs=[message_input, chatbot, file_upload],
            outputs=[chatbot, message_input, file_info]
        )
        
        message_input.submit(
            fn=process_user_request,
            inputs=[message_input, chatbot, file_upload],
            outputs=[chatbot, message_input, file_info]
        )
        
        # Add sample prompt buttons functionality
        def create_prompt_handler(prompt_text):
            def handler():
                return prompt_text
            return handler
        
        # Add click handlers for sample prompts (simplified version)
        examples = gr.Examples(
            examples=sample_prompts,
            inputs=message_input,
            label="Click on examples to use them:"
        )
    
    return interface


def main():
    """Main function to run the Gradio app."""
    
    print("Starting Multi-Agent Data Analysis System...")
    
    # Initialize the system
    if not initialize_system():
        print("Failed to initialize system. Exiting...")
        return
    
    # Create and launch the interface
    interface = create_interface()
    
    print("Launching Gradio interface...")
    
    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        show_api=False,
        max_file_size="50mb"  # Allow larger files
    )


if __name__ == "__main__":
    main()