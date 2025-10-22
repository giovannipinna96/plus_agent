"""Gradio interface for the multi-agent data analysis system using smolagents."""

import gradio as gr
import pandas as pd
import os
import json
import sys
import io
import tempfile
import shutil
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# Import smolagents system
from smolagents import CodeAgent, TransformersModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
   # "model_id": os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    "model_id": os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct"),
    "max_new_tokens": int(os.getenv("MAX_TOKENS", "1024")),
    "temperature": float(os.getenv("TEMPERATURE", "0.1")),
    "default_dataset": os.getenv("DEFAULT_DATASET_PATH", "data/titanic.csv"),
    "huggingface_token": os.getenv("HUGGINGFACE_TOKEN"),
}

# Global variables
model = None
manager_agent = None


def get_available_models() -> List[str]:
    """Get list of available LLM models."""
    return [
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "microsoft/Phi-4-mini-instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]


def create_smolagent_system(model_name: str):
    """Create the smolagent multi-agent system."""
    from smolagents_multiagent_system import create_agents, create_manager_agent

    print("🔧 Initializing Model...")

    # Initialize the LLM model
    model = TransformersModel(
        model_id=model_name,
        temperature=CONFIG["temperature"],
        max_new_tokens=CONFIG["max_new_tokens"],
        trust_remote_code=True,
        token=CONFIG["huggingface_token"]
    )

    print(f"✅ Model initialized: {model_name}")
    print("\n🤖 Creating Specialized Agents...")

    # Create specialized agents
    model_instance, data_reader, data_manipulation, statistical_agent, visualization_agent, ml_prediction, answer_agent = create_agents()

    # Create manager agent
    print("\n👔 Creating Manager Agent (Orchestrator)...")
    manager = create_manager_agent(model_instance, data_reader, data_manipulation, statistical_agent, visualization_agent, ml_prediction, answer_agent)

    print("✅ All agents created successfully!")

    return model_instance, manager


def initialize_system(model_name: str = None):
    """Initialize the multi-agent system with optional model selection."""
    global model, manager_agent, CONFIG

    try:
        print("\n" + "="*60)
        print("Initializing smolagent multi-agent system...")
        print("="*60 + "\n")

        # Update model if specified
        if model_name:
            CONFIG["model_id"] = model_name

        current_model = CONFIG["model_id"]

        # Load model into memory
        print(f"📦 Loading LLM model into VRAM: {current_model}")
        print("⏳ This may take 5-10 seconds on first load...")

        start_time = time.time()

        # Create the smolagent system
        model, manager_agent = create_smolagent_system(current_model)

        load_time = time.time() - start_time
        print(f"✅ System loaded successfully in {load_time:.2f} seconds")

        print(f"\n{'='*60}")
        print(f"✅ Multi-agent system initialized successfully!")
        print(f"   Model: {current_model}")
        print(f"   Load time: {load_time:.2f}s")
        print(f"{'='*60}\n")

        return True, f"System initialized with model: {current_model}"

    except Exception as e:
        error_msg = f"Failed to initialize system: {e}"
        print(f"\n{'='*60}")
        print(f"❌ ERROR: {error_msg}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        return False, error_msg


def process_user_request(message: str, uploaded_file, model_choice: str) -> Tuple[str, str, str]:
    """
    Process user request through the smolagent multi-agent system.

    Args:
        message: User's message/request
        uploaded_file: Uploaded file (if any)
        model_choice: Selected LLM model

    Returns:
        Tuple of (response_text, logs_text, status_message)
    """
    global manager_agent, CONFIG

    try:
        # Validate inputs
        if not message or message.strip() == "":
            return "", "", "Please enter a question or request."

        # Determine file path
        file_path = None
        status_msg = "Using default Titanic dataset"

        if uploaded_file is not None:
            # Create local directory structure for uploaded files
            local_data_dir = Path("tmp/data")
            local_data_dir.mkdir(parents=True, exist_ok=True)

            # Check file format before copying
            if not uploaded_file.name.endswith(('.csv', '.json')):
                error_msg = "Error: Please upload a CSV or JSON file."
                return "", "", error_msg

            try:
                # Get original filename and create local path
                original_filename = os.path.basename(uploaded_file.name)
                local_file_path = local_data_dir / original_filename

                # Copy the uploaded file to local directory
                shutil.copy2(uploaded_file.name, local_file_path)

                # Verify the file was copied successfully
                if not local_file_path.exists():
                    error_msg = "Error: Failed to save uploaded file to local directory."
                    return "", "", error_msg

                file_path = str(local_file_path)
                status_msg = f"Processing with uploaded file: {original_filename}"

                print(f"✅ File saved to local directory:")
                print(f"   Path: {file_path}")
                print(f"   Size: {os.path.getsize(file_path):,} bytes")

            except Exception as e:
                error_msg = f"Error: Failed to save uploaded file: {str(e)}"
                return "", "", error_msg
        else:
            # Use default Titanic dataset
            file_path = CONFIG["default_dataset"]
            if not os.path.exists(file_path):
                error_msg = "Error: Default dataset not found. Please upload a file."
                return "", "", error_msg

        # Reinitialize if model changed
        if model_choice and model_choice != CONFIG["model_id"]:
            status_msg += f"\n\nReinitializing system with model: {model_choice}..."
            success, init_msg = initialize_system(model_choice)
            if not success:
                return "", "", f"Error: {init_msg}"
            status_msg += f"\n{init_msg}"

        # Check manager agent
        if manager_agent is None:
            error_msg = "Error: System not initialized. Please refresh the page."
            return "", "", error_msg

        status_msg += f"\n\nProcessing request: '{message}...'"

        # Capture stdout/stderr for print statements
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            print(f"\n{'='*60}")
            print(f"🚀 STARTING ANALYSIS (smolagents)")
            print(f"   Prompt: {message}")
            print(f"   File: {file_path}")
            print(f"{'='*60}\n")

            # Format the task for the manager
            task = f"""
Task: {message}
Data file: {file_path}

IMPORTANT INSTRUCTIONS:
- You MUST ONLY use the tools and agents that have been explicitly provided to you
- DO NOT invent, create, or imagine new tools, functions, or agents
- DO NOT attempt to use tools or methods that are not in the provided list
- If you cannot complete a task with the available tools, state this clearly

You have access to ONLY these specialized agents:
- data_reader: For reading and exploring the dataset (tools: read_csv_file, read_json_file, get_column_info, get_data_summary, preview_data)
- data_manipulation: For data cleaning and preprocessing (tools: handle_missing_values, create_dummy_variables, modify_column_values, convert_data_types)
- data_operations: For mathematical operations and aggregations (tools: filter_data, perform_math_operations, aggregate_data, string_operations)
- ml_prediction: For training and evaluating machine learning models (tools: train_regression_model, train_svm_model, train_random_forest_model, train_knn_model, evaluate_model)

Analyze the task and delegate work ONLY to the appropriate agents listed above in the correct order.
Use ONLY the tools available to each agent. Do ONLY what is required of you, nothing more.
Provide a comprehensive final answer with all results.
"""

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Run the manager agent
                result = manager_agent.run(task)

            print(f"\n{'='*60}")
            print(f"✅ ANALYSIS COMPLETED")
            print(f"{'='*60}\n")

            # Format the response
            response = f"## 🎯 Analysis Results\n\n{result}\n"

            status_msg += "\n\n✅ Request processed successfully!"

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ ANALYSIS FAILED")
            print(f"   Error: {str(e)}")
            print(f"{'='*60}\n")
            import traceback
            traceback.print_exc()

            error_msg = str(e)
            response = f"❌ **Error:**\n{error_msg}"
            status_msg += f"\n\n❌ Error: {error_msg}"

        # Combine all logs
        logs = ""

        # Add stdout logs
        stdout_content = stdout_capture.getvalue()
        if stdout_content:
            logs += f"=== STDOUT (Print Statements) ===\n{stdout_content}\n"

        # Add stderr if any
        stderr_content = stderr_capture.getvalue()
        if stderr_content:
            logs += f"\n=== STDERR (Errors & Warnings) ===\n{stderr_content}\n"

        if not logs.strip():
            logs = "⚠️ No logs captured during execution."

        return response, logs, status_msg

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        import traceback
        traceback.print_exc()
        return f"❌ **Error:**\n{error_msg}", "", f"❌ {error_msg}"


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
        ["Provide summary statistics for numerical columns"],

        ["Calculate the average age by gender"],
        ["Create dummy variables for categorical columns and handle missing values"],
        ["Filter passengers who survived and analyze their characteristics"],

        ["Train a random forest model to predict survival and show feature importance"],
        ["Compare the performance of different ML models for classification"],
        ["Perform a complete analysis: data exploration, preprocessing, and predictive modeling"]
    ]


def create_interface():
    """Create the Gradio interface with enhanced user experience."""

    with gr.Blocks(
        title="Multi-Agent Data Analysis System (smolagents)",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            padding: 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .response-box {
            border: 2px solid #667eea;
            border-radius: 10px;
            padding: 15px;
            background-color: #f8f9ff;
        }
        .logs-box {
            border: 2px solid #666;
            border-radius: 10px;
            padding: 15px;
            background-color: #f5f5f5;
            font-family: monospace;
        }
        """
    ) as interface:

        # Header
        gr.HTML("""
            <div class="header">
                <h1>🤖 Multi-Agent Data Analysis System</h1>
                <p style="font-size: 18px; margin-top: 10px;">
                    Powered by smolagents and HuggingFace Transformers
                </p>
            </div>
        """)

        # Main layout
        with gr.Row():
            # Left Column - Configuration & Upload
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")

                model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    value=CONFIG["model_id"],
                    label="Select LLM Model",
                    info="Choose the language model for analysis"
                )

                gr.Markdown("### 📁 Data Upload")

                file_upload = gr.File(
                    label="Drag & Drop CSV or JSON",
                    file_types=[".csv", ".json"],
                    type="filepath"
                )

                file_info = gr.Markdown(
                    "**Status:** No file uploaded yet\n\n"
                    "**Default:** Titanic dataset will be used"
                )

                gr.Markdown("### 💡 Sample Questions")
                gr.Markdown("*Click on a question to use it*")

                sample_prompts = get_sample_prompts()

                # Store button references
                sample_buttons = []

                gr.Markdown("**Simple Analysis:**")
                for prompt in sample_prompts[:3]:
                    btn = gr.Button(prompt[0], size="sm")
                    sample_buttons.append((btn, prompt[0]))

                gr.Markdown("**Data Processing:**")
                for prompt in sample_prompts[3:6]:
                    btn = gr.Button(prompt[0], size="sm")
                    sample_buttons.append((btn, prompt[0]))

                gr.Markdown("**Advanced ML:**")
                for prompt in sample_prompts[6:]:
                    btn = gr.Button(prompt[0], size="sm")
                    sample_buttons.append((btn, prompt[0]))

            # Right Column - Query & Results
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Ask Your Question")

                message_input = gr.Textbox(
                    placeholder="Enter your data analysis question here (e.g., 'Show me survival rates by passenger class')",
                    label="Your Question",
                    lines=3,
                    max_lines=5
                )

                submit_btn = gr.Button(
                    "🚀 Analyze Data",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown("---")
                gr.Markdown("### 📊 Response")

                response_output = gr.Textbox(
                    label="Analysis Results",
                    value="",
                    placeholder="",
                    lines=12,
                    max_lines=20,
                    show_copy_button=True,
                    elem_classes="response-box"
                )

                gr.Markdown("### 🔧 System Logs & Agent Output")

                logs_output = gr.Textbox(
                    label="Execution Logs",
                    lines=8,
                    max_lines=15,
                    show_copy_button=True,
                    elem_classes="logs-box"
                )

                status_output = gr.Markdown(
                    "**Status:** Ready to process requests"
                )

                gr.Markdown("""
                ---
                **Available Agents (smolagents):**
                - 📊 **Data Reader**: Dataset analysis, column info, summaries
                - 🔧 **Data Manipulation**: Data preprocessing, feature engineering
                - ⚡ **Data Operations**: Filtering, aggregation, calculations
                - 🎯 **ML Prediction**: Model training and evaluation
                - 👔 **Manager**: Orchestrates all specialized agents
                """)

        # Event Handlers
        file_upload.change(
            fn=handle_file_upload,
            inputs=file_upload,
            outputs=file_info
        )

        submit_btn.click(
            fn=process_user_request,
            inputs=[message_input, file_upload, model_dropdown],
            outputs=[response_output, logs_output, status_output]
        )

        message_input.submit(
            fn=process_user_request,
            inputs=[message_input, file_upload, model_dropdown],
            outputs=[response_output, logs_output, status_output]
        )

        # Add click handlers for sample question buttons
        for btn, prompt_text in sample_buttons:
            btn.click(
                fn=lambda text=prompt_text: text,
                inputs=None,
                outputs=message_input
            )

    return interface


def main():
    """Main function to run the Gradio app."""

    print("\n" + "="*60)
    print("🚀 Starting Multi-Agent Data Analysis System (smolagents)")
    print("="*60 + "\n")

    # Initialize the system with default model
    print(f"📦 Initializing with default model: {CONFIG['model_id']}")
    success, msg = initialize_system()

    if not success:
        print(f"❌ Failed to initialize system: {msg}")
        print("⚠️  The interface will launch, but model loading will happen on first use.")
    else:
        print(f"✅ {msg}\n")

    # Create and launch the interface
    interface = create_interface()

    print("\n" + "="*60)
    print("🌐 Launching Gradio Interface")
    print("="*60)
    #print(f"📍 Local URL: http://localhost:7861")
    #print(f"📍 Network URL: http://0.0.0.0:7861")
    #print("="*60 + "\n")

    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port to avoid conflict with app.py
        share=True,  # Set to True to create public link
        show_api=False,
        debug=True,
        max_file_size="500mb"  # Allow larger files
    )


if __name__ == "__main__":
    main()
