#!/bin/bash
#SBATCH --job-name=test_gradio_interface
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_gradio_interface_%j.out
#SBATCH --error=logs/test_gradio_interface_%j.err

# Script per testare l'interfaccia Gradio completa
echo "=== Test Gradio Interface ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Setup environment
module load Python/3.9-GCCcore-11.2.0
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Set environment variables
export MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
export DEVICE="auto"
export MAX_TOKENS="512"
export TEMPERATURE="0.1"
export LANGSMITH_TRACING="false"  # Disable for testing

cd plus_agent

echo "Testing Gradio interface initialization..."
python3 -c "
import sys
sys.path.append('.')
from plus_agent.app import initialize_system, process_user_request, handle_file_upload
import tempfile
import pandas as pd

print('=== Testing system initialization ===')
success = initialize_system()
print('System initialized:', success)

if success:
    print('\\n=== Testing file upload handling ===')
    
    # Create a temporary CSV file for testing
    test_data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['New York', 'London', 'Paris']
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_file_path = f.name
    
    # Mock file object for testing
    class MockFile:
        def __init__(self, path):
            self.name = path
    
    mock_file = MockFile(temp_file_path)
    result = handle_file_upload(mock_file)
    print('File upload result:', result)
    
    print('\\n=== Testing user request processing ===')
    test_prompts = [
        'Show me the basic information about this dataset',
        'Calculate summary statistics for numeric columns',
        'What are the column names and data types?'
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f'\\n--- Test prompt {i+1}: {prompt} ---')
        chat_history = []
        result_history, empty_msg, file_info = process_user_request(
            prompt, chat_history, None
        )
        print('Response generated successfully')
        print('File info:', file_info)
        if result_history:
            print('Response preview:', result_history[-1][1][:200] + '...' if len(result_history[-1][1]) > 200 else result_history[-1][1])
    
    print('\\n=== All Gradio interface tests completed successfully! ===')
else:
    print('System initialization failed - skipping interface tests')
"

echo "End time: $(date)"
echo "=== Test completed ==="