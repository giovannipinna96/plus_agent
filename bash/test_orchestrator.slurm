#!/bin/bash
#SBATCH --job-name=test_orchestrator
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/test_orchestrator_%j.out
#SBATCH --error=logs/test_orchestrator_%j.err

# Script per testare l'orchestratore multi-agent
echo "=== Test Multi-Agent Orchestrator ==="
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

echo "Testing multi-agent orchestration..."
python3 -c "
import sys
sys.path.append('.')
from plus_agent.core.orchestrator import MultiAgentOrchestrator

file_path = 'data/titanic.csv'
orchestrator = MultiAgentOrchestrator()

print('=== Test 1: Simple Data Analysis ===')
result = orchestrator.run_analysis('Show me basic information about this dataset', file_path)
print('Result status:', result['status'])
print('Messages:', result.get('messages', []))
print('Completed steps:', result.get('completed_steps', []))

print('\\n=== Test 2: Data Operations ===')
result = orchestrator.run_analysis('Calculate the average age by gender', file_path)
print('Result status:', result['status'])
print('Messages:', result.get('messages', []))
print('Completed steps:', result.get('completed_steps', []))

print('\\n=== Test 3: Machine Learning Task ===')
result = orchestrator.run_analysis('Train a model to predict survival', file_path)
print('Result status:', result['status'])
print('Messages:', result.get('messages', []))
print('Completed steps:', result.get('completed_steps', []))

print('\\n=== Test 4: Complex Workflow ===')
result = orchestrator.run_analysis('Analyze the dataset, handle missing values, and train a classification model', file_path)
print('Result status:', result['status'])
print('Messages:', result.get('messages', []))
print('Completed steps:', result.get('completed_steps', []))

print('\\n=== All orchestrator tests completed successfully! ===')
"

echo "End time: $(date)"
echo "=== Test completed ==="