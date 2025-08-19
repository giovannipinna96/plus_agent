#!/bin/bash
#SBATCH --job-name=test_llm_wrapper
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_llm_wrapper_%j.out
#SBATCH --error=logs/test_llm_wrapper_%j.err

# Script per testare il wrapper LLM
echo "=== Test LLM Wrapper ==="
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
export MAX_TOKENS="256"
export TEMPERATURE="0.1"

cd plus_agent

echo "Testing LLM wrapper initialization..."
python3 -c "
import sys
sys.path.append('.')
from plus_agent.core.llm_wrapper import llm_wrapper
from plus_agent.core.config import config

print('Config loaded:', config.model_name)
print('Getting LLM for planner agent...')
llm = llm_wrapper.get_llm_for_agent('planner')
print('LLM created successfully:', type(llm))

print('Testing simple generation...')
result = llm._call('What is 2+2? Answer briefly.')
print('LLM response:', result)
print('Test completed successfully!')
"

echo "End time: $(date)"
echo "=== Test completed ==="