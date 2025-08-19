#!/bin/bash
#SBATCH --job-name=test_single_agents
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_single_agents_%j.out
#SBATCH --error=logs/test_single_agents_%j.err

# Script per testare i singoli agenti
echo "=== Test Single Agents ==="
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

cd plus_agent

echo "Testing individual agents..."
python3 -c "
import sys
sys.path.append('.')
from plus_agent.agents.planner_agent import PlannerAgent
from plus_agent.agents.data_reader_agent import DataReaderAgent
from plus_agent.agents.data_manipulation_agent import DataManipulationAgent
from plus_agent.agents.data_operations_agent import DataOperationsAgent
from plus_agent.agents.ml_prediction_agent import MLPredictionAgent

file_path = 'data/titanic.csv'

print('=== Testing Planner Agent ===')
planner = PlannerAgent()
result = planner.plan('Analyze the Titanic dataset and train a model to predict survival')
print('Planner result:', result)

print('\\n=== Testing Data Reader Agent ===')
reader = DataReaderAgent()
result = reader.analyze_data(file_path, 'basic')
print('Reader result:', result)

print('\\n=== Testing Data Manipulation Agent ===')
manipulator = DataManipulationAgent()
result = manipulator.manipulate_data(file_path, 'Handle missing values in age column')
print('Manipulator result:', result)

print('\\n=== Testing Data Operations Agent ===')
operations = DataOperationsAgent()
result = operations.perform_operations(file_path, 'Calculate survival rate by gender')
print('Operations result:', result)

print('\\n=== Testing ML Prediction Agent ===')
ml_agent = MLPredictionAgent()
result = ml_agent.train_model(file_path, 'Train a random forest to predict survival')
print('ML Agent result:', result)

print('\\n=== All single agents tests completed successfully! ===')
"

echo "End time: $(date)"
echo "=== Test completed ==="