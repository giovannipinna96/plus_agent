#!/bin/bash
#SBATCH --job-name=test_ml_tools
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_ml_tools_%j.out
#SBATCH --error=logs/test_ml_tools_%j.err

# Script per testare i ML tools
echo "=== Test ML Tools ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Setup environment
module load Python/3.9-GCCcore-11.2.0
source venv/bin/activate

# Create logs directory
mkdir -p logs

cd plus_agent

echo "Testing ML tools..."
python3 -c "
import sys
sys.path.append('.')
from plus_agent.tools.ml_tools import train_random_forest_model, train_svm_model, train_knn_model, evaluate_model
import pandas as pd

file_path = 'data/titanic.csv'
target_col = 'survived'
features = 'pclass,age,sex,fare'

print('=== Testing Random Forest Model ===')
result = train_random_forest_model(file_path, target_col, features)
print(result)

print('\\n=== Testing SVM Model ===')
result = train_svm_model(file_path, target_col, features, 'classification')
print(result)

print('\\n=== Testing K-NN Model ===')
result = train_knn_model(file_path, target_col, features, 'classification')
print(result)

print('\\n=== Testing Model Evaluation ===')
# First train a model to get a model file
train_random_forest_model(file_path, target_col, features)
result = evaluate_model('trained_models/random_forest_model.pkl', file_path, target_col, features)
print(result)

print('\\n=== All ML tools tests completed successfully! ===')
"

echo "End time: $(date)"
echo "=== Test completed ==="