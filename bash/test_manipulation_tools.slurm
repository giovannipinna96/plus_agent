#!/bin/bash
#SBATCH --job-name=test_manipulation_tools
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/test_manipulation_tools_%j.out
#SBATCH --error=logs/test_manipulation_tools_%j.err

# Script per testare i manipulation tools
echo "=== Test Manipulation Tools ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Setup environment
module load Python/3.9-GCCcore-11.2.0
source venv/bin/activate

# Create logs directory
mkdir -p logs

cd plus_agent

echo "Testing manipulation tools..."
python3 -c "
import sys
sys.path.append('.')
from plus_agent.tools.manipulation_tools import handle_missing_values, create_dummy_variables, convert_data_types
import pandas as pd
import shutil

# Create a test copy of the dataset
original_file = 'data/titanic.csv'
test_file = 'data/titanic_test.csv'
shutil.copy2(original_file, test_file)

print('=== Testing handle_missing_values ===')
result = handle_missing_values(test_file, 'age', 'mean')
print(result)

print('\\n=== Testing create_dummy_variables ===')
result = create_dummy_variables(test_file, 'sex', 'one_hot')
print(result)

print('\\n=== Testing convert_data_types ===')
result = convert_data_types(test_file, 'pclass', 'str')
print(result)

print('\\n=== Checking final dataset ===')
df = pd.read_csv(test_file)
print(f'Dataset shape: {df.shape}')
print(f'Columns: {list(df.columns)[:10]}...')
print(f'Data types: {df.dtypes.to_dict()}')

print('\\n=== All manipulation tools tests completed successfully! ===')
"

echo "End time: $(date)"
echo "=== Test completed ==="