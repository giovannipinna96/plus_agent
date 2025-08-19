#!/bin/bash
#SBATCH --job-name=test_operations_tools
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/test_operations_tools_%j.out
#SBATCH --error=logs/test_operations_tools_%j.err

# Script per testare gli operations tools
echo "=== Test Operations Tools ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Setup environment
module load Python/3.9-GCCcore-11.2.0
source venv/bin/activate

# Create logs directory
mkdir -p logs

cd plus_agent

echo "Testing operations tools..."
python3 -c "
import sys
sys.path.append('.')
from plus_agent.tools.operations_tools import filter_data, perform_math_operations, aggregate_data

file_path = 'data/titanic.csv'

print('=== Testing filter_data ===')
result = filter_data(file_path, 'age > 30')
print(result)

print('\\n=== Testing perform_math_operations ===')
result = perform_math_operations(file_path, 'mean', 'age')
print(result)

print('\\n=== Testing aggregate_data ===')
result = aggregate_data(file_path, 'sex', 'age', 'mean')
print(result)

print('\\n=== Testing complex filtering ===')
result = filter_data(file_path, 'pclass == 1 and sex == \"female\"')
print(result)

print('\\n=== Testing multiple math operations ===')
result = perform_math_operations(file_path, 'std', 'fare')
print(result)

print('\\n=== All operations tools tests completed successfully! ===')
"

echo "End time: $(date)"
echo "=== Test completed ==="