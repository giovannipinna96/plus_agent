#!/bin/bash
#SBATCH --job-name=test_data_tools
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/test_data_tools_%j.out
#SBATCH --error=logs/test_data_tools_%j.err

# Script per testare i data tools
echo "=== Test Data Tools ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Start time: $(date)"

# Setup environment
module load Python/3.9-GCCcore-11.2.0
source venv/bin/activate

# Create logs directory
mkdir -p logs

cd plus_agent

echo "Testing data tools..."
python3 -c "
import sys
sys.path.append('.')
from plus_agent.tools.data_tools import read_csv_file, get_column_info, get_data_summary, preview_data

# Test with Titanic dataset
file_path = 'data/titanic.csv'

print('=== Testing read_csv_file ===')
result = read_csv_file(file_path)
print(result)

print('\\n=== Testing get_column_info ===')
result = get_column_info(file_path)
print(result)

print('\\n=== Testing preview_data ===')
result = preview_data(file_path, 5)
print(result)

print('\\n=== Testing get_data_summary ===')
result = get_data_summary(file_path)
print(result)

print('\\n=== All data tools tests completed successfully! ===')
"

echo "End time: $(date)"
echo "=== Test completed ==="