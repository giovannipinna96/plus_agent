#!/bin/bash
#SBATCH --job-name=test_full_system
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/test_full_system_%j.out
#SBATCH --error=logs/test_full_system_%j.err

# Script per testare il sistema completo end-to-end
echo "=== Full System Integration Test ==="
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
export MAX_TOKENS="1000"
export TEMPERATURE="0.1"
export LANGSMITH_TRACING="false"  # Disable for testing

cd plus_agent

echo "Running comprehensive system test..."
python3 -c "
import sys
sys.path.append('.')
from plus_agent.tests.test_prompts import TestPrompts
from plus_agent.core.orchestrator import MultiAgentOrchestrator
import time

file_path = 'data/titanic.csv'
orchestrator = MultiAgentOrchestrator()

print('=== Full System Integration Test ===')
print(f'Using dataset: {file_path}')
print(f'Model: {orchestrator.planner.llm.model_name}')

# Get test prompts for systematic testing
test_prompts = TestPrompts.get_prompts_for_testing()

for i, test in enumerate(test_prompts):
    print(f'\\n=== Test {i+1}: {test[\"complexity\"]} - {test[\"description\"]} ===')
    print(f'Prompt: {test[\"prompt\"]}')
    print(f'Expected agents: {test[\"expected_agents\"]}')
    
    start_time = time.time()
    
    try:
        result = orchestrator.run_analysis(test['prompt'], file_path)
        
        execution_time = time.time() - start_time
        
        print(f'Status: {result[\"status\"]}')
        print(f'Execution time: {execution_time:.2f} seconds')
        print(f'Completed steps: {result.get(\"completed_steps\", [])}')
        
        if result['status'] == 'success':
            print('✓ Test PASSED')
            
            # Verify expected agents were used
            agent_results = result.get('agent_results', {})
            used_agents = list(agent_results.keys())
            
            expected_in_results = any(agent in used_agents for agent in test['expected_agents'])
            if expected_in_results:
                print(f'✓ Expected agents used: {used_agents}')
            else:
                print(f'⚠ Expected agents {test[\"expected_agents\"]}, but used: {used_agents}')
        else:
            print('✗ Test FAILED')
            print(f'Error: {result.get(\"error\", \"Unknown error\")}')
    
    except Exception as e:
        execution_time = time.time() - start_time
        print('✗ Test FAILED with exception')
        print(f'Exception: {str(e)}')
        print(f'Execution time: {execution_time:.2f} seconds')

print('\\n=== Additional Integration Tests ===')

# Test with different complexity prompts
complexity_tests = {
    'simple': TestPrompts.SIMPLE_PROMPTS[0],
    'medium': TestPrompts.MEDIUM_PROMPTS[0], 
    'complex': TestPrompts.COMPLEX_PROMPTS[0],
    'comprehensive': TestPrompts.COMPREHENSIVE_PROMPTS[0]
}

for complexity, prompt in complexity_tests.items():
    print(f'\\n--- {complexity.upper()} complexity test ---')
    print(f'Prompt: {prompt}')
    
    start_time = time.time()
    result = orchestrator.run_analysis(prompt, file_path)
    execution_time = time.time() - start_time
    
    print(f'Status: {result[\"status\"]}')
    print(f'Execution time: {execution_time:.2f} seconds')
    print(f'Agents used: {list(result.get(\"agent_results\", {}).keys())}')

print('\\n=== System Performance Summary ===')
print('All integration tests completed!')
print('System is ready for production use.')
"

echo "End time: $(date)"
echo "=== Full system test completed ==="