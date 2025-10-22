"""Test script to verify all agent fixes work end-to-end."""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import MultiAgentOrchestrator
from core.config import config

def test_end_to_end():
    """Test the complete multi-agent system with fixed agents."""
    print("="*60)
    print("Testing Complete Multi-Agent System with Fixed Agents")
    print("="*60)

    # Initialize orchestrator
    print("\n1. Initializing MultiAgentOrchestrator...")
    try:
        orchestrator = MultiAgentOrchestrator()
        print("✅ Orchestrator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test with the user's example prompt
    test_prompt = "Show me the basic information about this dataset"
    test_file = "data/titanic.csv"

    print(f"\n2. Testing with prompt: '{test_prompt}'")
    print(f"   File: {test_file}")
    print("-"*60)

    try:
        result = orchestrator.run_analysis(test_prompt, test_file)

        print(f"\n3. Result:")
        print("-"*60)
        print(f"Status: {result.get('status')}")

        if result.get('status') == 'success':
            print(f"\n✅ SUCCESS!")
            print(f"\nExecution Plan:")
            print(result.get('execution_plan', 'No plan'))

            print(f"\nCompleted Steps:")
            for step in result.get('completed_steps', []):
                print(f"  - {step}")

            print(f"\nAgent Results:")
            for agent_name, agent_result in result.get('agent_results', {}).items():
                print(f"\n  {agent_name}:")
                print(f"    Status: {agent_result.get('status')}")
                if agent_result.get('status') == 'success':
                    # Show a preview of the result
                    if 'plan' in agent_result:
                        preview = agent_result['plan'][:200]
                    elif 'analysis' in agent_result:
                        preview = agent_result['analysis'][:200]
                    elif 'result' in agent_result:
                        preview = agent_result['result'][:200]
                    else:
                        preview = str(agent_result)[:200]
                    print(f"    Preview: {preview}...")
                else:
                    print(f"    Error: {agent_result.get('error', 'Unknown error')}")
        else:
            print(f"\n❌ ERROR!")
            print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n❌ EXCEPTION during analysis:")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Test completed")
    print("="*60)


if __name__ == "__main__":
    # Setup config
    print(f"Model: {config.model_name}")
    print(f"Temperature: {config.temperature}")
    print(f"Max tokens: {config.max_tokens}")
    print()

    test_end_to_end()
