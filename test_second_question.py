"""Test script for the second question with detailed logging."""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import MultiAgentOrchestrator
from core.config import config

def test_second_question():
    """Test with the second question that was failing."""
    print("="*60)
    print("Testing Second Question with Detailed Logging")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Max tokens: {config.max_tokens}")
    print(f"  Temperature: {config.temperature}")

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

    # Test with the second question
    test_prompt = "What are the data types and missing values in each column?"
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
                    # Show result based on agent type
                    if 'plan' in agent_result:
                        content = agent_result['plan']
                    elif 'analysis' in agent_result:
                        content = agent_result['analysis']
                    elif 'result' in agent_result:
                        content = agent_result['result']
                    else:
                        content = str(agent_result)

                    print(f"    Content length: {len(content)} chars")
                    print(f"    Preview: {content[:300]}...")
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
    test_second_question()
