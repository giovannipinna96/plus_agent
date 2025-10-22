"""Test script to verify planner agent fixes."""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.planner_agent import PlannerAgent
from core.config import config

def test_planner():
    """Test the planner agent with the example prompt."""
    print("="*60)
    print("Testing Planner Agent with Fixed Prompt")
    print("="*60)

    # Initialize planner
    print("\n1. Initializing PlannerAgent...")
    try:
        planner = PlannerAgent()
        print("✅ PlannerAgent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize PlannerAgent: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test with the user's example prompt
    test_prompt = "Show me the basic information about this dataset"

    print(f"\n2. Testing with prompt: '{test_prompt}'")
    print("-"*60)

    try:
        result = planner.plan(test_prompt)

        print(f"\n3. Result:")
        print("-"*60)
        print(f"Status: {result.get('status')}")

        if result.get('status') == 'success':
            print(f"\n✅ SUCCESS!")
            print(f"\nExecution Plan:")
            print(result.get('plan', 'No plan generated'))
        else:
            print(f"\n❌ ERROR!")
            print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n❌ EXCEPTION during planning:")
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

    test_planner()
