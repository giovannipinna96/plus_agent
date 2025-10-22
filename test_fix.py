"""Quick test script to verify data manipulation agent fixes."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.data_manipulation_agent import DataManipulationAgent

def test_data_manipulation():
    """Test the data manipulation agent with a simple task."""
    print("=" * 60)
    print("Testing Data Manipulation Agent Fixes")
    print("=" * 60)

    # Initialize the agent
    print("\n1. Initializing Data Manipulation Agent...")
    try:
        agent = DataManipulationAgent()
        print("   ✅ Agent initialized successfully")
    except Exception as e:
        print(f"   ❌ Error initializing agent: {e}")
        return False

    # Test file path (using the titanic dataset)
    test_file = "data/titanic.csv"

    # Check if file exists
    if not os.path.exists(test_file):
        print(f"\n❌ Test file not found: {test_file}")
        print("   Please run: python data/download_titanic.py")
        return False

    print(f"\n2. Testing with file: {test_file}")

    # Test the manipulation with a simple request
    print("\n3. Running manipulation test...")
    print("   Request: Handle missing values in Age column using median")

    try:
        result = agent.manipulate_data(
            file_path=test_file,
            manipulation_request="Handle missing values in Age column using median"
        )

        print(f"\n4. Result:")
        print(f"   Status: {result.get('status', 'unknown')}")

        if result.get('status') == 'success':
            print(f"   ✅ SUCCESS!")
            print(f"\n   Output (first 500 chars):")
            print(f"   {result.get('result', 'No result')[:500]}")
            return True
        else:
            print(f"   ❌ FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"\n   ❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🔧 Data Manipulation Agent Fix Test\n")

    success = test_data_manipulation()

    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ TESTS FAILED")
    print("=" * 60)

    sys.exit(0 if success else 1)
