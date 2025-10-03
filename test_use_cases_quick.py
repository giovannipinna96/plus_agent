#!/usr/bin/env python3
"""
Quick Use Cases Test - Tests use case creation and basic structure without full execution.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from use_cases import UseCaseLibrary

def test_use_case_creation():
    """Test that all use cases can be created properly."""
    print("🧪 Testing Use Case Creation")
    print("=" * 40)

    try:
        # Test Use Case 1
        uc1 = UseCaseLibrary.get_use_case_1_basic_analysis()
        print(f"✅ Use Case 1: {uc1.name}")
        print(f"   Complexity: {uc1.complexity}")
        print(f"   Prompts: {len(uc1.prompts)}")
        print(f"   Description: {uc1.description[:80]}...")

        # Test Use Case 2
        uc2 = UseCaseLibrary.get_use_case_2_preprocessing_pipeline()
        print(f"\n✅ Use Case 2: {uc2.name}")
        print(f"   Complexity: {uc2.complexity}")
        print(f"   Prompts: {len(uc2.prompts)}")
        print(f"   Description: {uc2.description[:80]}...")

        # Test Use Case 3
        uc3 = UseCaseLibrary.get_use_case_3_complete_ml_workflow()
        print(f"\n✅ Use Case 3: {uc3.name}")
        print(f"   Complexity: {uc3.complexity}")
        print(f"   Prompts: {len(uc3.prompts)}")
        print(f"   Description: {uc3.description[:80]}...")

        # Test all use cases
        all_cases = UseCaseLibrary.get_all_use_cases()
        print(f"\n📊 Total Use Cases: {len(all_cases)}")

        return True

    except Exception as e:
        print(f"❌ Use case creation failed: {e}")
        return False

def test_use_case_structure():
    """Test the structure and content of use cases."""
    print("\n🔍 Testing Use Case Structure")
    print("=" * 40)

    try:
        uc = UseCaseLibrary.get_use_case_1_basic_analysis()

        # Test prompts
        print(f"📝 Sample prompts from Use Case 1:")
        for i, prompt in enumerate(uc.prompts[:3], 1):
            print(f"   {i}. {prompt[:60]}...")

        # Test expected agents
        print(f"\n🤖 Expected agents:")
        for i, agents in enumerate(uc.expected_agents[:3], 1):
            print(f"   Prompt {i}: {', '.join(agents)}")

        return True

    except Exception as e:
        print(f"❌ Use case structure test failed: {e}")
        return False

def main():
    """Run quick use case tests."""
    print("🚀 Plus-Agent Use Cases Quick Test")
    print("=" * 50)

    tests = [
        ("Use Case Creation", test_use_case_creation),
        ("Use Case Structure", test_use_case_structure)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"💥 {test_name} failed!")

    print(f"\n{'='*50}")
    print(f"📊 RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All use case tests passed!")
        print("\n💡 Use Cases Summary:")
        print("   1. Basic Data Analysis (5 prompts, simple complexity)")
        print("   2. Data Preprocessing Pipeline (6 prompts, medium complexity)")
        print("   3. Complete ML Workflow (7 prompts, complex complexity)")
        print("\n✅ All use cases are ready for execution!")
        return True
    else:
        print("❌ Some use case tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)