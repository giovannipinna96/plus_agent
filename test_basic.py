#!/usr/bin/env python3
"""
Basic functionality test for Plus-Agent system.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing core imports...")

    try:
        from core.config import config
        print("✅ Config imported successfully")
        print(f"   Model: {config.model_name}")
        print(f"   Device: {config.device}")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

    try:
        from core.langsmith_integration import langsmith_logger
        print("✅ LangSmith integration imported")
    except Exception as e:
        print(f"❌ LangSmith integration import failed: {e}")
        return False

    try:
        from tools.data_tools import read_csv_file
        print("✅ Data tools imported")
    except Exception as e:
        print(f"❌ Data tools import failed: {e}")
        return False

    return True

def test_dataset():
    """Test dataset availability."""
    print("\n📊 Testing dataset...")

    from core.config import config

    if os.path.exists(config.default_dataset_path):
        print(f"✅ Dataset found: {config.default_dataset_path}")
        return True
    else:
        print(f"❌ Dataset not found: {config.default_dataset_path}")
        return False

def test_basic_tool():
    """Test a basic tool functionality."""
    print("\n🔧 Testing basic tool functionality...")

    try:
        from tools.data_tools import read_csv_file
        from core.config import config

        # Use invoke method instead of direct call
        result = read_csv_file.invoke({"file_path": config.default_dataset_path})
        if isinstance(result, str) and "CSV file loaded successfully" in result:
            print("✅ Basic tool test passed")
            print(f"   Result: {result[:100]}...")
            return True
        elif isinstance(result, dict) and result.get("status") == "success":
            print("✅ Basic tool test passed")
            print(f"   Dataset shape: {result.get('shape', 'Unknown')}")
            return True
        else:
            print(f"❌ Basic tool test failed: {result}")
            return False
    except Exception as e:
        print(f"❌ Basic tool test exception: {e}")
        return False

def main():
    """Run basic system tests."""
    print("🚀 Plus-Agent Basic System Test")
    print("=" * 40)

    tests = [
        ("Core Imports", test_imports),
        ("Dataset Availability", test_dataset),
        ("Basic Tool", test_basic_tool)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"💥 {test_name} failed!")

    print(f"\n{'='*40}")
    print(f"📊 RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All basic tests passed! System is ready.")
        return True
    else:
        print("❌ Some basic tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)