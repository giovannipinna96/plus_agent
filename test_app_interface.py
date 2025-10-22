"""Quick test to verify the Gradio interface builds correctly."""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing Gradio interface construction...")

try:
    # Import the app module
    from app import create_interface, get_available_models

    print("✅ Successfully imported app module")

    # Test get_available_models
    models = get_available_models()
    print(f"✅ Available models: {len(models)} models found")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model}")

    # Test interface creation (without launching)
    print("\nBuilding Gradio interface...")
    interface = create_interface()
    print("✅ Interface created successfully!")

    # Check interface components
    print("\n✅ Interface test passed!")
    print("\nTo launch the interface, run:")
    print("   python app.py")
    print("\nor:")
    print("   uv run python app.py")

except Exception as e:
    print(f"❌ Error during test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
