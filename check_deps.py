#!/usr/bin/env python3
"""Check which dependencies are installed."""

import importlib
import sys

packages = [
    'langchain', 'langchain_community', 'langgraph', 'langsmith',
    'transformers', 'torch', 'accelerate', 'pandas', 'numpy',
    'sklearn', 'gradio', 'dotenv', 'typing_extensions',
    'huggingface_hub', 'datasets', 'seaborn', 'joblib'
]

# Map package names for display
package_display_names = {
    'sklearn': 'scikit-learn',
    'dotenv': 'python-dotenv',
    'typing_extensions': 'typing-extensions',
    'huggingface_hub': 'huggingface-hub'
}

missing = []
installed = []

for package in packages:
    try:
        importlib.import_module(package)
        display_name = package_display_names.get(package, package)
        installed.append(display_name)
        print(f'✓ {display_name}')
    except ImportError:
        display_name = package_display_names.get(package, package)
        missing.append(display_name)
        print(f'✗ {display_name}')

print(f'\nInstalled: {len(installed)} packages')
print(f'Missing: {len(missing)} packages')
if missing:
    print(f'Missing packages: {", ".join(missing)}')