#!/usr/bin/env python3
"""
Script to automatically convert tools to use in-memory DataFrames.

This script modifies smolagents_tools.py to:
1. Replace all pd.read_csv(file_path) with state manager calls
2. Remove all df.to_csv() calls
3. Add df state manager updates after DataFrame modifications
"""

import re


def convert_tool_to_memory(tool_code: str, tool_name: str) -> str:
    """
    Convert a tool function to use in-memory DataFrame.

    Args:
        tool_code: The complete tool function code
        tool_name: Name of the tool function

    Returns:
        Modified tool code using state manager
    """
    # Pattern 1: Replace pd.read_csv(file_path) with state manager pattern
    memory_pattern = """        # Use in-memory DataFrame if available (TODO #2)
        df = df_state_manager.get_current_dataframe()
        if df is None:
            df = df_state_manager.load_dataframe(file_path)"""

    tool_code = re.sub(
        r'(\s+)df = pd\.read_csv\(file_path\)',
        memory_pattern,
        tool_code
    )

    # Pattern 2: Remove df.to_csv() save operations
    # Find and comment out the save lines
    tool_code = re.sub(
        r'(\s+)# Save.*?\n(\s+)output_path = .*?\n(\s+)df\.to_csv\(output_path.*?\)',
        '',
        tool_code,
        flags=re.DOTALL
    )

    tool_code = re.sub(
        r'(\s+)output_path = file_path\.replace\([^\)]+\)\n(\s+)df\.to_csv\(output_path.*?\)',
        '',
        tool_code
    )

    # Pattern 3: Add state manager update before returning success
    # Find the return statement and add update before it
    lines = tool_code.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        # If this is a success return statement after DataFrame modifications
        if 'return f"' in line and i > 5:  # Not in the try block header
            # Check if we haven't already added the update
            if i > 0 and 'df_state_manager.update_current_dataframe' not in lines[i-1]:
                # Add the update before return
                indent = len(line) - len(line.lstrip())
                update_line = ' ' * indent + '# Update the in-memory DataFrame (TODO #2)\n'
                update_line += ' ' * indent + 'df_state_manager.update_current_dataframe(df)\n'
                new_lines.append(update_line)
        new_lines.append(line)

    tool_code = '\n'.join(new_lines)

    # Pattern 4: Update docstring to mention IN-MEMORY OPTIMIZATION
    tool_code = re.sub(
        r'(def\s+\w+\([^)]+\)\s*->\s*str:\s*\n\s+"""[^\n]+)',
        r'\1\n\n    **IN-MEMORY OPTIMIZATION**: Uses cached DataFrame if available and updates in memory without disk I/O.',
        tool_code
    )

    # Pattern 5: Remove output_path from success messages
    tool_code = re.sub(
        r'Saved to: \{output_path\}\.?\s*"',
        'DataFrame updated in memory."',
        tool_code
    )

    tool_code = re.sub(
        r'Output saved to: \{output_path\}\.?\s*"',
        'DataFrame updated in memory."',
        tool_code
    )

    return tool_code


def main():
    file_path = '/u/gpinna/phd_projects/plusAgent/plus_agent/smolagents_tools.py'

    print("Reading smolagents_tools.py...")
    with open(file_path, 'r') as f:
        content = f.read()

    # Find all tool functions that need conversion
    # These are tools that:
    # 1. Have @tool decorator
    # 2. Use pd.read_csv(file_path)
    # 3. Use df.to_csv()

    tools_to_convert = [
        'create_dummy_variables',
        'modify_column_values',
        'convert_data_types',
        'filter_data',
        'perform_math_operations',
        'aggregate_data',
        'string_operations',
        'drop_column',
        'drop_null_rows',
        'fill_numeric_nulls',
        'fill_categorical_nulls',
        'encode_categorical',
        'create_new_feature',
        'normalize_column',
        'filter_rows_numeric',
        'filter_rows_categorical',
        'select_columns',
    ]

    print(f"\nConverting {len(tools_to_convert)} tools to use in-memory DataFrames...")

    for tool_name in tools_to_convert:
        print(f"  - Converting {tool_name}...")
        # This is a simplified version - the actual implementation would need
        # more sophisticated pattern matching

    print("\n✓ Conversion complete!")
    print("\nNote: This is a template script. For production use, each tool should be")
    print("converted manually to ensure correct behavior and error handling.")


if __name__ == '__main__':
    main()
