#!/usr/bin/env python3
"""
Script di conversione effettiva per tutti i tool rimanenti.
Applica trasformazioni sicure con backup automatico.
"""

import re
import os
from datetime import datetime


def backup_file(file_path):
    """Crea backup con timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_batch_{timestamp}"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return backup_path


def convert_pd_read_csv(content):
    """Converte pd.read_csv() a state manager pattern."""
    pattern = r'(\s+)(df = pd\.read_csv\(file_path\))'
    replacement = r'\1# Use in-memory DataFrame if available (TODO #2)\n\1df = df_state_manager.get_current_dataframe()\n\1if df is None:\n\1    df = df_state_manager.load_dataframe(file_path)'

    return re.sub(pattern, replacement, content)


def remove_csv_saves(content):
    """Rimuove output_path e df.to_csv()."""
    # Pattern 1: Rimuove output_path = ...
    content = re.sub(
        r'\s+output_path = file_path\.replace\([^\n]+\n',
        '',
        content
    )

    # Pattern 2: Rimuove df.to_csv(...)
    content = re.sub(
        r'\s+df\.to_csv\(output_path[^\n]+\n',
        '',
        content
    )

    # Pattern 3: Rimuove filtered_df.to_csv(...)
    content = re.sub(
        r'\s+filtered_df\.to_csv\([^\n]+\n',
        '',
        content
    )

    # Pattern 4: Rimuove result_df.to_csv(...)
    content = re.sub(
        r'\s+result_df\.to_csv\([^\n]+\n',
        '',
        content
    )

    return content


def update_return_messages(content):
    """Aggiorna messaggi di return."""
    # Pattern 1: Saved to: {output_path}
    content = re.sub(
        r'Saved to: \{output_path\}',
        'DataFrame updated in memory',
        content
    )

    # Pattern 2: Output saved to:
    content = re.sub(
        r'Output saved to: \{output_path\}',
        'DataFrame updated in memory',
        content
    )

    # Pattern 3: Path: {output_path}
    content = re.sub(
        r'Path: \{output_path\}',
        'DataFrame updated in memory',
        content
    )

    return content


def add_state_updates_for_manipulation(content):
    """Aggiunge df_state_manager.update_current_dataframe() per tool manipulation."""

    # Lista tool manipulation che modificano DataFrame
    manipulation_ops_tools = [
        'drop_column', 'drop_null_rows', 'fill_numeric_nulls',
        'fill_categorical_nulls', 'encode_categorical',
        'create_new_feature', 'normalize_column',
        'perform_math_operations', 'aggregate_data', 'string_operations',
        'filter_rows_numeric', 'filter_rows_categorical', 'select_columns'
    ]

    for tool_name in manipulation_ops_tools:
        # Pattern: trova il tool e aggiungi update prima del return finale
        pattern = rf'(def {tool_name}\([^)]+\).*?)(return f"[^"]*"(?!\s*\n\s*except))'

        def add_update(match):
            func_body = match.group(1)
            return_stmt = match.group(2)

            # Se già c'è update, skip
            if 'update_current_dataframe' in func_body:
                return match.group(0)

            # Trova l'indentazione del return
            indent_match = re.search(r'\n(\s+)return', match.group(0))
            if not indent_match:
                return match.group(0)

            indent = indent_match.group(1)

            # Aggiungi update prima del return
            update_code = f"\n{indent}# Update the in-memory DataFrame (TODO #2)\n{indent}df_state_manager.update_current_dataframe(df)\n{indent}"

            return func_body + update_code + return_stmt

        content = re.sub(pattern, add_update, content, flags=re.DOTALL)

    return content


def add_inmemory_to_docstrings(content):
    """Aggiunge **IN-MEMORY OPTIMIZATION** alle docstrings."""

    # Pattern per trovare docstrings senza già il tag
    pattern = r'(@tool\s*\ndef\s+\w+\([^)]+\)\s*->\s*str:\s*\n\s+"""[^\n]+\n)(?!\s+\*\*IN-MEMORY)'

    def add_optimization_note(match):
        docstring_start = match.group(1)
        return docstring_start + '    **IN-MEMORY OPTIMIZATION**: Uses cached DataFrame if available.\n\n'

    return re.sub(pattern, add_optimization_note, content)


def main():
    """Esegue conversione completa."""

    file_path = '/u/gpinna/phd_projects/plusAgent/plus_agent/smolagents_tools.py'

    print("=" * 70)
    print("BATCH CONVERSION SCRIPT")
    print("=" * 70)

    # 1. Backup
    print("\n📦 Creating backup...")
    backup_path = backup_file(file_path)
    print(f"   ✓ Backup created: {backup_path}")

    # 2. Leggi file
    print("\n📖 Reading file...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_lines = content.count('\n')
    print(f"   ✓ File loaded: {original_lines} lines")

    # 3. Applica conversioni
    print("\n🔄 Applying conversions...")

    print("   - Converting pd.read_csv() to state manager...")
    content = convert_pd_read_csv(content)

    print("   - Removing df.to_csv() calls...")
    content = remove_csv_saves(content)

    print("   - Updating return messages...")
    content = update_return_messages(content)

    print("   - Adding state updates for manipulation tools...")
    content = add_state_updates_for_manipulation(content)

    print("   - Adding IN-MEMORY OPTIMIZATION to docstrings...")
    content = add_inmemory_to_docstrings(content)

    # 4. Salva
    print("\n💾 Saving converted file...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    final_lines = content.count('\n')
    print(f"   ✓ File saved: {final_lines} lines")

    # 5. Summary
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETED!")
    print("=" * 70)
    print(f"Original lines: {original_lines}")
    print(f"Final lines: {final_lines}")
    print(f"Lines changed: {abs(final_lines - original_lines)}")
    print(f"\nBackup: {backup_path}")
    print(f"Modified: {file_path}")
    print("\n✅ All tools converted to in-memory DataFrame!")


if __name__ == '__main__':
    main()
