#!/usr/bin/env python3
"""
Fix script per correggere update_current_dataframe() messi nel posto sbagliato.
"""

import re
from datetime import datetime


def fix_misplaced_updates(content):
    """
    Rimuove update messi in blocchi if di errore e li sposta prima dei return di successo.
    """

    # Pattern 1: Rimuovi update seguiti immediatamente da return di errore
    # Questo pattern cerca update che sono seguiti da return con messaggi di errore
    pattern1 = r'\s+# Update the in-memory DataFrame \(TODO #2\)\s*\n\s+df_state_manager\.update_current_dataframe\(df\)\s*\n(\s+return f"(?:Column|Error|No|Unknown)[^"]*")'
    content = re.sub(pattern1, r'\1', content)

    # Pattern 2: Aggiunge update prima di return di successo per tool manipulation/operations
    # Lista di tool che devono avere update
    tools_needing_update = [
        'drop_column', 'drop_null_rows', 'fill_numeric_nulls',
        'fill_categorical_nulls', 'encode_categorical',
        'create_new_feature', 'normalize_column',
        'perform_math_operations', 'aggregate_data', 'string_operations',
        'filter_rows_numeric', 'filter_rows_categorical', 'select_columns'
    ]

    for tool_name in tools_needing_update:
        # Trova il tool
        tool_pattern = rf'(@tool\s*\ndef {tool_name}\([^)]+\)[^:]+:.*?)(except.*?:.*?return.*?\n\n)'

        def add_update_before_success_return(match):
            tool_code = match.group(0)

            # Se già ha un update in posizione corretta, skip
            if re.search(r'df_state_manager\.update_current_dataframe\(df\)\s*\n\s+return f"[^E]', tool_code):
                return tool_code

            # Trova l'ultimo return di successo nel try block (non nel except)
            # Cerchiamo return che non contengono "Error", "not found", "Unknown"
            lines = tool_code.split('\n')
            new_lines = []
            in_try = False
            in_except = False

            for i, line in enumerate(lines):
                if 'try:' in line:
                    in_try = True
                elif 'except' in line:
                    in_try = False
                    in_except = True

                # Se è un return di successo nel try block
                if in_try and 'return f"' in line:
                    # Controlla se è un messaggio di successo
                    if not any(word in line for word in ['Error', 'not found', 'Unknown', 'No missing']):
                        # Aggiungi update prima di questo return
                        indent = len(line) - len(line.lstrip())
                        # Ma solo se non c'è già nelle ultime 3 righe
                        has_update_nearby = False
                        for j in range(max(0, i-3), i):
                            if 'update_current_dataframe' in new_lines[j]:
                                has_update_nearby = True
                                break

                        if not has_update_nearby:
                            update_line = ' ' * indent + '# Update the in-memory DataFrame (TODO #2)\n' + ' ' * indent + 'df_state_manager.update_current_dataframe(df)\n'
                            new_lines.append(update_line)

                new_lines.append(line)

            return '\n'.join(new_lines)

        content = re.sub(tool_pattern, add_update_before_success_return, content, flags=re.DOTALL)

    return content


def main():
    """Esegue fix."""

    file_path = '/u/gpinna/phd_projects/plusAgent/plus_agent/smolagents_tools.py'

    print("=" * 70)
    print("FIX UPDATE PLACEMENT SCRIPT")
    print("=" * 70)

    # Backup
    print("\n📦 Creating backup...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.backup_fix_{timestamp}"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"   ✓ Backup: {backup_path}")

    # Fix
    print("\n🔧 Fixing misplaced updates...")
    content = fix_misplaced_updates(content)

    # Save
    print("\n💾 Saving fixed file...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n✅ Fix completed!")


if __name__ == '__main__':
    main()
