#!/usr/bin/env python3
"""
Script di conversione batch per convertire tool a in-memory DataFrame.
Applica il pattern di conversione automaticamente a tutti i tool rimanenti.
"""

import re
import sys


def convert_tool_to_inmemory(tool_code: str, tool_name: str, tool_category: str) -> str:
    """
    Converte un tool da file-based a in-memory.

    Args:
        tool_code: Codice completo del tool
        tool_name: Nome della funzione
        tool_category: Categoria (manipulation, operations, ml, visualization)

    Returns:
        Codice convertito
    """

    # Pattern 1: Aggiungi IN-MEMORY OPTIMIZATION alla docstring
    if "**IN-MEMORY OPTIMIZATION**" not in tool_code:
        # Trova la prima linea della docstring dopo """
        pattern = r'(def\s+\w+\([^)]+\)\s*->\s*str:\s*\n\s+"""[^\n]+)'

        if tool_category == "visualization":
            replacement = r'\1\n\n    **IN-MEMORY OPTIMIZATION**: Reads from cached DataFrame but does NOT modify it.'
        elif tool_category == "ml":
            replacement = r'\1\n\n    **IN-MEMORY OPTIMIZATION**: Reads from cached DataFrame for training. Does NOT modify DataFrame.'
        else:
            replacement = r'\1\n\n    **IN-MEMORY OPTIMIZATION**: Uses cached DataFrame if available and updates in memory without disk I/O.'

        tool_code = re.sub(pattern, replacement, tool_code)

    # Pattern 2: Sostituisci pd.read_csv(file_path) con state manager
    if "pd.read_csv(file_path)" in tool_code:
        old_pattern = r'(\s+)df = pd\.read_csv\(file_path\)'
        new_code = r'\1# Use in-memory DataFrame if available (TODO #2)\n\1df = df_state_manager.get_current_dataframe()\n\1if df is None:\n\1    df = df_state_manager.load_dataframe(file_path)'
        tool_code = re.sub(old_pattern, new_code, tool_code)

    # Pattern 3: Rimuovi df.to_csv() e output_path
    # Rimuove le linee di save
    tool_code = re.sub(
        r'\s+output_path = file_path\.replace\([^\)]+\)\s*\n\s+df\.to_csv\(output_path[^\)]*\)\s*\n',
        '\n',
        tool_code
    )

    # Pattern 4: Aggiungi state manager update prima del return (solo per manipulation/operations)
    if tool_category in ["manipulation", "operations"]:
        # Trova return di successo e aggiungi update prima
        lines = tool_code.split('\n')
        new_lines = []
        for i, line in enumerate(lines):
            # Se è un return di successo (non errore)
            if 'return f"' in line and 'Error' not in line and i > 10:
                # Check se già c'è update nelle righe precedenti
                has_update = False
                for j in range(max(0, i-3), i):
                    if 'df_state_manager.update_current_dataframe' in lines[j]:
                        has_update = True
                        break

                if not has_update and 'try:' not in line:
                    indent = len(line) - len(line.lstrip())
                    update_line = ' ' * indent + '# Update the in-memory DataFrame (TODO #2)\n' + ' ' * indent + 'df_state_manager.update_current_dataframe(df)\n'
                    new_lines.append(update_line)

            new_lines.append(line)

        tool_code = '\n'.join(new_lines)

    # Pattern 5: Migliora i messaggi di return
    tool_code = re.sub(
        r'Saved to: \{output_path\}',
        'DataFrame updated in memory',
        tool_code
    )

    tool_code = re.sub(
        r'Output saved to: \{output_path\}',
        'DataFrame updated in memory',
        tool_code
    )

    # Pattern 6: Aggiungi emoji check mark ai messaggi di successo
    tool_code = re.sub(
        r'return f"((?!Error|✓)[^"]*)"',
        r'return f"✓ \1"',
        tool_code
    )

    # Pattern 7: Converti error generici in eccezioni custom per column not found
    if 'if column_name not in df.columns:' in tool_code and 'raise ColumnNotFoundError' not in tool_code:
        old_pattern = r'(\s+)if column_name not in df\.columns:\s*\n\s+return f"Column.*?not found.*?"'
        new_code = r'''\1if column_name not in df.columns:
\1    raise ColumnNotFoundError(
\1        problem=f"Column '{column_name}' does not exist in the DataFrame",
\1        context={"tool_name": "''' + tool_name + r'''", "parameters": {"column_name": column_name}},
\1        cause=f"Available columns are: {list(df.columns)}",
\1        solution=f"Use one of the existing column names",
\1        example=f"''' + tool_name + r'''(file_path, '{df.columns[0] if len(df.columns) > 0 else 'column_name'}')"
\1    )'''

        tool_code = re.sub(old_pattern, new_code, tool_code, flags=re.DOTALL)

    # Pattern 8: Aggiungi try-except per ToolException
    if 'except ToolException:' not in tool_code and 'except Exception as e:' in tool_code:
        tool_code = re.sub(
            r'(\s+)(except Exception as e:)',
            r'\1except ToolException:\n\1    raise  # Re-raise custom exceptions\n\1\2',
            tool_code
        )

    return tool_code


def main():
    """Conversione batch di tutti i tool."""

    file_path = '/u/gpinna/phd_projects/plusAgent/plus_agent/smolagents_tools.py'

    print("📖 Lettura smolagents_tools.py...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Tool da convertire per categoria
    manipulation_tools = [
        'drop_column', 'drop_null_rows', 'fill_numeric_nulls',
        'fill_categorical_nulls', 'encode_categorical', 'create_new_feature',
        'normalize_column'
    ]

    operations_tools = [
        'perform_math_operations', 'aggregate_data', 'string_operations',
        'filter_rows_numeric', 'filter_rows_categorical', 'select_columns',
        'calculate_correlation', 'perform_ttest', 'chi_square_test',
        'calculate_group_statistics'
    ]

    visualization_tools = [
        'create_histogram', 'create_scatter_plot', 'create_bar_chart',
        'create_correlation_heatmap'
    ]

    ml_tools = [
        'train_regression_model', 'train_random_forest_model', 'train_knn_model',
        'train_svm_model', 'evaluate_model', 'train_logistic_regression',
        'train_decision_tree', 'train_gradient_boosting', 'make_prediction',
        'perform_cross_validation', 'feature_selection'
    ]

    data_tools = [
        'load_dataset', 'get_column_names', 'get_data_types', 'get_null_counts',
        'get_unique_values', 'get_numeric_summary', 'get_first_rows', 'read_json_file'
    ]

    all_tools = {
        'manipulation': manipulation_tools,
        'operations': operations_tools,
        'visualization': visualization_tools,
        'ml': ml_tools,
        'data': data_tools
    }

    # Estrai tutti i tool
    tool_pattern = r'(@tool\s*\ndef\s+(\w+)\([^)]+\)[^:]+:.*?(?=@tool|$))'
    matches = list(re.finditer(tool_pattern, content, re.DOTALL))

    converted_count = 0
    for match in matches:
        tool_code = match.group(1)
        tool_name = match.group(2)

        # Determina categoria
        category = None
        for cat, tools in all_tools.items():
            if tool_name in tools:
                category = cat
                break

        if category is None:
            continue  # Skip tool già convertiti o non nella lista

        # Converti
        if "IN-MEMORY OPTIMIZATION" not in tool_code:  # Non ancora convertito
            print(f"  🔄 Converting {tool_name} ({category})...")
            converted_tool = convert_tool_to_inmemory(tool_code, tool_name, category)
            content = content.replace(tool_code, converted_tool)
            converted_count += 1

    # Salva file modificato
    print(f"\n💾 Salvando modifiche...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✅ Conversione completata!")
    print(f"   Tool convertiti: {converted_count}")
    print(f"   File aggiornato: {file_path}")


if __name__ == '__main__':
    main()
