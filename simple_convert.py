#!/usr/bin/env python3
"""
Script semplice per mostrare i pattern di conversione da applicare manualmente.
"""

# Template per conversione tool manipulation/operations
TEMPLATE_MANIPULATION = '''
# PRIMA:
df = pd.read_csv(file_path)

# DOPO:
df = df_state_manager.get_current_dataframe()
if df is None:
    df = df_state_manager.load_dataframe(file_path)

# PRIMA (alla fine):
output_path = file_path.replace('.csv', '_suffix.csv')
df.to_csv(output_path, index=False)
return f"Success. Saved to: {output_path}"

# DOPO (alla fine):
df_state_manager.update_current_dataframe(df)
return f"✓ Success. DataFrame updated in memory."
'''

# Template per ML tools
TEMPLATE_ML = '''
# PRIMA:
df = pd.read_csv(file_path)

# DOPO:
df = df_state_manager.get_current_dataframe()
if df is None:
    df = df_state_manager.load_dataframe(file_path)

# NOTA: ML tools NON aggiornano il DataFrame!
# Non aggiungere: df_state_manager.update_current_dataframe(df)
# I modelli vengono salvati su disco, il DataFrame non cambia
'''

# Lista tool da convertire
TOOLS_TO_CONVERT = {
    "Data (read-only - no update)": [
        "read_json_file", "load_dataset", "get_column_names",
        "get_data_types", "get_null_counts", "get_unique_values",
        "get_numeric_summary", "get_first_rows"
    ],
    "Manipulation (with update)": [
        "drop_column", "drop_null_rows", "fill_numeric_nulls",
        "fill_categorical_nulls", "encode_categorical",
        "create_new_feature", "normalize_column"
    ],
    "Operations (with update)": [
        "perform_math_operations", "aggregate_data", "string_operations",
        "filter_rows_numeric", "filter_rows_categorical", "select_columns"
    ],
    "Statistical (read-only - no update)": [
        "calculate_correlation", "perform_ttest", "chi_square_test",
        "calculate_group_statistics"
    ],
    "Visualization (read-only - no update)": [
        "create_histogram", "create_scatter_plot",
        "create_bar_chart", "create_correlation_heatmap"
    ],
    "ML (read-only - no update)": [
        "train_regression_model", "train_random_forest_model",
        "train_knn_model", "train_svm_model", "evaluate_model",
        "train_logistic_regression", "train_decision_tree",
        "train_gradient_boosting", "make_prediction",
        "perform_cross_validation", "feature_selection"
    ],
    "Special Titanic (read-only)": [
        "answer_survival_question", "predict_single_passenger_survival",
        "get_dataset_insights"
    ]
}

def print_status():
    """Stampa stato conversione."""
    print("=" * 70)
    print("TOOL DA CONVERTIRE")
    print("=" * 70)

    total = 0
    for category, tools in TOOLS_TO_CONVERT.items():
        print(f"\n{category}: {len(tools)} tools")
        for tool in tools:
            print(f"  ⏳ {tool}")
        total += len(tools)

    print(f"\n{'=' * 70}")
    print(f"TOTALE: {total} tools da convertire")
    print(f"{'=' * 70}")

    print("\n\nREGOLE DI CONVERSIONE:")
    print("-" * 70)
    print("1. Tutti i tool: Sostituire pd.read_csv() con state manager")
    print("2. Manipulation/Operations: Aggiungere update_current_dataframe()")
    print("3. ML/Visualization/Statistical: NO update (solo lettura)")
    print("4. Rimuovere SEMPRE df.to_csv() e output_path")
    print("5. Aggiungere docstring: **IN-MEMORY OPTIMIZATION**")
    print("6. Messaggi: 'Saved to' → 'DataFrame updated in memory'")
    print("7. Aggiungere: '✓' ai messaggi di successo")


if __name__ == '__main__':
    print_status()
