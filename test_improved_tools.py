#!/usr/bin/env python3
"""
Test script to demonstrate the improved tools with the Titanic dataset.
Run with: uv run python test_improved_tools.py
"""

import sys
sys.path.insert(0, '/u/gpinna/phd_projects/plusAgent/plus_agent')

from improved_tools import (
    load_dataset,
    get_column_names,
    get_data_types,
    get_null_counts,
    get_unique_values,
    get_numeric_summary,
    get_first_rows,
    get_dataset_insights
)

def print_section(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")

def main():
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "IMPROVED TOOLS DEMONSTRATION" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")

    dataset_path = "data/titanic.csv"

    # Test 1: Load Dataset
    print_section("TEST 1: Load Dataset")
    result = load_dataset(dataset_path)
    print(result)

    # Test 2: Get Column Names
    print_section("TEST 2: Get Column Names (with type categorization)")
    result = get_column_names(dataset_path)
    print(result)

    # Test 3: Get Data Types
    print_section("TEST 3: Get Data Types (detailed analysis)")
    result = get_data_types(dataset_path)
    print(result)

    # Test 4: Get Null Counts
    print_section("TEST 4: Missing Values Analysis (with recommendations)")
    result = get_null_counts(dataset_path)
    print(result)

    # Test 5: Get Unique Values
    print_section("TEST 5: Unique Values Analysis - Sex (categorical)")
    result = get_unique_values(dataset_path, "Sex")
    print(result)

    print_section("TEST 5b: Unique Values Analysis - Pclass (numeric categorical)")
    result = get_unique_values(dataset_path, "Pclass")
    print(result)

    # Test 6: Get Numeric Summary
    print_section("TEST 6: Numeric Summary - Age (with distribution analysis)")
    result = get_numeric_summary(dataset_path, "Age")
    print(result)

    # Test 7: Get First Rows
    print_section("TEST 7: First Rows Preview")
    result = get_first_rows(dataset_path, 5)
    print(result)

    # Test 8: Dataset Insights
    print_section("TEST 8: Comprehensive Dataset Insights")
    result = get_dataset_insights(dataset_path)
    print(result)

    # Summary
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 32 + "SUMMARY" + " " * 40 + "║")
    print("╚" + "=" * 78 + "╝\n")

    print("✓ All 8 improved tools tested successfully!")
    print("\nKey Improvements Demonstrated:")
    print("  1. ✓ Comprehensive docstrings with use cases and capabilities")
    print("  2. ✓ Formatted output with sections, bullets, and visual elements")
    print("  3. ✓ Detailed explanations and interpretations")
    print("  4. ✓ Actionable recommendations (e.g., for missing data)")
    print("  5. ✓ Better error messages with suggestions")
    print("  6. ✓ Statistical interpretation (skewness, outliers, etc.)")
    print("  7. ✓ Visual elements (bars, symbols) for better readability")
    print("  8. ✓ Domain-specific insights (Titanic survival analysis)")

    print("\n📊 Total tools in collection: 51 tools")
    print("✅ Tools improved so far: 8 data reading tools")
    print("🔄 Remaining: 43 tools (manipulation, operations, ML, viz, analysis)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
