"""
Save Dataset Tool - Explicit DataFrame saving
"""

from smolagents import tool
from smolagents_tools import df_state_manager


@tool
def save_dataset(output_path: str, include_index: bool = False) -> str:
    """
    Explicitly saves the current in-memory DataFrame to a CSV file.

    **USE CASE**: Call this tool when you need to persist analysis results to disk.
    During normal operations, the DataFrame stays in memory for performance.
    Only save when the user explicitly requests to export results.

    Args:
        output_path: Path where to save the CSV file (e.g., "results/output.csv")
        include_index: Whether to include DataFrame index in the CSV (default: False)

    Returns:
        Success message with file path and DataFrame info

    Example:
        >>> # After completing analysis, save results
        >>> save_dataset("results/titanic_processed.csv")
        ✓ Dataset saved to results/titanic_processed.csv (891 rows, 15 columns)
    """
    try:
        df = df_state_manager.get_current_dataframe()

        if df is None:
            return "❌ Error: No DataFrame currently loaded in memory. Load a dataset first."

        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Save
        df.to_csv(output_path, index=include_index)

        return f"✅ Dataset saved to {output_path} ({len(df)} rows, {len(df.columns)} columns)"

    except Exception as e:
        return f"❌ Error saving dataset: {str(e)}"


@tool
def clear_dataframe_cache() -> str:
    """
    Clears the in-memory DataFrame cache to start a new analysis session.

    **USE CASE**: Call this when switching to a completely different dataset
    or starting a new analysis from scratch.

    Returns:
        Confirmation message
    """
    try:
        df_state_manager.clear_all()
        return "✅ DataFrame cache cleared. Ready for new analysis."
    except Exception as e:
        return f"❌ Error clearing cache: {str(e)}"


@tool
def get_dataframe_info() -> str:
    """
    Returns information about the current in-memory DataFrame.

    Returns:
        DataFrame metadata (shape, columns, memory usage) or message if no DataFrame loaded
    """
    try:
        if not df_state_manager.has_dataframe():
            return "ℹ️ No DataFrame currently loaded in memory."

        metadata = df_state_manager.get_metadata()
        if not metadata:
            return "ℹ️ No metadata available."

        df = df_state_manager.get_current_dataframe()

        info = [
            f"📊 Current DataFrame Info:",
            f"",
            f"Original file: {metadata.get('original_file', 'unknown')}",
            f"Loaded at: {metadata.get('loaded_at', 'unknown')}",
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}",
            f"Memory: {metadata.get('memory_usage', 0) / 1024 / 1024:.2f} MB",
            f"",
            f"✓ DataFrame is cached in memory for fast access"
        ]

        return "\n".join(info)

    except Exception as e:
        return f"❌ Error getting DataFrame info: {str(e)}"
