"""Titanic-specific analysis tools."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from langchain.tools import tool


@tool
def calculate_survival_rate_by_group(file_path: str, group_column: str) -> str:
    """
    Calculate survival rates by a specific grouping column.

    Args:
        file_path: Path to the data file
        group_column: Column to group by (e.g., 'pclass', 'sex', 'embarked')

    Returns:
        String containing survival rates for each group
    """
    try:
        df = pd.read_csv(file_path)

        if group_column not in df.columns:
            return f"Column '{group_column}' not found in dataset"

        if 'survived' not in df.columns:
            return "Column 'survived' not found in dataset"

        # Calculate survival rates
        survival_stats = df.groupby(group_column).agg({
            'survived': ['count', 'sum', 'mean']
        }).round(4)

        # Flatten column names
        survival_stats.columns = ['total_passengers', 'survivors', 'survival_rate']
        survival_stats = survival_stats.reset_index()

        # Convert to readable format
        results = {}
        for _, row in survival_stats.iterrows():
            group_value = row[group_column]
            results[f"{group_column}_{group_value}"] = {
                "total_passengers": int(row['total_passengers']),
                "survivors": int(row['survivors']),
                "survival_rate": f"{row['survival_rate']:.1%}"
            }

        return f"Survival rates by {group_column}: {results}"

    except Exception as e:
        return f"Error calculating survival rates: {str(e)}"


@tool
def get_statistics_for_profile(file_path: str, filters: str, target_column: str) -> str:
    """
    Get statistics for a specific passenger profile.

    Args:
        file_path: Path to the data file
        filters: Comma-separated filters in format "column=value,column=value"
        target_column: Column to get statistics for

    Returns:
        String containing statistics for the filtered profile
    """
    try:
        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            return f"Column '{target_column}' not found in dataset"

        # Apply filters
        filtered_df = df.copy()
        filter_conditions = []

        for filter_str in filters.split(','):
            if '=' in filter_str:
                col, val = filter_str.strip().split('=')
                col, val = col.strip(), val.strip()

                if col not in df.columns:
                    return f"Filter column '{col}' not found in dataset"

                # Convert value to appropriate type
                if df[col].dtype in ['int64', 'float64']:
                    try:
                        val = float(val)
                    except:
                        pass
                elif val.lower() in ['true', 'false']:
                    val = val.lower() == 'true'

                filtered_df = filtered_df[filtered_df[col] == val]
                filter_conditions.append(f"{col}={val}")

        if len(filtered_df) == 0:
            return f"No passengers found matching filters: {filter_conditions}"

        # Calculate statistics
        if df[target_column].dtype in ['int64', 'float64']:
            stats = {
                "count": len(filtered_df),
                "mean": filtered_df[target_column].mean(),
                "median": filtered_df[target_column].median(),
                "std": filtered_df[target_column].std(),
                "min": filtered_df[target_column].min(),
                "max": filtered_df[target_column].max()
            }
        else:
            # For categorical columns
            value_counts = filtered_df[target_column].value_counts()
            stats = {
                "count": len(filtered_df),
                "value_counts": value_counts.to_dict(),
                "most_common": value_counts.index[0] if len(value_counts) > 0 else None
            }

        return f"Statistics for profile {filter_conditions} on {target_column}: {stats}"

    except Exception as e:
        return f"Error getting profile statistics: {str(e)}"


@tool
def calculate_survival_probability_by_features(file_path: str, sex: str, pclass: int, age_range: str = "all") -> str:
    """
    Calculate survival probability for specific passenger characteristics.

    Args:
        file_path: Path to the data file
        sex: Gender (male/female)
        pclass: Passenger class (1, 2, or 3)
        age_range: Age range filter (e.g., "18-30", "30-50", "all")

    Returns:
        String containing survival probability and supporting data
    """
    try:
        df = pd.read_csv(file_path)

        required_cols = ['survived', 'sex', 'pclass', 'age']
        for col in required_cols:
            if col not in df.columns:
                return f"Required column '{col}' not found in dataset"

        # Apply filters
        filtered_df = df.copy()

        # Filter by sex and class
        filtered_df = filtered_df[
            (filtered_df['sex'] == sex.lower()) &
            (filtered_df['pclass'] == pclass)
        ]

        # Apply age filter if specified
        if age_range != "all":
            try:
                if '-' in age_range:
                    min_age, max_age = map(int, age_range.split('-'))
                    filtered_df = filtered_df[
                        (filtered_df['age'] >= min_age) &
                        (filtered_df['age'] <= max_age)
                    ]
            except:
                return f"Invalid age_range format: {age_range}. Use format like '18-30' or 'all'"

        if len(filtered_df) == 0:
            return f"No passengers found matching criteria: sex={sex}, class={pclass}, age_range={age_range}"

        # Calculate survival statistics
        total_passengers = len(filtered_df)
        survivors = filtered_df['survived'].sum()
        survival_rate = survivors / total_passengers if total_passengers > 0 else 0

        # Additional insights
        age_stats = filtered_df['age'].describe() if not filtered_df['age'].isna().all() else None
        fare_stats = filtered_df['fare'].describe() if 'fare' in filtered_df.columns else None

        results = {
            "profile": f"{sex.title()} in {pclass}{'st' if pclass == 1 else 'nd' if pclass == 2 else 'rd'} class",
            "age_range": age_range,
            "sample_size": total_passengers,
            "survivors": int(survivors),
            "survival_probability": f"{survival_rate:.1%}",
            "survival_rate_decimal": round(survival_rate, 3)
        }

        if age_stats is not None:
            results["age_stats"] = {
                "mean": round(age_stats['mean'], 1),
                "median": round(age_stats['50%'], 1)
            }

        if fare_stats is not None:
            results["fare_stats"] = {
                "mean": round(fare_stats['mean'], 2),
                "median": round(fare_stats['50%'], 2)
            }

        return f"Survival analysis results: {results}"

    except Exception as e:
        return f"Error calculating survival probability: {str(e)}"


@tool
def get_fare_estimate_by_profile(file_path: str, sex: str, pclass: int, age: float) -> str:
    """
    Estimate fare price based on passenger profile.

    Args:
        file_path: Path to the data file
        sex: Gender (male/female)
        pclass: Passenger class (1, 2, or 3)
        age: Age of the passenger

    Returns:
        String containing fare estimate and supporting statistics
    """
    try:
        df = pd.read_csv(file_path)

        required_cols = ['sex', 'pclass', 'age', 'fare']
        for col in required_cols:
            if col not in df.columns:
                return f"Required column '{col}' not found in dataset"

        # Filter by similar profiles
        similar_passengers = df[
            (df['sex'] == sex.lower()) &
            (df['pclass'] == pclass) &
            (df['age'].notna()) &
            (df['fare'].notna())
        ]

        if len(similar_passengers) == 0:
            return f"No passengers found with similar profile: sex={sex}, class={pclass}"

        # Find passengers with similar age (±5 years)
        age_similar = similar_passengers[
            (similar_passengers['age'] >= age - 5) &
            (similar_passengers['age'] <= age + 5)
        ]

        if len(age_similar) == 0:
            # If no age-similar passengers, use all passengers with same sex and class
            age_similar = similar_passengers

        # Calculate fare statistics
        fare_stats = age_similar['fare'].describe()

        results = {
            "profile": f"{age}-year-old {sex.lower()} in {pclass}{'st' if pclass == 1 else 'nd' if pclass == 2 else 'rd'} class",
            "sample_size": len(age_similar),
            "estimated_fare": round(fare_stats['mean'], 2),
            "fare_range": {
                "min": round(fare_stats['min'], 2),
                "max": round(fare_stats['max'], 2),
                "median": round(fare_stats['50%'], 2),
                "q25": round(fare_stats['25%'], 2),
                "q75": round(fare_stats['75%'], 2)
            },
            "most_likely_fare": round(fare_stats['50%'], 2)  # median is often more representative
        }

        return f"Fare estimate results: {results}"

    except Exception as e:
        return f"Error estimating fare: {str(e)}"


@tool
def count_passengers_by_criteria(file_path: str, criteria: str) -> str:
    """
    Count passengers matching specific criteria.

    Args:
        file_path: Path to the data file
        criteria: Criteria in format "column1=value1,column2=value2" or complex filters

    Returns:
        String containing count and breakdown
    """
    try:
        df = pd.read_csv(file_path)

        # Parse criteria
        filtered_df = df.copy()
        applied_filters = []

        if criteria.strip():
            for criterion in criteria.split(','):
                if '=' in criterion:
                    col, val = criterion.strip().split('=', 1)
                    col, val = col.strip(), val.strip()

                    if col not in df.columns:
                        return f"Column '{col}' not found in dataset"

                    # Handle different comparison operators
                    if val.startswith('<='):
                        val = float(val[2:])
                        filtered_df = filtered_df[filtered_df[col] <= val]
                        applied_filters.append(f"{col} <= {val}")
                    elif val.startswith('>='):
                        val = float(val[2:])
                        filtered_df = filtered_df[filtered_df[col] >= val]
                        applied_filters.append(f"{col} >= {val}")
                    elif val.startswith('<'):
                        val = float(val[1:])
                        filtered_df = filtered_df[filtered_df[col] < val]
                        applied_filters.append(f"{col} < {val}")
                    elif val.startswith('>'):
                        val = float(val[1:])
                        filtered_df = filtered_df[filtered_df[col] > val]
                        applied_filters.append(f"{col} > {val}")
                    else:
                        # Exact match
                        if df[col].dtype in ['int64', 'float64']:
                            try:
                                val = float(val)
                            except:
                                pass
                        elif val.lower() in ['true', 'false']:
                            val = val.lower() == 'true'

                        filtered_df = filtered_df[filtered_df[col] == val]
                        applied_filters.append(f"{col} = {val}")

        count = len(filtered_df)
        total = len(df)
        percentage = (count / total * 100) if total > 0 else 0

        results = {
            "total_passengers": total,
            "matching_criteria": count,
            "percentage": f"{percentage:.1f}%",
            "applied_filters": applied_filters
        }

        return f"Passenger count results: {results}"

    except Exception as e:
        return f"Error counting passengers: {str(e)}"