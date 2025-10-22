#!/usr/bin/env python3
"""
New Tool: predict_single_passenger_survival

This tool can be integrated into smolagents_singleagent.py.
It provides personalized Titanic survival predictions based on passenger characteristics.

Test with: uv run python predict_survival_tool.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from smolagents import tool


@tool
def predict_single_passenger_survival(
    file_path: str,
    passenger_age: float,
    passenger_sex: str,
    passenger_class: int,
    siblings_spouses: int = 0,
    parents_children: int = 0,
    fare: float = None
) -> str:
    """
    Predicts survival probability for a single Titanic passenger based on their characteristics.

    This specialized tool trains a machine learning model on the Titanic dataset and then
    predicts the survival probability for a hypothetical passenger with specified characteristics.
    It provides a detailed breakdown of the prediction with confidence levels, feature importance,
    and contextual comparison to historical survival rates.

    **Capabilities:**
    - Trains Random Forest classifier on complete Titanic dataset
    - Handles missing values automatically
    - Encodes categorical variables (sex, embarked)
    - Provides probability estimates (not just binary prediction)
    - Explains prediction with feature importance
    - Compares to historical survival rates
    - Gives confidence intervals

    **Use Cases:**
    - Answer "What if I was on the Titanic?" questions
    - Understand how different factors affected survival
    - Educational demonstrations of ML prediction
    - Exploring counterfactual scenarios
    - Testing model sensitivity to different features

    Args:
        file_path (str): Path to the Titanic dataset CSV file.
        passenger_age (float): Age of the passenger in years (0.1 to 120).
        passenger_sex (str): Gender ('male' or 'female', case-insensitive).
        passenger_class (int): Ticket class (1=First, 2=Second, 3=Third).
        siblings_spouses (int): Number of siblings/spouses aboard (default: 0).
        parents_children (int): Number of parents/children aboard (default: 0).
        fare (float): Ticket fare in pounds (optional, uses class median if None).

    Returns:
        str: Comprehensive prediction report with probability, context, and interpretation.
    """
    try:
        # Input validation
        passenger_sex = passenger_sex.lower().strip()
        if passenger_sex not in ['male', 'female']:
            return (f"✗ Error: Invalid sex value '{passenger_sex}'\n\n"
                   "Must be 'male' or 'female' (case-insensitive)")

        if passenger_class not in [1, 2, 3]:
            return (f"✗ Error: Invalid passenger class {passenger_class}\n\n"
                   "Must be 1, 2, or 3:\n"
                   "  1 = First class (upper/wealthy)\n"
                   "  2 = Second class (middle)\n"
                   "  3 = Third class (lower/working)")

        if not (0.1 <= passenger_age <= 120):
            return (f"✗ Error: Invalid age {passenger_age}\n\n"
                   "Age must be between 0.1 and 120 years")

        if siblings_spouses < 0 or parents_children < 0:
            return "✗ Error: Family counts cannot be negative"

        # Load dataset
        df = pd.read_csv(file_path)

        # Handle different column naming conventions (lowercase/uppercase)
        col_map = {col.lower(): col for col in df.columns}

        # Map required columns
        required = ['survived', 'pclass', 'sex', 'age']
        if not all(req in col_map for req in required):
            missing = [r for r in required if r not in col_map]
            return f"✗ Error: Dataset missing required columns: {missing}"

        # Prepare features
        feature_names_orig = [
            col_map['pclass'],
            col_map['sex'],
            col_map['age']
        ]

        # Add optional features if available
        if 'sibsp' in col_map:
            feature_names_orig.append(col_map['sibsp'])
        if 'parch' in col_map:
            feature_names_orig.append(col_map['parch'])
        if 'fare' in col_map:
            feature_names_orig.append(col_map['fare'])

        # Create training dataset
        df_train = df[[col_map['survived']] + feature_names_orig].copy()
        df_train.columns = ['survived', 'pclass', 'sex', 'age'] + \
                          (['sibsp'] if 'sibsp' in col_map else []) + \
                          (['parch'] if 'parch' in col_map else []) + \
                          (['fare'] if 'fare' in col_map else [])

        # Handle missing values
        for col in df_train.columns:
            if col in ['age', 'fare']:
                df_train[col] = df_train[col].fillna(df_train[col].median())
            elif col == 'sex':
                df_train[col] = df_train[col].fillna('male')

        # Encode sex
        df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})

        # Separate features and target
        feature_cols = [c for c in df_train.columns if c != 'survived']
        X = df_train[feature_cols]
        y = df_train['survived']

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)

        # Calculate model metrics
        train_acc = model.score(X, y)
        cv_scores = cross_val_score(model, X, y, cv=5)

        # Prepare passenger data
        if fare is None and 'fare' in feature_cols:
            class_fares = df_train[df_train['pclass'] == passenger_class]['fare']
            fare = class_fares.median() if len(class_fares) > 0 else 0
            fare_estimated = True
        else:
            fare_estimated = False

        passenger_data = {
            'pclass': passenger_class,
            'sex': 1 if passenger_sex == 'male' else 0,
            'age': passenger_age
        }

        if 'sibsp' in feature_cols:
            passenger_data['sibsp'] = siblings_spouses
        if 'parch' in feature_cols:
            passenger_data['parch'] = parents_children
        if 'fare' in feature_cols:
            passenger_data['fare'] = fare if fare is not None else 0

        passenger_df = pd.DataFrame([passenger_data])[feature_cols]

        # Make prediction
        survival_prob = model.predict_proba(passenger_df)[0][1]
        prediction = model.predict(passenger_df)[0]

        # Get feature importance
        importances = list(zip(feature_cols, model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)

        # Calculate historical rates
        overall_survival = y.mean()
        sex_survival = df_train[df_train['sex'] == passenger_data['sex']]['survived'].mean()
        class_sex_survival = df_train[
            (df_train['pclass'] == passenger_class) &
            (df_train['sex'] == passenger_data['sex'])
        ]['survived'].mean()

        # Find similar passengers
        similar = df_train[
            (df_train['pclass'] == passenger_class) &
            (df_train['sex'] == passenger_data['sex']) &
            (df_train['age'].between(passenger_age - 5, passenger_age + 5))
        ]

        # Build comprehensive result
        class_names = {
            1: "1st class (First/Upper class)",
            2: "2nd class (Second/Middle class)",
            3: "3rd class (Third/Working class)"
        }

        result = [
            "",
            "🚢 Survival Prediction for Titanic Passenger",
            "=" * 60,
            "",
            "👤 Passenger Profile:",
            f"   • Age: {passenger_age} years old",
            f"   • Sex: {passenger_sex.capitalize()}",
            f"   • Class: {class_names[passenger_class]}",
            f"   • Family: {siblings_spouses} siblings/spouses, {parents_children} parents/children",
        ]

        if 'fare' in feature_cols:
            fare_str = f"£{fare:.2f}"
            if fare_estimated:
                fare_str += " (estimated from class median)"
            result.append(f"   • Fare: {fare_str}")

        result.extend([
            "",
            "🎯 Prediction Results:",
            f"   • Survival Probability: {survival_prob*100:.1f}%",
            f"   • Prediction: {'✓ SURVIVED' if prediction == 1 else '✗ DID NOT SURVIVE'}",
        ])

        # Confidence assessment
        if survival_prob > 0.75:
            conf_desc = "Very High"
            conf_emoji = "🟢"
        elif survival_prob > 0.60:
            conf_desc = "High"
            conf_emoji = "🟢"
        elif survival_prob > 0.55:
            conf_desc = "Moderate-High"
            conf_emoji = "🟡"
        elif survival_prob > 0.45:
            conf_desc = "Uncertain"
            conf_emoji = "🟡"
        elif survival_prob > 0.30:
            conf_desc = "Moderate-Low"
            conf_emoji = "🟠"
        else:
            conf_desc = "Low"
            conf_emoji = "🔴"

        result.append(f"   • Confidence: {conf_emoji} {conf_desc} ({survival_prob*100:.0f}%)")
        result.append("")

        # Feature importance
        result.append("📊 Feature Importance (What Mattered Most):")

        feature_names_pretty = {
            'sex': 'Sex (Gender)',
            'pclass': 'Passenger Class',
            'age': 'Age',
            'fare': 'Ticket Fare',
            'sibsp': 'Siblings/Spouses Aboard',
            'parch': 'Parents/Children Aboard'
        }

        for idx, (feat, imp) in enumerate(importances[:5], 1):
            pretty_name = feature_names_pretty.get(feat, feat)
            result.append(f"   {idx}. {pretty_name}: {imp*100:.1f}%")

        result.append("")

        # Historical context
        result.extend([
            "📜 Historical Context:",
            f"   • Overall survival rate: {overall_survival*100:.1f}%",
            f"   • {passenger_sex.capitalize()} survival rate: {sex_survival*100:.1f}%",
            f"   • {class_names[passenger_class].split('(')[0].strip()} {passenger_sex} survival: {class_sex_survival*100:.1f}%",
        ])

        if class_sex_survival > 0:
            ratio = survival_prob / class_sex_survival
            if ratio > 1.2:
                result.append(f"   → Your profile: {survival_prob*100:.1f}% ({ratio:.1f}x BETTER than average)")
            elif ratio < 0.8:
                result.append(f"   → Your profile: {survival_prob*100:.1f}% ({1/ratio:.1f}x WORSE than average)")
            else:
                result.append(f"   → Your profile: {survival_prob*100:.1f}% (similar to average)")

        result.append("")

        # Similar passengers analysis
        if len(similar) > 0:
            similar_survived = similar['survived'].sum()
            result.extend([
                f"👥 Similar Passengers (Age {max(0, passenger_age-5):.0f}-{passenger_age+5:.0f}, {passenger_sex.capitalize()}, {class_names[passenger_class].split('(')[0].strip()}):",
                f"   • Total found: {len(similar)} passengers",
                f"   • Survived: {int(similar_survived)} ({similar_survived/len(similar)*100:.1f}%)",
                f"   • Did not survive: {len(similar) - int(similar_survived)} ({(1-similar_survived/len(similar))*100:.1f}%)",
                ""
            ])

        # Interpretation
        result.append("💡 Interpretation:")

        if passenger_sex == 'female':
            if survival_prob > 0.7:
                result.append("   ✓ Being female significantly improved survival odds")
            result.append("   ✓ Women and children were prioritized in evacuations")
        else:
            if survival_prob > 0.5:
                result.append("   ⚠ Despite being male, other factors improved odds")
            result.append("   ⚠ Male passengers had much lower survival rates overall")

        if passenger_class == 1:
            result.append("   ✓ First class passengers had best access to lifeboats")
        elif passenger_class == 3:
            result.append("   ⚠ Third class passengers faced barriers to lifeboats")

        if passenger_age < 18:
            result.append("   ✓ Children were prioritized in evacuations")
        elif passenger_age > 60:
            result.append("   ⚠ Elderly passengers had lower survival rates")

        if siblings_spouses + parents_children == 0:
            result.append("   • Traveling alone had mixed effects on survival")
        elif siblings_spouses + parents_children > 3:
            result.append("   ⚠ Large families had difficulty staying together")

        result.append("")

        # Model performance
        result.extend([
            "🔬 Model Performance:",
            f"   • Training accuracy: {train_acc*100:.1f}%",
            f"   • Cross-validation score: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%",
            f"   • Model reliability: {'High' if cv_scores.mean() > 0.80 else 'Moderate' if cv_scores.mean() > 0.75 else 'Fair'}",
            "",
            "=" * 60
        ])

        return "\n".join(result)

    except FileNotFoundError:
        return f"✗ Error: File not found at '{file_path}'"
    except KeyError as e:
        return f"✗ Error: Missing required column in dataset: {str(e)}"
    except Exception as e:
        return f"✗ Error predicting survival: {type(e).__name__}: {str(e)}"


# Test function
def test_tool():
    """Test the prediction tool with sample passengers."""
    print("\n" + "="*80)
    print(" TESTING: predict_single_passenger_survival Tool")
    print("="*80)

    test_cases = [
        {
            'name': '20-year-old male, 1st class (User\'s question)',
            'params': {
                'file_path': 'data/titanic.csv',
                'passenger_age': 20,
                'passenger_sex': 'male',
                'passenger_class': 1
            }
        },
        {
            'name': '30-year-old female, 3rd class',
            'params': {
                'file_path': 'data/titanic.csv',
                'passenger_age': 30,
                'passenger_sex': 'female',
                'passenger_class': 3,
                'siblings_spouses': 1,
                'parents_children': 2
            }
        },
        {
            'name': '5-year-old child, 2nd class',
            'params': {
                'file_path': 'data/titanic.csv',
                'passenger_age': 5,
                'passenger_sex': 'male',
                'passenger_class': 2,
                'parents_children': 2
            }
        }
    ]

    for idx, test in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"Test Case {idx}: {test['name']}")
        print('─'*80)

        result = predict_single_passenger_survival(**test['params'])
        print(result)

    print("\n" + "="*80)
    print(" All test cases completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run tests
    test_tool()
