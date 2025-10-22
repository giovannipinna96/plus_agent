"""
Tool ML Generalizzati - TODO #4
================================

Questi 3 tool sostituiscono i 28 tool ML specifici, offrendo:
- Interfaccia unificata
- Supporto multi-algoritmo
- Facile estensibilità
- Configurazione flessibile
"""

from smolagents import tool
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import joblib
import os

# Import modelli
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


# Mapping algoritmi disponibili
AVAILABLE_MODELS = {
    # Classification
    'random_forest': (RandomForestClassifier, 'classification'),
    'logistic_regression': (LogisticRegression, 'classification'),
    'svm': (SVC, 'classification'),
    'knn': (KNeighborsClassifier, 'classification'),
    'decision_tree': (DecisionTreeClassifier, 'classification'),
    'gradient_boosting': (GradientBoostingClassifier, 'classification'),
    'naive_bayes': (GaussianNB, 'classification'),

    # Regression
    'linear_regression': (LinearRegression, 'regression'),
    'ridge': (Ridge, 'regression'),
    'lasso': (Lasso, 'regression'),
    'random_forest_regressor': (RandomForestRegressor, 'regression'),
    'svr': (SVR, 'regression'),
    'knn_regressor': (KNeighborsRegressor, 'regression'),
    'decision_tree_regressor': (DecisionTreeRegressor, 'regression'),
    'gradient_boosting_regressor': (GradientBoostingRegressor, 'regression'),
}


@tool
def train_model(
    file_path: str,
    target_column: str,
    feature_columns: str,
    model_type: str = "random_forest",
    task_type: str = "auto",
    test_size: float = 0.2,
    scale_features: bool = False,
    cv_folds: int = 5,
    model_params: Optional[str] = None
) -> str:
    """
    Universal ML model training tool supporting multiple algorithms.

    **IN-MEMORY OPTIMIZATION**: Reads from cached DataFrame for training. Does NOT modify DataFrame.

    This tool replaces 28+ specific model training tools with a single unified interface.
    Supports classification and regression with automatic task detection.

    Args:
        file_path: Path to the CSV file containing training data
        target_column: Name of the target variable column
        feature_columns: Comma-separated list of feature column names (e.g., "Age,Fare,Pclass")
        model_type: Algorithm to use. Options:
            Classification: 'random_forest', 'logistic_regression', 'svm', 'knn',
                          'decision_tree', 'gradient_boosting', 'naive_bayes'
            Regression: 'linear_regression', 'ridge', 'lasso', 'random_forest_regressor',
                       'svr', 'knn_regressor', 'decision_tree_regressor', 'gradient_boosting_regressor'
        task_type: 'classification', 'regression', or 'auto' (auto-detect from model_type)
        test_size: Fraction of data to use for testing (0.0-1.0), default 0.2
        scale_features: Whether to apply StandardScaler to features (recommended for SVM/KNN)
        cv_folds: Number of cross-validation folds for performance estimation
        model_params: Optional JSON string with model hyperparameters (e.g., '{"n_estimators": 200, "max_depth": 10}')

    Returns:
        Comprehensive training report including:
        - Model type and task
        - Train/test split info
        - Performance metrics (accuracy, precision, recall, F1 for classification; MSE, MAE, R² for regression)
        - Cross-validation scores
        - Feature importance (if available)
        - Model save path

    Examples:
        >>> # Train Random Forest classifier
        >>> train_model("data/titanic.csv", "Survived", "Age,Fare,Pclass", "random_forest")

        >>> # Train Linear Regression with scaling
        >>> train_model("data/housing.csv", "Price", "Bedrooms,Sqft,Location", "linear_regression", scale_features=True)

        >>> # Train SVM with custom parameters
        >>> train_model("data/iris.csv", "Species", "SepalLength,SepalWidth", "svm",
        ...            model_params='{"C": 10, "kernel": "rbf"}')
    """
    try:
        # Use in-memory DataFrame (TODO #2)
        from smolagents_tools import df_state_manager
        df = df_state_manager.get_current_dataframe()
        if df is None:
            df = df_state_manager.load_dataframe(file_path)

        # Validate model type
        if model_type not in AVAILABLE_MODELS:
            available = ', '.join(AVAILABLE_MODELS.keys())
            return f"❌ Error: Unknown model_type '{model_type}'. Available: {available}"

        # Get model class and default task type
        ModelClass, default_task = AVAILABLE_MODELS[model_type]
        if task_type == "auto":
            task_type = default_task

        # Parse feature columns
        features = [f.strip() for f in feature_columns.split(',')]

        # Validate columns exist
        missing_cols = [col for col in features + [target_column] if col not in df.columns]
        if missing_cols:
            return f"❌ Error: Columns not found: {missing_cols}. Available: {list(df.columns)}"

        # Prepare data
        X = df[features].copy()
        y = df[target_column].copy()

        # Handle missing values
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        y = y.fillna(y.mode()[0] if task_type == 'classification' else y.mean())

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Feature scaling if requested
        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Parse model parameters
        params = {}
        if model_params:
            try:
                import json
                params = json.loads(model_params)
            except:
                return f"❌ Error: Invalid model_params JSON: {model_params}"

        # Initialize and train model
        model = ModelClass(**params)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate metrics based on task type
        metrics = {}
        if task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        else:  # regression
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2_score'] = r2_score(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)

        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(features, model.feature_importances_))
            # Sort by importance
            feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])

        # Save model
        os.makedirs('models', exist_ok=True)
        model_filename = f"models/{model_type}_{target_column}.pkl"
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'features': features,
            'target': target_column,
            'task_type': task_type,
            'model_type': model_type
        }, model_filename)

        # Build report
        report = [
            f"✅ Model Training Completed",
            f"",
            f"Model: {model_type} ({task_type})",
            f"Target: {target_column}",
            f"Features: {', '.join(features)}",
            f"",
            f"Dataset Split:",
            f"  • Train: {len(X_train)} samples",
            f"  • Test: {len(X_test)} samples",
            f"  • Split ratio: {test_size:.0%}",
            f""
        ]

        # Add metrics
        if task_type == 'classification':
            report.extend([
                f"Performance Metrics:",
                f"  • Accuracy: {metrics['accuracy']:.4f}",
                f"  • Precision: {metrics['precision']:.4f}",
                f"  • Recall: {metrics['recall']:.4f}",
                f"  • F1 Score: {metrics['f1_score']:.4f}",
            ])
        else:
            report.extend([
                f"Performance Metrics:",
                f"  • R² Score: {metrics['r2_score']:.4f}",
                f"  • RMSE: {metrics['rmse']:.4f}",
                f"  • MAE: {metrics['mae']:.4f}",
            ])

        report.extend([
            f"",
            f"Cross-Validation ({cv_folds}-fold):",
            f"  • Mean: {cv_scores.mean():.4f}",
            f"  • Std: {cv_scores.std():.4f}",
        ])

        if feature_importance:
            report.extend([
                f"",
                f"Top 5 Important Features:",
            ])
            for feat, imp in feature_importance.items():
                report.append(f"  • {feat}: {imp:.4f}")

        report.extend([
            f"",
            f"Model saved to: {model_filename}"
        ])

        return "\n".join(report)

    except Exception as e:
        import traceback
        return f"❌ Error training model: {str(e)}\n{traceback.format_exc()}"


@tool
def evaluate_model_universal(model_path: str, test_data_path: str, target_column: str, feature_columns: Optional[str] = None) -> str:
    """
    Evaluate a trained model on new test data.

    **IN-MEMORY OPTIMIZATION**: Reads test data from cached DataFrame. Does NOT modify DataFrame.

    Args:
        model_path: Path to the saved model file (.pkl)
        test_data_path: Path to CSV with test data
        target_column: Name of the target column in test data
        feature_columns: Optional comma-separated features (uses model's features if None)

    Returns:
        Evaluation report with metrics and predictions summary
    """
    try:
        # Load model
        if not os.path.exists(model_path):
            return f"❌ Error: Model file not found: {model_path}"

        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data.get('scaler')
        features = feature_columns.split(',') if feature_columns else model_data['features']
        task_type = model_data['task_type']

        # Load test data
        from smolagents_tools import df_state_manager
        df = df_state_manager.get_current_dataframe()
        if df is None:
            df = df_state_manager.load_dataframe(test_data_path)

        # Prepare data
        X_test = df[features]
        y_test = df[target_column]

        # Scale if needed
        if scaler:
            X_test = scaler.transform(X_test)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {}
        if task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        else:
            metrics['r2'] = r2_score(y_test, y_pred)
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])

        # Build report
        report = [
            f"✅ Model Evaluation Completed",
            f"",
            f"Model: {model_path}",
            f"Test Data: {test_data_path} ({len(y_test)} samples)",
            f"Task: {task_type}",
            f"",
            f"Performance:"
        ]

        for metric, value in metrics.items():
            report.append(f"  • {metric.upper()}: {value:.4f}")

        return "\n".join(report)

    except Exception as e:
        return f"❌ Error evaluating model: {str(e)}"


@tool
def predict_with_model_universal(model_path: str, data_path: str, feature_columns: Optional[str] = None, output_column: str = "prediction") -> str:
    """
    Make predictions using a trained model.

    **IN-MEMORY OPTIMIZATION**: Reads data from cache and adds predictions column in memory.

    Args:
        model_path: Path to saved model (.pkl)
        data_path: Path to CSV with data to predict on
        feature_columns: Optional features (uses model's features if None)
        output_column: Name for the prediction column (default: "prediction")

    Returns:
        Success message with prediction stats and first 5 predictions
    """
    try:
        # Load model
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data.get('scaler')
        features = feature_columns.split(',') if feature_columns else model_data['features']

        # Load data
        from smolagents_tools import df_state_manager
        df = df_state_manager.get_current_dataframe()
        if df is None:
            df = df_state_manager.load_dataframe(data_path)

        # Prepare features
        X = df[features]
        if scaler:
            X = scaler.transform(X)

        # Predict
        predictions = model.predict(X)

        # Add to DataFrame
        df[output_column] = predictions

        # Update state manager
        df_state_manager.update_current_dataframe(df)

        # Build report
        report = [
            f"✅ Predictions Generated",
            f"",
            f"Model: {model_path}",
            f"Samples: {len(predictions)}",
            f"Prediction column: '{output_column}'",
            f"",
            f"First 5 predictions:",
        ]

        for i in range(min(5, len(predictions))):
            report.append(f"  {i+1}. {predictions[i]}")

        report.append(f"")
        report.append(f"DataFrame updated in memory with predictions.")

        return "\n".join(report)

    except Exception as e:
        return f"❌ Error making predictions: {str(e)}"
