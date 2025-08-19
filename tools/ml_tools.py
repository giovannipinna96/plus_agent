"""Machine Learning tools."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Dict, Any, List, Optional, Union
from langchain.tools import tool


@tool
def train_regression_model(file_path: str, target_column: str, feature_columns: str, model_type: str = "linear") -> str:
    """
    Train a regression model.
    
    Args:
        file_path: Path to the data file
        target_column: Name of the target variable column
        feature_columns: Comma-separated list of feature columns
        model_type: Type of regression (linear, random_forest)
        
    Returns:
        String describing the model training results
    """
    try:
        df = pd.read_csv(file_path)
        
        # Parse feature columns
        features = [col.strip() for col in feature_columns.split(',')]
        
        # Check if columns exist
        for col in features + [target_column]:
            if col not in df.columns:
                return f"Column '{col}' not found in dataset"
        
        # Prepare data
        X = df[features]
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Handle categorical variables in features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            return f"Unknown regression model type '{model_type}'"
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        # Save model
        model_path = file_path.replace('.csv', f'_{model_type}_regression_model.joblib')
        joblib.dump(model, model_path)
        
        results = {
            "model_type": f"{model_type} regression",
            "features": features,
            "target": target_column,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "cv_r2_mean": round(cv_scores.mean(), 4),
            "cv_r2_std": round(cv_scores.std(), 4),
            "model_saved": model_path
        }
        
        return f"Regression model trained: {results}"
        
    except Exception as e:
        return f"Error training regression model: {str(e)}"


@tool
def train_svm_model(file_path: str, target_column: str, feature_columns: str, task_type: str = "classification") -> str:
    """
    Train a Support Vector Machine model.
    
    Args:
        file_path: Path to the data file
        target_column: Name of the target variable column
        feature_columns: Comma-separated list of feature columns
        task_type: Type of task (classification, regression)
        
    Returns:
        String describing the model training results
    """
    try:
        df = pd.read_csv(file_path)
        
        # Parse feature columns
        features = [col.strip() for col in feature_columns.split(',')]
        
        # Check if columns exist
        for col in features + [target_column]:
            if col not in df.columns:
                return f"Column '{col}' not found in dataset"
        
        # Prepare data
        X = df[features]
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        if task_type == "classification":
            y = y.fillna(y.mode()[0])
        else:
            y = y.fillna(y.mean())
        
        # Handle categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        if task_type == "classification" and y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if task_type == "classification":
            model = SVC(kernel='rbf', random_state=42)
            scoring_metric = 'accuracy'
        else:
            model = SVR(kernel='rbf')
            scoring_metric = 'r2'
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                "mse": round(mse, 4),
                "r2_score": round(r2, 4)
            }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring_metric)
        
        # Save model
        model_path = file_path.replace('.csv', f'_svm_{task_type}_model.joblib')
        joblib.dump(model, model_path)
        
        results = {
            "model_type": f"SVM {task_type}",
            "features": features,
            "target": target_column,
            "train_size": len(X_train),
            "test_size": len(X_test),
            **metrics,
            "cv_score_mean": round(cv_scores.mean(), 4),
            "cv_score_std": round(cv_scores.std(), 4),
            "model_saved": model_path
        }
        
        return f"SVM model trained: {results}"
        
    except Exception as e:
        return f"Error training SVM model: {str(e)}"


@tool
def train_random_forest_model(file_path: str, target_column: str, feature_columns: str, task_type: str = "classification") -> str:
    """
    Train a Random Forest model.
    
    Args:
        file_path: Path to the data file
        target_column: Name of the target variable column
        feature_columns: Comma-separated list of feature columns
        task_type: Type of task (classification, regression)
        
    Returns:
        String describing the model training results including feature importance
    """
    try:
        df = pd.read_csv(file_path)
        
        # Parse feature columns
        features = [col.strip() for col in feature_columns.split(',')]
        
        # Check if columns exist
        for col in features + [target_column]:
            if col not in df.columns:
                return f"Column '{col}' not found in dataset"
        
        # Prepare data
        X = df[features]
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        if task_type == "classification":
            y = y.fillna(y.mode()[0])
        else:
            y = y.fillna(y.mean())
        
        # Handle categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        if task_type == "classification" and y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            scoring_metric = 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            scoring_metric = 'r2'
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                "mse": round(mse, 4),
                "r2_score": round(r2, 4)
            }
        
        # Feature importance
        feature_importance = dict(zip(features, [round(imp, 4) for imp in model.feature_importances_]))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring_metric)
        
        # Save model
        model_path = file_path.replace('.csv', f'_rf_{task_type}_model.joblib')
        joblib.dump(model, model_path)
        
        results = {
            "model_type": f"Random Forest {task_type}",
            "features": features,
            "target": target_column,
            "train_size": len(X_train),
            "test_size": len(X_test),
            **metrics,
            "cv_score_mean": round(cv_scores.mean(), 4),
            "cv_score_std": round(cv_scores.std(), 4),
            "feature_importance": feature_importance,
            "model_saved": model_path
        }
        
        return f"Random Forest model trained: {results}"
        
    except Exception as e:
        return f"Error training Random Forest model: {str(e)}"


@tool
def train_knn_model(file_path: str, target_column: str, feature_columns: str, task_type: str = "classification", n_neighbors: int = 5) -> str:
    """
    Train a K-Nearest Neighbors model.
    
    Args:
        file_path: Path to the data file
        target_column: Name of the target variable column
        feature_columns: Comma-separated list of feature columns
        task_type: Type of task (classification, regression)
        n_neighbors: Number of neighbors to use
        
    Returns:
        String describing the model training results
    """
    try:
        df = pd.read_csv(file_path)
        
        # Parse feature columns
        features = [col.strip() for col in feature_columns.split(',')]
        
        # Check if columns exist
        for col in features + [target_column]:
            if col not in df.columns:
                return f"Column '{col}' not found in dataset"
        
        # Prepare data
        X = df[features]
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        if task_type == "classification":
            y = y.fillna(y.mode()[0])
        else:
            y = y.fillna(y.mean())
        
        # Handle categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        if task_type == "classification" and y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if task_type == "classification":
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            scoring_metric = 'accuracy'
        else:
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
            scoring_metric = 'r2'
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                "mse": round(mse, 4),
                "r2_score": round(r2, 4)
            }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring_metric)
        
        # Save model
        model_path = file_path.replace('.csv', f'_knn_{task_type}_model.joblib')
        joblib.dump(model, model_path)
        
        results = {
            "model_type": f"KNN {task_type}",
            "n_neighbors": n_neighbors,
            "features": features,
            "target": target_column,
            "train_size": len(X_train),
            "test_size": len(X_test),
            **metrics,
            "cv_score_mean": round(cv_scores.mean(), 4),
            "cv_score_std": round(cv_scores.std(), 4),
            "model_saved": model_path
        }
        
        return f"KNN model trained: {results}"
        
    except Exception as e:
        return f"Error training KNN model: {str(e)}"


@tool
def evaluate_model(model_path: str, test_data_path: str, target_column: str, feature_columns: str) -> str:
    """
    Evaluate a trained model on new test data.
    
    Args:
        model_path: Path to the saved model file
        test_data_path: Path to the test data file
        target_column: Name of the target variable column
        feature_columns: Comma-separated list of feature columns
        
    Returns:
        String describing the model evaluation results
    """
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Load test data
        df = pd.read_csv(test_data_path)
        
        # Parse feature columns
        features = [col.strip() for col in feature_columns.split(',')]
        
        # Check if columns exist
        for col in features + [target_column]:
            if col not in df.columns:
                return f"Column '{col}' not found in test dataset"
        
        # Prepare test data
        X = df[features]
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Handle categorical variables (same as training)
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Determine if this is classification or regression based on model type
        model_name = str(type(model).__name__)
        is_classification = any(clf in model_name for clf in ['Classifier', 'SVC'])
        
        # Calculate appropriate metrics
        if is_classification:
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            results = {
                "model_type": model_name,
                "task": "classification",
                "test_samples": len(X),
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        else:
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mse)
            
            results = {
                "model_type": model_name,
                "task": "regression",
                "test_samples": len(X),
                "mse": round(mse, 4),
                "rmse": round(rmse, 4),
                "r2_score": round(r2, 4)
            }
        
        return f"Model evaluation results: {results}"
        
    except Exception as e:
        return f"Error evaluating model: {str(e)}"