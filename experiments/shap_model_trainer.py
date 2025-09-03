"""
Model Training and Evaluation Module for SHAP Analysis
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_random_forest(
    X_train,
    y_train,
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
):
    """
    Train a Random Forest model for interpretability analysis.

    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees
        max_depth: Maximum depth of trees
        min_samples_split: Minimum number of samples required to split an internal node
        min_samples_leaf: Minimum number of samples required to be at a leaf node
        random_state: Random seed

    Returns:
        model: Trained Random Forest model
    """
    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )

    model.fit(X_train, y_train)
    print("Model training completed!")

    return model


def evaluate_model(model, X, y, dataset_name="Test"):
    """
    Evaluate the trained model and return predictions.

    Args:
        model: Trained model
        X: Features
        y: Labels
        dataset_name: Name of the dataset (for printing)

    Returns:
        y_pred: Predictions
        accuracy: Model accuracy
    """
    print(f"Evaluating model performance on {dataset_name} set...")

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(y, y_pred, target_names=["TDC", "ASD"]))

    return y_pred, accuracy


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Complete model training and evaluation pipeline.

    Args:
        X_train, X_test, y_train, y_test: Split datasets

    Returns:
        model: Trained model
        y_pred_test: Test predictions
        train_accuracy: Training accuracy
        test_accuracy: Test accuracy
    """
    # Train model
    model = train_random_forest(X_train, y_train)

    # Evaluate on training set
    y_pred_train, train_accuracy = evaluate_model(model, X_train, y_train, "Training")

    # Evaluate on test set
    y_pred_test, test_accuracy = evaluate_model(model, X_test, y_test, "Test")

    print(f"\n=== Model Performance Summary ===")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Overfitting Check: {train_accuracy - test_accuracy:.4f}")

    return model, y_pred_test, train_accuracy, test_accuracy
