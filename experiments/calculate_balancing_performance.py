"""
Script to calculate performance metrics for balanced data.
"""

import sys

sys.path.append("..")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import ast

# Load the dataset
subjects_df = pd.read_csv("../outputs/filtered_subjects_df.csv")
subjects_df["features_value"] = subjects_df["features_value"].apply(ast.literal_eval)


def calculate_metrics(y_true, y_pred):
    """Calculate all performance metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred) * 100
    recall = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100
    sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) != 0 else 0
    specificity = (tn / (tn + fp)) * 100 if (tn + fp) != 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def main():
    # Process each atlas type
    for atlas in subjects_df["atlas"].unique():
        print(f"\nProcessing atlas: {atlas}")

        # Filter data for current atlas
        current_data = subjects_df[
            (subjects_df["atlas"] == atlas)
            & (subjects_df["graph_feature_type"] == "node_based")
            & (subjects_df["feature"] == "degree")
            & (subjects_df["feature_engineering"] == "mi_10")
        ]

        if current_data.empty:
            print(f"No data found for atlas {atlas}")
            continue

        # Check feature dimensions
        feature_lengths = current_data["features_value"].apply(len)
        if len(feature_lengths.unique()) > 1:
            print(f"Warning: Inconsistent feature dimensions for {atlas}")
            print(f"Found dimensions: {feature_lengths.unique()}")
            continue

        try:
            # Convert features to numpy array
            X = np.vstack([np.array(x) for x in current_data["features_value"].values])
            y = current_data["ASD"].values

            # Initialize classifier
            clf = RandomForestClassifier(random_state=42)

            # Get predictions using cross-validation
            y_pred = cross_val_predict(clf, X, y, cv=5)

            # Calculate metrics
            metrics = calculate_metrics(y, y_pred)

            print("\nPerformance Metrics:")
            print(f"Accuracy: {metrics['accuracy']:.2f}%")
            print(f"Precision: {metrics['precision']:.2f}%")
            print(f"Recall: {metrics['recall']:.2f}%")
            print(f"F1 Score: {metrics['f1']:.2f}%")
            print(f"Sensitivity: {metrics['sensitivity']:.2f}%")
            print(f"Specificity: {metrics['specificity']:.2f}%")

        except Exception as e:
            print(f"Error processing {atlas}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
