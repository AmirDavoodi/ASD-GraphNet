"""
Utility functions for calculating performance metrics across sites.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def calculate_site_weighted_metrics(X, y, site_ids, n_splits=5, random_state=42):
    """
    Calculate weighted performance metrics across all sites using stratified k-fold cross-validation.

    Args:
        X: Features array
        y: Labels array
        site_ids: Array of site IDs for each sample
        n_splits: Number of cross-validation splits
        random_state: Random seed for reproducibility

    Returns:
        dict: Dictionary containing weighted performance metrics
    """
    # Initialize classifier
    clf = RandomForestClassifier(random_state=random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize metrics per site
    site_metrics = {}
    for site in np.unique(site_ids):
        site_metrics[site] = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "sensitivity": [],
            "specificity": [],
            "n_samples": np.sum(site_ids == site),
        }

    # Perform cross-validation
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        site_train, site_test = site_ids[train_idx], site_ids[test_idx]

        # Train model
        clf.fit(X_train, y_train)

        # Calculate metrics for each site
        for site in np.unique(site_ids):
            site_mask = site_test == site
            if np.sum(site_mask) > 0:  # Only calculate if site has test samples
                y_true_site = y_test[site_mask]
                y_pred_site = clf.predict(X_test[site_mask])

                # Calculate basic metrics
                site_metrics[site]["accuracy"].append(
                    accuracy_score(y_true_site, y_pred_site)
                )
                site_metrics[site]["precision"].append(
                    precision_score(y_true_site, y_pred_site)
                )
                site_metrics[site]["recall"].append(
                    recall_score(y_true_site, y_pred_site)
                )
                site_metrics[site]["f1"].append(f1_score(y_true_site, y_pred_site))

                # Calculate sensitivity and specificity
                tn, fp, fn, tp = confusion_matrix(y_true_site, y_pred_site).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
                site_metrics[site]["sensitivity"].append(sensitivity)
                site_metrics[site]["specificity"].append(specificity)

    # Calculate weighted averages
    total_samples = sum(metrics["n_samples"] for metrics in site_metrics.values())
    weighted_metrics = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "sensitivity": 0,
        "specificity": 0,
    }

    for site, metrics in site_metrics.items():
        weight = metrics["n_samples"] / total_samples
        for metric in weighted_metrics.keys():
            if len(metrics[metric]) > 0:  # Only include if site has metrics
                weighted_metrics[metric] += weight * np.mean(metrics[metric])

    # Convert to percentages
    for metric in weighted_metrics:
        weighted_metrics[metric] *= 100

    return weighted_metrics


def print_performance_summary(metrics):
    """
    Print a formatted summary of performance metrics.

    Args:
        metrics: Dictionary containing performance metrics
    """
    print("\nWeighted Average Performance Metrics Across All Sites:")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall: {metrics['recall']:.2f}%")
    print(f"F1 Score: {metrics['f1']:.2f}%")
    print(f"Sensitivity: {metrics['sensitivity']:.2f}%")
    print(f"Specificity: {metrics['specificity']:.2f}%")
