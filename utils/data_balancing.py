"""
Utility functions for handling imbalanced datasets in the ASD-GraphNet project.
"""

import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN


def get_class_distribution(y):
    """
    Get the distribution of classes in the dataset.

    Args:
        y: Labels array

    Returns:
        dict: Dictionary with class counts
    """
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))


def apply_random_oversampling(X, y, random_state=42):
    """
    Apply random oversampling to balance the dataset.

    Args:
        X: Features
        y: Labels
        random_state: Random seed for reproducibility

    Returns:
        X_resampled, y_resampled: Balanced dataset
    """
    # Separate majority and minority classes
    X_array = np.array(X)
    classes = np.unique(y)
    max_size = max([sum(y == c) for c in classes])

    X_resampled = []
    y_resampled = []

    for c in classes:
        idx = y == c
        X_c = X_array[idx]
        y_c = y[idx]

        # If this is the minority class, oversample
        if len(X_c) < max_size:
            X_c_resampled, y_c_resampled = resample(
                X_c, y_c, replace=True, n_samples=max_size, random_state=random_state
            )
        else:
            X_c_resampled, y_c_resampled = X_c, y_c

        X_resampled.extend(X_c_resampled)
        y_resampled.extend(y_c_resampled)

    return np.array(X_resampled), np.array(y_resampled)


def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE to balance the dataset.

    Args:
        X: Features
        y: Labels
        random_state: Random seed for reproducibility

    Returns:
        X_resampled, y_resampled: Balanced dataset
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def apply_adasyn(X, y, random_state=42):
    """
    Apply ADASYN to balance the dataset.

    Args:
        X: Features
        y: Labels
        random_state: Random seed for reproducibility

    Returns:
        X_resampled, y_resampled: Balanced dataset
    """
    adasyn = ADASYN(random_state=random_state)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled


def apply_hybrid_sampling(X, y, method="smote_tomek", random_state=42):
    """
    Apply hybrid sampling methods to balance the dataset.

    Args:
        X: Features
        y: Labels
        method: 'smote_tomek' or 'smote_enn'
        random_state: Random seed for reproducibility

    Returns:
        X_resampled, y_resampled: Balanced dataset
    """
    if method == "smote_tomek":
        sampler = SMOTETomek(random_state=random_state)
    else:  # smote_enn
        sampler = SMOTEENN(random_state=random_state)

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled
