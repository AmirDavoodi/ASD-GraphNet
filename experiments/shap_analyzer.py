"""
SHAP Analysis Module for Model Interpretability
"""

import numpy as np
import pandas as pd
import shap


def calculate_shap_values(model, X_train, X_test):
    """
    Calculate SHAP values for the trained model.

    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features

    Returns:
        explainer: SHAP explainer
        shap_values_train: SHAP values for training set
        shap_values_test: SHAP values for test set
    """
    print("Calculating SHAP values...")

    # Initialize SHAP explainer for tree-based models
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for training set (for global analysis)
    print("Calculating SHAP values for training set...")
    shap_values_train = explainer.shap_values(X_train)

    # Calculate SHAP values for test set (for individual predictions)
    print("Calculating SHAP values for test set...")
    shap_values_test = explainer.shap_values(X_test)

    print(f"SHAP values shape (train): {shap_values_train[1].shape}")
    print(f"SHAP values shape (test): {shap_values_test[1].shape}")
    print(f"Base value (average prediction): {explainer.expected_value[1]:.4f}")

    return explainer, shap_values_train, shap_values_test


def create_feature_importance_dataframe(
    shap_values_train, X_train, y_train, feature_names, model
):
    """
    Create a comprehensive dataframe with feature importance analysis.

    Args:
        shap_values_train: SHAP values for training set
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
        model: Trained model

    Returns:
        importance_df: DataFrame with comprehensive feature analysis
    """
    print("Creating feature importance analysis...")

    # Calculate SHAP importance - handle both list and array cases
    if isinstance(shap_values_train, list):
        # For binary classification, use SHAP values for positive class
        feature_importance = np.abs(shap_values_train[1]).mean(0)
    else:
        # For 3D array (samples, features, classes), get positive class values
        if shap_values_train.ndim == 3:
            # Extract SHAP values for positive class (ASD = class 1)
            feature_importance = np.abs(shap_values_train[:, :, 1]).mean(0)
        else:
            # For single output
            feature_importance = np.abs(shap_values_train).mean(0)

    # Calculate connectivity statistics
    mean_connectivity_asd = [
        X_train[y_train == 1, i].mean() for i in range(X_train.shape[1])
    ]
    mean_connectivity_tdc = [
        X_train[y_train == 0, i].mean() for i in range(X_train.shape[1])
    ]

    # Create comprehensive dataframe
    importance_df = pd.DataFrame(
        {
            "Region_ID": range(X_train.shape[1]),
            "Region_Name": feature_names,
            "SHAP_Importance": feature_importance,
            "RF_Importance": model.feature_importances_,
            "Mean_Connectivity_ASD": mean_connectivity_asd,
            "Mean_Connectivity_TDC": mean_connectivity_tdc,
        }
    )

    # Calculate connectivity difference between groups
    importance_df["Connectivity_Difference"] = (
        importance_df["Mean_Connectivity_ASD"] - importance_df["Mean_Connectivity_TDC"]
    )

    # Sort by SHAP importance
    importance_df = importance_df.sort_values("SHAP_Importance", ascending=False)

    return importance_df


def get_top_features(importance_df, n_features=10):
    """
    Get top N most important features.

    Args:
        importance_df: Feature importance dataframe
        n_features: Number of top features to return

    Returns:
        top_features_df: Top N features
        top_features_idx: Indices of top features
    """
    top_features_df = importance_df.head(n_features)
    top_features_idx = top_features_df["Region_ID"].values

    print(f"Top {n_features} most important brain regions:")
    print(
        top_features_df[
            ["Region_Name", "SHAP_Importance", "Connectivity_Difference"]
        ].to_string(index=False)
    )

    return top_features_df, top_features_idx


def analyze_connectivity_patterns(importance_df):
    """
    Analyze connectivity patterns between ASD and TDC groups.

    Args:
        importance_df: Feature importance dataframe

    Returns:
        analysis_stats: Dictionary with analysis statistics
    """
    # Count regions with positive vs negative connectivity differences
    hyper_regions = len(importance_df[importance_df["Connectivity_Difference"] > 0])
    hypo_regions = len(importance_df[importance_df["Connectivity_Difference"] < 0])

    analysis_stats = {
        "hyper_regions": hyper_regions,
        "hypo_regions": hypo_regions,
        "total_regions": len(importance_df),
        "mean_shap_importance": importance_df["SHAP_Importance"].mean(),
        "std_shap_importance": importance_df["SHAP_Importance"].std(),
        "most_important_region": importance_df.iloc[0]["Region_Name"],
        "most_important_shap": importance_df.iloc[0]["SHAP_Importance"],
    }

    print(f"Regions with higher connectivity in ASD: {hyper_regions}")
    print(f"Regions with lower connectivity in ASD: {hypo_regions}")
    print(
        f"Most important region: {analysis_stats['most_important_region']} (SHAP: {analysis_stats['most_important_shap']:.4f})"
    )

    return analysis_stats


def save_analysis_results(
    importance_df, output_path="../outputs/comprehensive_feature_importance.csv"
):
    """
    Save feature importance analysis to CSV.

    Args:
        importance_df: Feature importance dataframe
        output_path: Path to save the CSV file
    """
    importance_df.to_csv(output_path, index=False)
    print(f"Feature importance analysis saved to: {output_path}")


def complete_shap_analysis(model, X_train, X_test, y_train, feature_names):
    """
    Complete SHAP analysis pipeline.

    Args:
        model: Trained model
        X_train, X_test: Training and test features
        y_train: Training labels
        feature_names: List of feature names

    Returns:
        explainer: SHAP explainer
        shap_values_train, shap_values_test: SHAP values
        importance_df: Feature importance dataframe
        analysis_stats: Analysis statistics
    """
    # Calculate SHAP values
    explainer, shap_values_train, shap_values_test = calculate_shap_values(
        model, X_train, X_test
    )

    # Create feature importance dataframe
    importance_df = create_feature_importance_dataframe(
        shap_values_train, X_train, y_train, feature_names, model
    )

    # Analyze connectivity patterns
    analysis_stats = analyze_connectivity_patterns(importance_df)

    # Save results
    save_analysis_results(importance_df)

    return explainer, shap_values_train, shap_values_test, importance_df, analysis_stats
