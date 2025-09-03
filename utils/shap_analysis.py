"""
Module for SHAP (SHapley Additive exPlanations) analysis of the ASD classification model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def prepare_data_for_shap(
    df,
    atlas="cc200",
    feature_type="node_based",
    feature="degree",
    engineering="original",
):
    """
    Prepare data for SHAP analysis by filtering and organizing features.

    Args:
        df: DataFrame containing the dataset
        atlas: Brain atlas to use ('cc200', 'aal', or 'dos160')
        feature_type: Type of graph features ('node_based', 'edge_based', or 'graph_level')
        feature: Specific feature to analyze ('degree', 'average_degree', etc.)
        engineering: Feature engineering method ('original', 'mi_10', 'pca_10')

    Returns:
        X: Feature matrix
        y: Labels
        test_split: Test split indicators (0=train, 1=test)
        feature_names: List of feature names
    """
    # Filter dataset based on parameters
    filtered_df = df[
        (df["atlas"] == atlas)
        & (df["graph_feature_type"] == feature_type)
        & (df["feature"] == feature)
        & (df["feature_engineering"] == engineering)
    ]

    # Convert string representation of lists to actual numpy arrays
    def convert_string_to_array(s):
        # Remove 'np.str_' from the string and evaluate
        s = s.replace("np.str_", "")
        return np.array(ast.literal_eval(s), dtype=float)

    # Prepare features and labels
    X = np.stack([convert_string_to_array(x) for x in filtered_df["features_value"]])
    y = filtered_df["ASD"].values
    test_split = filtered_df["test"].values

    # Create feature names
    feature_names = [f"Region_{i+1}" for i in range(X.shape[1])]

    return X, y, test_split, feature_names


def train_and_explain_model(X, y, test_split, feature_names, random_state=42):
    """
    Train a Random Forest model and calculate SHAP values using existing train/test split.

    Args:
        X: Feature matrix
        y: Labels
        test_split: Test split indicators (0=train, 1=test)
        feature_names: List of feature names
        random_state: Random seed for reproducibility

    Returns:
        model: Trained Random Forest model
        shap_values: SHAP values for the training data
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    # Split the data based on the existing 'test' column
    X_train = X[test_split == 0]
    X_test = X[test_split == 1]
    y_train = y[test_split == 0]
    y_test = y[test_split == 1]

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)

    # Set feature names
    shap_values.feature_names = feature_names

    return model, shap_values, X_train, X_test, y_train, y_test


def plot_global_importance(shap_values, X_train, feature_names, output_dir):
    """
    Create and save global feature importance plots.

    Args:
        shap_values: SHAP values
        X_train: Training features
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(0)

    # Create DataFrame for plotting
    feature_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": mean_abs_shap.flatten(),  # Ensure 1D array
        }
    )
    feature_importance = feature_importance.sort_values("Importance", ascending=True)

    # Bar plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(feature_importance)), feature_importance["Importance"])
    plt.yticks(range(len(feature_importance)), feature_importance["Feature"])
    plt.xlabel("Mean |SHAP Value|")
    plt.title("Feature Importance (Impact on Model Output)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_feature_importance_bar.png")
    plt.close()

    # Summary plot (beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.title("Feature Impact on Model Predictions")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_feature_importance_beeswarm.png")
    plt.close()


def plot_feature_dependence(shap_values, X_train, feature_names, output_dir, top_n=10):
    """
    Create and save dependence plots for top features.

    Args:
        shap_values: SHAP values (shap.Explanation object)
        X_train: Training features
        feature_names: List of feature names
        output_dir: Directory to save plots
        top_n: Number of top features to plot
    """
    # Get top N most important features
    feature_importance = np.abs(shap_values.values).mean(0)
    top_features_idx = np.argsort(feature_importance)[-top_n:]

    # Create dependence plots for top features
    for idx in top_features_idx:
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train[:, idx], shap_values.values[:, idx], alpha=0.5)
        plt.xlabel(f"Feature Value ({feature_names[idx]})")
        plt.ylabel("SHAP Value")
        plt.title(f"SHAP Dependence Plot for {feature_names[idx]}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_dependence_{feature_names[idx]}.png")
        plt.close()


def plot_individual_predictions(
    explainer, X_test, feature_names, output_dir, num_examples=5
):
    """
    Create and save individual prediction explanations.

    Args:
        explainer: SHAP explainer object
        X_test: Test features
        feature_names: List of feature names
        output_dir: Directory to save plots
        num_examples: Number of individual examples to plot
    """
    # Select random examples
    indices = np.random.choice(
        len(X_test), min(num_examples, len(X_test)), replace=False
    )

    for i, idx in enumerate(indices):
        # Calculate SHAP values for this instance
        shap_values = explainer(X_test[idx : idx + 1])

        # Sort features by absolute SHAP value
        feature_importance = pd.DataFrame(
            {"Feature": feature_names, "SHAP Value": shap_values.values[0]}
        )
        feature_importance = feature_importance.sort_values(
            "SHAP Value", key=abs, ascending=True
        )

        # Plot waterfall chart
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importance)), feature_importance["SHAP Value"])
        plt.yticks(range(len(feature_importance)), feature_importance["Feature"])
        plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        plt.title(f"Individual Prediction Explanation (Example {i+1})")
        plt.xlabel("SHAP Value (Impact on Model Output)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_waterfall_example_{i+1}.png")
        plt.close()


def save_feature_importance(shap_values, feature_names, output_dir):
    """
    Save feature importance scores to a CSV file.

    Args:
        shap_values: SHAP values
        feature_names: List of feature names
        output_dir: Directory to save the file
    """
    # Calculate feature importance
    feature_importance = np.abs(shap_values).mean(0)

    # Create and save DataFrame
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importance}
    )
    importance_df = importance_df.sort_values("Importance", ascending=False)
    importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)

    return importance_df
