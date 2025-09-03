"""
Data Loading and Preprocessing Module for SHAP Analysis
"""

import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split


def load_and_filter_data(
    csv_path="../outputs/filtered_subjects_df.csv",
    atlas="cc200",
    graph_feature_type="node_based",
    feature="degree",
    feature_engineering="original",
):
    """
    Load and filter the dataset for SHAP analysis.

    Args:
        csv_path: Path to the CSV file
        atlas: Brain atlas to use
        graph_feature_type: Type of graph features
        feature: Specific feature to analyze
        feature_engineering: Feature engineering method

    Returns:
        filtered_df: Filtered DataFrame
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Total samples in dataset: {len(df)}")
    print(f"Unique atlases: {df['atlas'].unique()}")
    print(f"Unique feature types: {df['graph_feature_type'].unique()}")
    print(f"Unique features: {df['feature'].unique()}")
    print(f"Feature engineering types: {df['feature_engineering'].unique()}")

    # Filter for specified configuration
    filtered_df = df[
        (df["atlas"] == atlas)
        & (df["graph_feature_type"] == graph_feature_type)
        & (df["feature"] == feature)
        & (df["feature_engineering"] == feature_engineering)
    ]

    print(f"Filtered samples: {len(filtered_df)}")
    print(f"Class distribution:")
    print(filtered_df["ASD"].value_counts())

    return filtered_df


def convert_string_to_array(s):
    """
    Convert string representation of arrays to numpy arrays.
    Handles both regular string lists and those with 'np.str_' prefix.

    Args:
        s: String representation of array

    Returns:
        numpy array
    """
    # Remove 'np.str_' from the string if present
    s = s.replace("np.str_", "")
    return np.array(ast.literal_eval(s), dtype=float)


def prepare_features_and_labels(filtered_df):
    """
    Parse features from string format and prepare X, y arrays.

    Args:
        filtered_df: Filtered DataFrame

    Returns:
        X: Feature matrix
        y: Labels
        test_flags: Test/train flags (0=train, 1=test)
    """
    print("Parsing feature values from string format...")

    # Parse string representations back to arrays using the convert function
    X = np.stack([convert_string_to_array(x) for x in filtered_df["features_value"]])
    y = filtered_df["ASD"].values
    test_flags = filtered_df["test"].values

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of brain regions (features): {X.shape[1]}")
    print(f"Sample feature values for first subject: {X[0][:10]}...")

    return X, y, test_flags


def analyze_stratification_by_site(filtered_df):
    """
    Analyze the train/test split proportions for each SITE_ID.

    Args:
        filtered_df: Filtered DataFrame with 'test' and 'SITE_ID' columns

    Returns:
        site_analysis: DataFrame with split analysis by site
    """
    print("\n=== Analyzing Train/Test Split by SITE_ID ===")

    # Group by SITE_ID and analyze the split
    site_analysis = []

    for site_id in sorted(filtered_df["SITE_ID"].unique()):
        site_data = filtered_df[filtered_df["SITE_ID"] == site_id]

        total_samples = len(site_data)
        train_samples = len(site_data[site_data["test"] == 0])
        test_samples = len(site_data[site_data["test"] == 1])

        train_percent = (train_samples / total_samples) * 100
        test_percent = (test_samples / total_samples) * 100

        # Analyze ASD distribution within each split
        train_asd = len(site_data[(site_data["test"] == 0) & (site_data["ASD"] == 1)])
        train_tdc = len(site_data[(site_data["test"] == 0) & (site_data["ASD"] == 0)])
        test_asd = len(site_data[(site_data["test"] == 1) & (site_data["ASD"] == 1)])
        test_tdc = len(site_data[(site_data["test"] == 1) & (site_data["ASD"] == 0)])

        site_analysis.append(
            {
                "SITE_ID": site_id,
                "Total_Samples": total_samples,
                "Train_Samples": train_samples,
                "Test_Samples": test_samples,
                "Train_Percent": train_percent,
                "Test_Percent": test_percent,
                "Train_ASD": train_asd,
                "Train_TDC": train_tdc,
                "Test_ASD": test_asd,
                "Test_TDC": test_tdc,
            }
        )

    site_analysis_df = pd.DataFrame(site_analysis)

    print(f"Number of unique sites: {len(site_analysis_df)}")
    print(
        f"Overall train/test split: {len(filtered_df[filtered_df['test'] == 0])}/{len(filtered_df[filtered_df['test'] == 1])}"
    )
    print(
        f"Overall test percentage: {(len(filtered_df[filtered_df['test'] == 1]) / len(filtered_df)) * 100:.2f}%"
    )

    print("\nPer-site analysis:")
    print(
        site_analysis_df[
            [
                "SITE_ID",
                "Total_Samples",
                "Train_Samples",
                "Test_Samples",
                "Train_Percent",
                "Test_Percent",
            ]
        ].to_string(index=False)
    )

    print(f"\nTest percentage statistics across sites:")
    print(f"Mean: {site_analysis_df['Test_Percent'].mean():.2f}%")
    print(f"Std: {site_analysis_df['Test_Percent'].std():.2f}%")
    print(f"Min: {site_analysis_df['Test_Percent'].min():.2f}%")
    print(f"Max: {site_analysis_df['Test_Percent'].max():.2f}%")

    return site_analysis_df


def split_data_by_test_column(X, y, test_flags):
    """
    Split data based on the existing 'test' column.

    Args:
        X: Feature matrix
        y: Labels
        test_flags: Test/train flags (0=train, 1=test)

    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    # Split based on test column
    train_mask = test_flags == 0
    test_mask = test_flags == 1

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training ASD distribution: {np.bincount(y_train)}")
    print(f"Test ASD distribution: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test


def load_shap_data(
    csv_path="../outputs/filtered_subjects_df.csv",
    atlas: str = "cc200",
    graph_feature_type: str = "node_based",
    feature: str = "degree",
    feature_engineering: str = "original",
):
    """
    Complete data loading pipeline for SHAP analysis using existing test column.

    Args:
        csv_path: Path to the CSV file

    Returns:
        X_train, X_test, y_train, y_test: Split datasets
        feature_names: List of feature names
        site_analysis_df: Analysis of stratification by site
    """
    # Load and filter data
    filtered_df = load_and_filter_data(
        csv_path, atlas, graph_feature_type, feature, feature_engineering
    )

    # Analyze stratification by site
    site_analysis_df = analyze_stratification_by_site(filtered_df)

    # Prepare features and labels
    X, y, test_flags = prepare_features_and_labels(filtered_df)

    # Split data using existing test column
    X_train, X_test, y_train, y_test = split_data_by_test_column(X, y, test_flags)

    # Create feature names
    feature_names = [f"Region_{i+1:03d}" for i in range(X.shape[1])]

    return X_train, X_test, y_train, y_test, feature_names, site_analysis_df
