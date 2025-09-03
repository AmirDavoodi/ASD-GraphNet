"""
SHAP Visualization Module for Model Interpretability
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap


def plot_global_shap_summary(
    shap_values_train,
    X_train,
    feature_names,
    output_path="../outputs/shap_summary_plot.png",
):
    """
    Create SHAP summary plot showing feature importance and effects.

    Args:
        shap_values_train: SHAP values for training set
        X_train: Training features
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))

    # Handle 3D SHAP values array
    if shap_values_train.ndim == 3:
        shap_values_for_plot = shap_values_train[:, :, 1]  # ASD class
    else:
        shap_values_for_plot = (
            shap_values_train[1]
            if isinstance(shap_values_train, list)
            else shap_values_train
        )

    shap.summary_plot(
        shap_values_for_plot,
        X_train,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.title(
        "SHAP Summary Plot: Feature Impact on ASD Classification", fontsize=16, pad=20
    )
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Summary Plot Interpretation:")
    print("- Each dot represents one patient's SHAP value for a specific brain region")
    print(
        "- X-axis: SHAP value (positive = increases ASD probability, negative = decreases)"
    )
    print("- Color: Feature value (red = high connectivity, blue = low connectivity)")
    print("- Features are ranked by importance (most important at top)")


def plot_shap_beeswarm(
    shap_values_train,
    X_train,
    feature_names,
    output_path="../outputs/shap_beeswarm_plot.png",
):
    """
    Create SHAP beeswarm plot (alternative visualization).

    Args:
        shap_values_train: SHAP values for training set
        X_train: Training features
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))

    # Handle 3D SHAP values array
    if shap_values_train.ndim == 3:
        shap_values_for_plot = shap_values_train[:, :, 1]  # ASD class
    else:
        shap_values_for_plot = (
            shap_values_train[1]
            if isinstance(shap_values_train, list)
            else shap_values_train
        )

    shap.plots.beeswarm(
        shap.Explanation(
            values=shap_values_for_plot, data=X_train, feature_names=feature_names
        ),
        max_display=20,
        show=False,
    )
    plt.title(
        "SHAP Beeswarm Plot: Distribution of Feature Impacts", fontsize=16, pad=20
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_feature_importance_bar(
    shap_values_train,
    X_train,
    feature_names,
    output_path="../outputs/shap_feature_importance_bar.png",
):
    """
    Create SHAP feature importance bar plot.

    Args:
        shap_values_train: SHAP values for training set
        X_train: Training features
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Handle 3D SHAP values array
    if shap_values_train.ndim == 3:
        shap_values_for_plot = shap_values_train[:, :, 1]  # ASD class
    else:
        shap_values_for_plot = (
            shap_values_train[1]
            if isinstance(shap_values_train, list)
            else shap_values_train
        )

    shap.summary_plot(
        shap_values_for_plot,
        X_train,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.title(
        "SHAP Feature Importance: Mean Absolute Impact on ASD Classification",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("Mean |SHAP Value|", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_individual_waterfall(
    explainer, shap_values_test, X_test, y_test, y_pred, feature_names, case_type="ASD"
):
    """
    Create waterfall plots for individual predictions.

    Args:
        explainer: SHAP explainer
        shap_values_test: SHAP values for test set
        X_test: Test features
        y_test: Test labels
        y_pred: Predictions
        feature_names: List of feature names
        case_type: 'ASD' or 'TDC'

    Returns:
        case_idx: Index of the selected case
    """
    if case_type == "ASD":
        correct_cases = np.where((y_test == 1) & (y_pred == 1))[0]
        title_prefix = "ASD Patient"
    else:
        correct_cases = np.where((y_test == 0) & (y_pred == 0))[0]
        title_prefix = "Typical Control"

    if len(correct_cases) > 0:
        case_idx = correct_cases[0]

        plt.figure(figsize=(12, 8))

        # Handle 3D SHAP values array
        if shap_values_test.ndim == 3:
            shap_values_for_case = shap_values_test[
                case_idx, :, 1
            ]  # ASD class for this case
            expected_value = explainer.expected_value[1]
        else:
            shap_values_for_case = (
                shap_values_test[1][case_idx]
                if isinstance(shap_values_test, list)
                else shap_values_test[case_idx]
            )
            expected_value = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, list)
                else explainer.expected_value
            )

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values_for_case,
                base_values=expected_value,
                data=X_test[case_idx],
                feature_names=feature_names,
            ),
            max_display=15,
            show=False,
        )
        plt.title(
            f"Individual Prediction Explanation: {title_prefix} (Case {case_idx})",
            fontsize=16,
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            f"../outputs/shap_waterfall_{case_type.lower()}_case.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        if case_type == "ASD":
            print("Waterfall Plot Interpretation:")
            print("- Shows how each feature pushes the prediction from the baseline")
            print("- Baseline: Average model prediction across all patients")
            print("- Red bars: Features increasing ASD probability")
            print("- Blue bars: Features decreasing ASD probability")
            print("- Final prediction at the top")

        return case_idx
    else:
        print(f"No correctly classified {case_type} cases found!")
        return None


def plot_force_plot(
    explainer, shap_values_test, X_test, case_idx, feature_names, case_type="ASD"
):
    """
    Create force plot for individual prediction.

    Args:
        explainer: SHAP explainer
        shap_values_test: SHAP values for test set
        X_test: Test features
        case_idx: Case index
        feature_names: List of feature names
        case_type: 'ASD' or 'TDC'
    """
    if case_idx is not None:
        # Handle 3D SHAP values array
        if shap_values_test.ndim == 3:
            shap_values_for_case = shap_values_test[
                case_idx, :, 1
            ]  # ASD class for this case
            expected_value = explainer.expected_value[1]
        else:
            shap_values_for_case = (
                shap_values_test[1][case_idx]
                if isinstance(shap_values_test, list)
                else shap_values_test[case_idx]
            )
            expected_value = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, list)
                else explainer.expected_value
            )

        shap.force_plot(
            expected_value,
            shap_values_for_case,
            X_test[case_idx],
            feature_names=feature_names,
            matplotlib=True,
            figsize=(16, 4),
            show=False,
        )
        plt.title(
            f"Force Plot: {case_type} Patient (Case {case_idx})", fontsize=14, pad=20
        )
        plt.tight_layout()
        plt.savefig(
            f"../outputs/shap_force_{case_type.lower()}_case.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        if case_type == "ASD":
            print("Force Plot Interpretation:")
            print(
                "- Shows forces pushing prediction higher (red) or lower (blue) than baseline"
            )
            print("- Arrow length represents the magnitude of each feature's impact")
            print("- Feature values are shown for context")


def plot_dependence_analysis(
    shap_values_train,
    X_train,
    feature_names,
    top_features_idx,
    output_path="../outputs/shap_dependence_plots.png",
):
    """
    Create dependence plots for top features.

    Args:
        shap_values_train: SHAP values for training set
        X_train: Training features
        feature_names: List of feature names
        top_features_idx: Indices of top features
        output_path: Path to save the plot
    """
    # Set to 6 features to include Region 104 and top 5 others
    n_features = min(6, len(top_features_idx))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Handle 3D SHAP values array
    if shap_values_train.ndim == 3:
        shap_values_for_plot = shap_values_train[:, :, 1]  # ASD class
    else:
        shap_values_for_plot = (
            shap_values_train[1]
            if isinstance(shap_values_train, list)
            else shap_values_train
        )

    # Get the top 6 most important features (in descending order of importance)
    # top_features_idx should already be sorted by importance
    top_6_indices = top_features_idx[:n_features]  # Take first 6 (most important)

    print(f"Creating dependence plots for top {n_features} features:")
    for i, idx in enumerate(top_6_indices):
        print(f"  {i+1}. {feature_names[idx]} (index: {idx})")

    for i in range(n_features):
        idx = top_6_indices[i]  # Get the i-th most important feature
        ax = axes[i]

        # Use traditional dependence plot which works better with subplots
        shap.dependence_plot(
            idx,
            shap_values_for_plot,
            X_train,
            feature_names=feature_names,
            ax=ax,
            show=False,
        )
        ax.set_title(f"Dependence: {feature_names[idx]}", fontsize=12)
        ax.set_xlabel("Feature Value (Degree Connectivity)", fontsize=10)
        ax.set_ylabel("SHAP Value", fontsize=10)

    # Hide remaining subplots if any
    for i in range(n_features, 6):
        axes[i].set_visible(False)

    plt.suptitle("SHAP Dependence Plots: Top 6 Brain Regions", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Dependence Plot Interpretation:")
    print("- X-axis: Feature value (degree connectivity of brain region)")
    print("- Y-axis: SHAP value (impact on ASD prediction)")
    print("- Shows how changing connectivity affects model predictions")
    print("- Non-linear relationships indicate complex interactions")


def plot_importance_vs_difference(
    importance_df, output_path="../outputs/importance_vs_difference.png"
):
    """
    Create scatter plot of SHAP importance vs connectivity differences.

    Args:
        importance_df: Feature importance dataframe
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        importance_df["Connectivity_Difference"],
        importance_df["SHAP_Importance"],
        c=importance_df["SHAP_Importance"],
        cmap="viridis",
        alpha=0.6,
        s=50,
    )

    # Highlight top 10 regions
    top_10 = importance_df.head(10)
    plt.scatter(
        top_10["Connectivity_Difference"],
        top_10["SHAP_Importance"],
        c="red",
        s=100,
        marker="x",
        linewidths=3,
        label="Top 10 Important Regions",
    )

    plt.xlabel("Connectivity Difference (ASD - TDC)", fontsize=12)
    plt.ylabel("SHAP Importance", fontsize=12)
    plt.title(
        "Relationship between Feature Importance and Group Differences", fontsize=14
    )
    plt.colorbar(scatter, label="SHAP Importance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Analysis Insights:")
    print("- Positive x-axis: Regions with higher connectivity in ASD")
    print("- Negative x-axis: Regions with lower connectivity in ASD")
    print("- Y-axis: How important each region is for model predictions")
    print("- Red X marks: Most discriminative regions for ASD classification")


def plot_stacked_force_plot(
    explainer,
    shap_values_train,
    X_train,
    feature_names,
    max_samples=820,
    output_path="../outputs/shap_stacked_force_plot.html",
):
    """
    Create stacked force plot for training data using top 10 most important features.
    Uses hierarchical agglomerative clustering to order instances.

    Args:
        explainer: SHAP explainer
        shap_values_train: SHAP values for training set
        X_train: Training features
        feature_names: List of feature names
        max_samples: Maximum number of samples to include (for performance)
        output_path: Path to save the HTML plot
    """
    print("Creating stacked force plot with top 10 features...")

    # Handle 3D SHAP values array
    if shap_values_train.ndim == 3:
        shap_values_for_plot = shap_values_train[:, :, 1]  # ASD class
        expected_value = explainer.expected_value[1]
    else:
        shap_values_for_plot = (
            shap_values_train[1]
            if isinstance(shap_values_train, list)
            else shap_values_train
        )
        expected_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, list)
            else explainer.expected_value
        )

    # Calculate feature importance (mean absolute SHAP values)
    feature_importance = np.mean(np.abs(shap_values_for_plot), axis=0)

    # Get top 10 most important features
    top_10_indices = np.argsort(feature_importance)[-10:]
    top_10_features = [feature_names[i] for i in top_10_indices]

    print(f"Top 10 most important features:")
    for i, (idx, name) in enumerate(zip(top_10_indices, top_10_features)):
        print(f"{i+1:2d}. {name} (importance: {feature_importance[idx]:.4f})")

    # Limit number of samples for performance
    n_samples = min(max_samples, shap_values_for_plot.shape[0])
    sample_indices = np.random.choice(
        shap_values_for_plot.shape[0], n_samples, replace=False
    )

    # Extract data for top 10 features and selected samples
    shap_subset = shap_values_for_plot[np.ix_(sample_indices, top_10_indices)]
    X_subset = X_train[np.ix_(sample_indices, top_10_indices)]

    print(
        f"\nCreating stacked force plot for {n_samples} samples and {len(top_10_features)} features..."
    )
    print("This will use hierarchical clustering to order instances by similarity.")

    # Create stacked force plot
    # SHAP automatically applies hierarchical clustering when multiple samples are provided
    force_plot = shap.force_plot(
        expected_value, shap_subset, X_subset, feature_names=top_10_features, show=False
    )

    # Save as HTML file
    shap.save_html(output_path, force_plot)
    print(f"Stacked force plot saved to: {output_path}")

    # Display the plot (if in Jupyter notebook)
    try:
        from IPython.display import display

        display(force_plot)
    except ImportError:
        print(
            "Note: To view the interactive plot, open the HTML file in a web browser."
        )

    print("\nStacked Force Plot Interpretation:")
    print("- Each row represents one training sample")
    print(
        "- Samples are ordered using hierarchical clustering (similar patterns grouped together)"
    )
    print("- Red regions: Features pushing prediction towards ASD")
    print("- Blue regions: Features pushing prediction towards TDC")
    print("- Width of colored regions: Magnitude of feature impact")
    print("- Only top 10 most discriminative brain regions are shown")
    print(
        "- Clustering reveals subgroups of patients with similar connectivity patterns"
    )

    return top_10_indices, top_10_features


def create_all_visualizations(
    explainer,
    shap_values_train,
    shap_values_test,
    X_train,
    X_test,
    y_test,
    y_pred,
    feature_names,
    importance_df,
    top_features_idx,
):
    """
    Create all SHAP visualizations in sequence.

    Args:
        explainer: SHAP explainer
        shap_values_train, shap_values_test: SHAP values
        X_train, X_test: Feature matrices
        y_test, y_pred: Test labels and predictions
        feature_names: List of feature names
        importance_df: Feature importance dataframe
        top_features_idx: Indices of top features
    """
    print("=== Creating SHAP Visualizations ===")

    # Global plots
    print("\n1. Creating global SHAP summary plot...")
    plot_global_shap_summary(shap_values_train, X_train, feature_names)

    print("\n2. Creating SHAP beeswarm plot...")
    plot_shap_beeswarm(shap_values_train, X_train, feature_names)

    print("\n3. Creating feature importance bar plot...")
    plot_feature_importance_bar(shap_values_train, X_train, feature_names)

    # Stacked force plot with top 10 features
    print("\n4. Creating stacked force plot with top 10 features...")
    top_10_indices, top_10_features = plot_stacked_force_plot(
        explainer, shap_values_train, X_train, feature_names
    )

    # Individual prediction plots
    print("\n5. Creating individual prediction explanations...")
    case_asd = plot_individual_waterfall(
        explainer, shap_values_test, X_test, y_test, y_pred, feature_names, "ASD"
    )
    case_tdc = plot_individual_waterfall(
        explainer, shap_values_test, X_test, y_test, y_pred, feature_names, "TDC"
    )

    print("\n6. Creating force plots...")
    plot_force_plot(explainer, shap_values_test, X_test, case_asd, feature_names, "ASD")
    plot_force_plot(explainer, shap_values_test, X_test, case_tdc, feature_names, "TDC")

    # Dependence analysis
    print("\n7. Creating dependence plots...")
    plot_dependence_analysis(
        shap_values_train, X_train, feature_names, top_features_idx
    )

    # Importance vs difference plot
    print("\n8. Creating importance vs difference plot...")
    plot_importance_vs_difference(importance_df)

    print("\n=== All Visualizations Complete ===")
    print("\nGenerated files:")
    print("- shap_summary_plot.png")
    print("- shap_beeswarm_plot.png")
    print("- shap_feature_importance_bar.png")
    print("- shap_stacked_force_plot.html")
    print("- shap_waterfall_asd_case.png")
    print("- shap_waterfall_tdc_case.png")
    print("- shap_force_asd_case.png")
    print("- shap_force_tdc_case.png")
    print("- shap_dependence_plots.png")
    print("- importance_vs_difference.png")

    return top_10_indices, top_10_features
