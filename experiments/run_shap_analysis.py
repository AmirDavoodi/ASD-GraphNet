#!/usr/bin/env python3
"""
Complete SHAP Analysis Pipeline for ASD-GraphNet Model Interpretability
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import ast

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from shap_data_loader import load_shap_data
from shap_model_trainer import train_and_evaluate_model
from shap_analyzer import complete_shap_analysis, get_top_features
from shap_visualizer import create_all_visualizations

# Set random seed
np.random.seed(42)
plt.style.use("default")


def main():
    """
    Main function to run the complete SHAP analysis pipeline.
    """
    print("=== ASD-GraphNet SHAP Analysis Pipeline ===")
    print("Addressing reviewer comments on model interpretability and explainable AI\n")

    # Step 1: Load and prepare data using existing train/test split
    print("Step 1: Loading data with existing stratified train/test split...")
    X_train, X_test, y_train, y_test, feature_names, site_analysis = load_shap_data()

    print(f"\nData loaded successfully!")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Number of features (brain regions): {X_train.shape[1]}")

    # Display site stratification summary
    print(f"\nSite stratification summary:")
    print(f"Number of sites: {len(site_analysis)}")
    print(
        f"Test percentage across sites - Mean: {site_analysis['Test_Percent'].mean():.2f}%, Std: {site_analysis['Test_Percent'].std():.2f}%"
    )

    # Save site analysis
    site_analysis.to_csv("../outputs/site_stratification_analysis.csv", index=False)
    print("Site analysis saved to: ../outputs/site_stratification_analysis.csv")

    # Step 2: Train Random Forest model
    print("\nStep 2: Training Random Forest model...")
    model, y_pred, accuracy = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    print(f"Model training completed with accuracy: {accuracy:.4f}")

    # Step 3: Perform SHAP analysis
    print("\nStep 3: Performing SHAP analysis for model interpretability...")
    explainer, shap_values_train, shap_values_test, importance_df, analysis_stats = (
        complete_shap_analysis(model, X_train, X_test, y_train, feature_names)
    )

    print("SHAP analysis completed!")
    print(f"Most important region: {analysis_stats['most_important_region']}")
    print(f"Regions with higher ASD connectivity: {analysis_stats['hyper_regions']}")
    print(f"Regions with lower ASD connectivity: {analysis_stats['hypo_regions']}")

    # Step 4: Get top features
    print("\nStep 4: Identifying top discriminative features...")
    top_features_df, top_features_idx = get_top_features(importance_df, n_features=10)
    print("Top 10 most discriminative brain regions identified!")

    # Step 5: Generate visualizations
    print("\nStep 5: Generating comprehensive SHAP visualizations...")
    create_all_visualizations(
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
    )

    # Step 6: Summary for paper revision
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER REVISION")
    print("=" * 60)
    print(f"Model Performance: {accuracy:.1%} accuracy on test set")
    print(
        f"Feature Interpretability: Top {len(top_features_df)} discriminative brain regions identified"
    )
    print(f"Biological Significance:")
    print(
        f"  - {analysis_stats['hyper_regions']} regions show increased connectivity in ASD"
    )
    print(
        f"  - {analysis_stats['hypo_regions']} regions show decreased connectivity in ASD"
    )
    print(f"  - Most important region: {analysis_stats['most_important_region']}")
    print(f"\nGenerated outputs:")
    print(
        f"  - Feature importance analysis: ../outputs/comprehensive_feature_importance.csv"
    )
    print(
        f"  - Site stratification analysis: ../outputs/site_stratification_analysis.csv"
    )
    print(f"  - Multiple visualization files for paper figures")
    print(f"\nAddressing Reviewer Comments:")
    print(
        f"  - Interpretability (Comment 2): ✓ SHAP values provide biological interpretation"
    )
    print(
        f"  - Explainable AI (Comment 4): ✓ Multiple visualization types for global/individual explanations"
    )
    print(
        f"  - Clinical Applicability: ✓ Individual case explanations for diagnostic utility"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
