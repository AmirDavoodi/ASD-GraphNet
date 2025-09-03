"""
Test script to verify the updated data loading function works correctly
"""

from shap_data_loader import load_shap_data

if __name__ == "__main__":
    print("Testing updated data loading with existing test column...")

    try:
        # Load data using the updated function
        X_train, X_test, y_train, y_test, feature_names, site_analysis = (
            load_shap_data()
        )

        print("\n=== Data Loading Summary ===")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Feature names (first 5): {feature_names[:5]}")

        print("\n=== Site Analysis Summary ===")
        print(f"Number of sites analyzed: {len(site_analysis)}")
        print(
            f"Test percentage across sites - Mean: {site_analysis['Test_Percent'].mean():.2f}%, Std: {site_analysis['Test_Percent'].std():.2f}%"
        )

        # Verify the proportions are close to 20% test / 80% train
        expected_test_percent = 20.0
        tolerance = 5.0  # Allow 5% deviation

        sites_within_tolerance = site_analysis[
            (site_analysis["Test_Percent"] >= expected_test_percent - tolerance)
            & (site_analysis["Test_Percent"] <= expected_test_percent + tolerance)
        ]

        print(
            f"\nSites within {tolerance}% of expected 20% test split: {len(sites_within_tolerance)}/{len(site_analysis)}"
        )

        if (
            len(sites_within_tolerance) >= len(site_analysis) * 0.8
        ):  # At least 80% of sites should be close
            print("✅ Stratification looks good - most sites follow ~20% test split")
        else:
            print("⚠️  Some sites may have uneven splits")

        print("\n=== Data loading test completed successfully! ===")

    except Exception as e:
        print(f"❌ Error during data loading: {e}")
        import traceback

        traceback.print_exc()
