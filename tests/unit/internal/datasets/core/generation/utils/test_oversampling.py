import pandas as pd
import numpy as np

from plexe.internal.datasets.core.generation.utils.oversampling import oversample_with_smote


class TestSMOTEOversampling:
    def test_smote_maintains_class_proportions(self):
        """Test that SMOTE maintains relative class frequencies when oversampling"""
        df = pd.DataFrame(
            {"feature1": range(20), "feature2": range(20), "target": [0] * 14 + [1] * 6}  # 70% class 0, 30% class 1
        )

        n_samples = len(df) * 2
        oversampled_df = oversample_with_smote(df, "target", n_samples)

        original_proportions = df["target"].value_counts(normalize=True)
        new_proportions = oversampled_df["target"].value_counts(normalize=True)

        pd.testing.assert_series_equal(
            original_proportions.sort_index(), new_proportions.sort_index(), check_exact=False, rtol=0.1
        )
        assert len(oversampled_df) == n_samples

    def test_smote_with_multiple_classes(self):
        """Test SMOTE works correctly with more than two classes"""
        df = pd.DataFrame(
            {
                "feature1": range(24),
                "feature2": range(24),
                "target": [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6,  # 4 classes, 6 samples each
            }
        )

        n_samples = 48  # Double the size
        oversampled_df = oversample_with_smote(df, "target", n_samples)

        original_proportions = df["target"].value_counts(normalize=True)
        new_proportions = oversampled_df["target"].value_counts(normalize=True)

        pd.testing.assert_series_equal(
            original_proportions.sort_index(), new_proportions.sort_index(), check_exact=False, rtol=0.1
        )
        assert len(oversampled_df) == n_samples

    def test_smote_with_extreme_imbalance(self):
        """Test SMOTE handles extremely imbalanced datasets"""
        df = pd.DataFrame(
            {
                "feature1": range(106),
                "feature2": range(106),
                "target": [0] * 100 + [1] * 6,  # ~94% vs ~6%, with enough samples for SMOTE
            }
        )

        n_samples = 200
        oversampled_df = oversample_with_smote(df, "target", n_samples)

        original_proportions = df["target"].value_counts(normalize=True)
        new_proportions = oversampled_df["target"].value_counts(normalize=True)

        pd.testing.assert_series_equal(
            original_proportions.sort_index(), new_proportions.sort_index(), check_exact=False, rtol=0.1
        )
        # Allow for ±1 sample difference
        assert abs(len(oversampled_df) - n_samples) <= 1

    def test_smote_preserves_feature_distributions(self):
        """Test SMOTE generates synthetic samples that preserve feature distributions"""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 120),
                "feature2": np.random.uniform(-1, 1, 120),
                "target": [0] * 80 + [1] * 40,
            }
        )

        n_samples = 240  # Double the size
        oversampled_df = oversample_with_smote(df, "target", n_samples)

        for feature in ["feature1", "feature2"]:
            assert np.abs(df[feature].mean() - oversampled_df[feature].mean()) < 0.5
            assert np.abs(df[feature].std() - oversampled_df[feature].std()) < 0.5
