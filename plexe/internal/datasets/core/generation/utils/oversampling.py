"""
This module contains utility functions for oversampling datasets.
"""

import pandas as pd

from imblearn.over_sampling import SMOTE


def oversample_with_smote(dataset: pd.DataFrame, target_column: str, n_records_to_generate) -> pd.DataFrame:
    """
    Oversample the dataset using SMOTE.
    :param dataset: dataset to oversample
    :param target_column: name of the column containing the target variable
    :param n_records_to_generate: number of records to generate
    :return: oversampled dataset with original class proportions
    """
    # we sample in the same proportion of classes as the original dataset
    smote = SMOTE(
        sampling_strategy={
            target: int(n_samples / dataset[target_column].count() * n_records_to_generate)
            for target, n_samples in dataset[target_column].value_counts().items()
        }
    )

    x_resampled, y_resampled = smote.fit_resample(X=dataset.drop(columns=[target_column]), y=dataset[target_column])

    # Create a new dataframe with the resampled data
    resampled_df = pd.DataFrame(x_resampled, columns=dataset.columns)
    resampled_df[target_column] = y_resampled

    return resampled_df
