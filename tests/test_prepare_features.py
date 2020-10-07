import numpy as np
import pandas as pd
import pytest

from cds_ensemble.prepare_features import (
    normalize_col,
    standardize_col_name,
    prepare_numeric_features,
    prepare_categorical_features,
)


@pytest.mark.parametrize(
    "col,expected",
    [
        pytest.param(
            pd.Series([0.0, 0.0, 1.0, 0.0]),
            pd.Series([0.0, 0.0, 1.0, 0.0]),
            id="one-hot encoded",
        ),
        pytest.param(
            pd.Series([0.0, 0.0, 1.0, np.nan]),
            pd.Series([0.0, 0.0, 1.0, np.nan]),
            id="one-hot encoded with nan",
        ),
        pytest.param(
            pd.Series([2, 4, 4, 4, 5, 5, 5, 7, 9]),
            pd.Series([-1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 2.0]),
            id="numeric",
        ),
        pytest.param(
            pd.Series([np.nan, 2, 4, 4, 4, 5, 5, 5, 7, 9]),
            pd.Series([np.nan, -1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 2.0]),
            id="numeric with nan",
        ),
    ],
)
def test_normalize_col(col: pd.Series, expected: pd.Series):
    assert normalize_col(col).equals(expected)


def test_standardize_col_name():
    df = pd.DataFrame(columns=["column 1", "column-2", "column  3", "column4"])
    dataset_name = "a neat dataset"

    renamed_df, feature_metadata = standardize_col_name(df, dataset_name)
    expected_df = pd.DataFrame(
        columns=[
            "column_1_a_neat_dataset",
            "column_2_a_neat_dataset",
            "column__3_a_neat_dataset",
            "column4_a_neat_dataset",
        ]
    )
    expected_feature_metadata = pd.DataFrame(
        {
            "feature_id": expected_df.columns,
            "feature_name": df.columns,
            "dataset": dataset_name,
        }
    )
    assert renamed_df.equals(expected_df)
    assert feature_metadata.equals(expected_feature_metadata)


# TODO: check that null values work

# TODO: check that confounders does not get normalized


def test_prepare_numeric_features():
    pass


def test_prepare_categorical_features():
    pass
