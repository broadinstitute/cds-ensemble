import os

import numpy as np
import pandas as pd
import pytest

from cds_ensemble.prepare_features import (
    normalize_col,
    standardize_col_name,
    prepare_numeric_features,
    prepare_categorical_features,
)

from .conftest import FIXTURE_DIR


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


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "full_matrix.csv"))
def test_prepare_numeric_features(feature_df: pd.DataFrame):
    processed_df, feature_metadata = prepare_numeric_features(
        feature_df, "a neat dataset", True
    )
    for i, row in feature_metadata.iterrows():
        original_col = feature_df[row["feature_name"]]
        processed_col = processed_df[row["feature_id"]]
        assert processed_col.equals(
            (original_col - original_col.mean()) / original_col.std()
        )


@pytest.mark.datafiles(os.path.join(FIXTURE_DIR, "full_table.csv"))
def test_prepare_categorical_features(feature_df: pd.DataFrame):
    categorical_df = feature_df.select_dtypes(exclude="number")
    processed_df, feature_metadata = prepare_categorical_features(
        categorical_df, "a neat dataset"
    )
    assert processed_df.shape == (
        categorical_df.shape[0],
        categorical_df["categorical feature"].unique().size,
    )
    assert (feature_metadata["feature_name"] == "categorical feature").all()
    assert processed_df.isin([1, 0, np.nan]).all(axis=None)
