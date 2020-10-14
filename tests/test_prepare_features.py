import os

import numpy as np
import pandas as pd
import pytest

from cds_ensemble.prepare_features import (
    normalize_col,
    standardize_col_name,
    prepare_numeric_features,
    prepare_categorical_features,
    prepare_single_dataset_features,
    prepare_universal_feature_set,
    subset_by_model_config,
)
from cds_ensemble.models import ModelConfig, FeatureInfo

from .conftest import TEST_DATA_DIR, parse_feature_df


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
    feature_df = parse_feature_df("full_matrix.csv")
    processed_df, feature_metadata = prepare_numeric_features(
        feature_df, "a neat dataset", True
    )
    for i, row in feature_metadata.iterrows():
        original_col = feature_df[row["feature_name"]]
        processed_col = processed_df[row["feature_id"]]
        assert processed_col.equals(
            (original_col - original_col.mean()) / original_col.std()
        )


def test_prepare_categorical_features():
    feature_df = parse_feature_df("full_table.csv")
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


def test_prepare_single_dataset_features():
    feature_df = parse_feature_df(os.path.join(TEST_DATA_DIR, "full_table.csv"))
    #     df: pd.DataFrame, dataset_name: str, normalize: bool = True
    processed_df, feature_metadata = prepare_single_dataset_features(
        feature_df, "a dataset"
    )
    assert processed_df.shape[0] == feature_df.shape[0]
    # There's one categorical column with 3 values
    assert processed_df.shape[1] == feature_df.shape[1] + 2


def test_prepare_universal_feature_set():
    feature_files = {
        "full_matrix": os.path.join(TEST_DATA_DIR, "full_matrix.csv"),
        "partial_matrix": os.path.join(TEST_DATA_DIR, "partial_matrix.csv"),
        "full_table": os.path.join(TEST_DATA_DIR, "full_table.csv"),
        "partial_table": os.path.join(TEST_DATA_DIR, "partial_table.csv"),
    }

    feature_infos = [
        FeatureInfo(dataset_name, file_name)
        for dataset_name, file_name in feature_files.items()
    ]

    target_samples = pd.read_csv(
        os.path.join(TEST_DATA_DIR, "target_matrix.csv"), index_col=0
    ).index

    universal_feature_set, feature_metadata = prepare_universal_feature_set(
        target_samples, feature_infos, None
    )

    assert universal_feature_set.notnull().all(axis=None)
    assert universal_feature_set.index.size == target_samples.size
    assert universal_feature_set.columns.size == sum(
        feature_info.data.columns.size for feature_info in feature_infos
    )

    confounders = pd.read_csv(
        os.path.join(TEST_DATA_DIR, "confounders.csv"), index_col=0
    )

    confounders_feature_info = FeatureInfo(
        "confounders", os.path.join(TEST_DATA_DIR, "confounders.csv")
    )

    universal_feature_set, feature_metadata = prepare_universal_feature_set(
        target_samples, feature_infos, confounders_feature_info
    )

    assert (
        universal_feature_set.columns.size
        == sum(feature_info.data.columns.size for feature_info in feature_infos)
        + confounders_feature_info.data.columns.size
    )

    # Confounders should not be scaled
    assert np.allclose(
        confounders.select_dtypes("number").values,
        confounders_feature_info.data[
            feature_metadata[
                (feature_metadata["dataset"] == "confounders")
                & feature_metadata["feature_name"].isin(
                    confounders.select_dtypes("number").columns
                )
            ]["feature_id"]
        ].values,
    )


def test_subset_by_model_config():
    feature_files = {
        "full_matrix": os.path.join(TEST_DATA_DIR, "full_matrix.csv"),
        "partial_matrix": os.path.join(TEST_DATA_DIR, "partial_matrix.csv"),
        "full_table": os.path.join(TEST_DATA_DIR, "full_table.csv"),
        "partial_table": os.path.join(TEST_DATA_DIR, "partial_table.csv"),
    }

    feature_infos = [
        FeatureInfo(dataset_name, file_name)
        for dataset_name, file_name in feature_files.items()
    ]

    target_samples = pd.read_csv(
        os.path.join(TEST_DATA_DIR, "target_matrix.csv"), index_col=0
    ).index

    confounders_feature_info = FeatureInfo(
        "confounders", os.path.join(TEST_DATA_DIR, "confounders.csv")
    )

    universal_feature_set, feature_metadata = prepare_universal_feature_set(
        target_samples, feature_infos, confounders_feature_info
    )

    model_config = ModelConfig(
        "Unbiased",
        ["full_matrix", "partial_matrix", "full_table", "partial_table"],
        ["full_matrix", "confounders"],
        "All",
    )
    all_feature_set, all_feature_metadata = subset_by_model_config(
        model_config,
        feature_infos,
        "confounders",
        universal_feature_set,
        feature_metadata,
    )
    assert all_feature_set.equals(universal_feature_set)

    model_config = ModelConfig(
        "Unbiased", ["partial_matrix", "partial_table"], ["partial_table"], "All"
    )
    all_feature_set, all_feature_metadata = subset_by_model_config(
        model_config,
        feature_infos,
        "confounders",
        universal_feature_set,
        feature_metadata,
    )
    assert all_feature_set.index.size < universal_feature_set.index.size
