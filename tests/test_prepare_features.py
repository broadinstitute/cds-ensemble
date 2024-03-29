import os

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from cds_ensemble.prepare_features import (
    normalize_col,
    standardize_col_name,
    prepare_numeric_features,
    prepare_categorical_features,
    prepare_single_dataset_features,
    prepare_universal_feature_set,
    get_valid_samples_for_model,
    format_related,
)
from cds_ensemble.data_models import ModelConfig, FeatureInfo

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
            "gene_symbol": None,
            "entrez_id": None,
        }
    )
    assert_frame_equal(renamed_df, expected_df)
    # NaN vs None
    assert_frame_equal(feature_metadata, expected_feature_metadata, check_dtype=False)

    df = pd.DataFrame(
        columns=[
            "A1BG (1)",
            "A1CF (29974)",
            "A2M (2)",
            "A2ML1 (144568)",
            "A3GALT2 (127550)",
        ]
    )
    expected_df = pd.DataFrame(
        columns=[
            "A1BG_(1)_a_neat_dataset",
            "A1CF_(29974)_a_neat_dataset",
            "A2M_(2)_a_neat_dataset",
            "A2ML1_(144568)_a_neat_dataset",
            "A3GALT2_(127550)_a_neat_dataset",
        ]
    )
    expected_feature_metadata = pd.DataFrame(
        {
            "feature_id": expected_df.columns,
            "feature_name": df.columns,
            "dataset": dataset_name,
            "gene_symbol": ["A1BG", "A1CF", "A2M", "A2ML1", "A3GALT2"],
            "entrez_id": [1, 29974, 2, 144568, 127550],
        }
    )


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
        target_samples, feature_infos
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
        "confounders", os.path.join(TEST_DATA_DIR, "confounders.csv"), normalize=False
    )

    feature_infos.append(confounders_feature_info)

    universal_feature_set, feature_metadata = prepare_universal_feature_set(
        target_samples, feature_infos
    )

    assert universal_feature_set.columns.size == sum(
        feature_info.data.columns.size for feature_info in feature_infos
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


def test_get_valid_samples_for_model(prepared_universal_feature_set):
    (
        feature_infos,
        universal_feature_set,
        feature_metadata,
    ) = prepared_universal_feature_set

    model_config = ModelConfig(
        name="Unbiased",
        features=["full_matrix", "partial_matrix", "full_table", "partial_table"],
        required_features=["full_matrix", "confounders"],
        related_dataset=None,
        relation="All",
        exempt=None,
    )
    valid_samples = get_valid_samples_for_model(
        model_config, feature_infos, universal_feature_set
    )
    assert np.sum(valid_samples) == universal_feature_set.index.size

    model_config = ModelConfig(
        name="Unbiased",
        features=["partial_matrix", "partial_table"],
        required_features=["partial_table"],
        related_dataset=None,
        relation="All",
        exempt=None,
    )
    valid_samples = get_valid_samples_for_model(
        model_config, feature_infos, universal_feature_set
    )
    assert np.sum(valid_samples) < universal_feature_set.index.size


def test_format_related():
    related_table = format_related(
        {
            "Related": ModelConfig(
                name="Related",
                features=[
                    "full_matrix",
                    "partial_matrix",
                    "full_table",
                    "partial_table",
                ],
                required_features=["full_matrix", "confounders"],
                related_dataset="related",
                relation="MatchRelated",
                exempt=None,
            )
        },
        [FeatureInfo("related", os.path.join(TEST_DATA_DIR, "related.csv"))],
    )
    expected = pd.DataFrame(
        {
            "target_gene_symbol": ["TARGET-0", "TARGET-1", "TARGET-2", "TARGET-3"],
            "target_entrez_id": [0, 1, 2, 3],
            "partner_gene_symbol": ["TARGET-1", "TARGET-0", "TARGET-3", "TARGET-2"],
            "partner_entrez_id": [1, 0, 3, 2],
        }
    )
    assert_frame_equal(related_table, expected, check_dtype=False)
