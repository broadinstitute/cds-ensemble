import os

import pytest

from cds_ensemble.prepare_features import format_related
from cds_ensemble.run_ensemble import filter_run_ensemble_inputs, run_model
from cds_ensemble.parsing_utilities import read_dataframe, read_model_config
from cds_ensemble.data_models import ModelConfig, FeatureInfo
from .conftest import TEST_DATA_DIR, TEST_CONFIG_DIR


@pytest.mark.parametrize(
    "valid_samples,feature_subset,target_range,targets,expected_X,expected_Y,expected_start_col,expected_end_col",
    [
        pytest.param(
            None,
            None,
            None,
            None,
            lambda X: X,
            lambda Y: Y,
            0,
            lambda Y: Y.shape[1],
            id="no filtering",
        ),
        pytest.param(
            ["sample-0", "sample-2", "sample-nonexistent"],
            None,
            None,
            None,
            lambda X: X.loc[["sample-0", "sample-2"]],
            lambda Y: Y.loc[["sample-0", "sample-2"]],
            0,
            lambda Y: Y.shape[1],
            id="filter valid samples",
        ),
        pytest.param(
            None,
            [
                "TARGET_1_(1)_full_matrix",
                "TARGET_2_(2)_full_matrix",
                "nonexistent_feature",
            ],
            None,
            None,
            lambda X: X[["TARGET_1_(1)_full_matrix", "TARGET_2_(2)_full_matrix"]],
            lambda Y: Y,
            0,
            lambda Y: Y.shape[1],
            id="filter feature subset",
        ),
        pytest.param(
            None,
            None,
            (1, 3),
            None,
            lambda X: X,
            lambda Y: Y.iloc[:, 1:3],
            1,
            lambda Y: 3,
            id="filter target range",
        ),
        pytest.param(
            None,
            None,
            None,
            ["TARGET-1 (1)", "TARGET-2 (2)", "fake_target"],
            lambda X: X,
            lambda Y: Y[["TARGET-1 (1)", "TARGET-2 (2)"]],
            0,
            lambda Y: Y.shape[1],
            id="filter targets",
        ),
    ],
)
def test_filter_run_ensemble_inputs(
    prepared_universal_feature_set,
    valid_samples,
    feature_subset,
    target_range,
    targets,
    expected_X,
    expected_Y,
    expected_start_col,
    expected_end_col,
):
    _, X, _ = prepared_universal_feature_set
    Y = read_dataframe(os.path.join(TEST_DATA_DIR, "target_matrix.csv"))
    filtered_X, filtered_Y, start_col, end_col = filter_run_ensemble_inputs(
        X,
        Y,
        valid_samples=valid_samples,
        feature_subset=feature_subset,
        target_range=target_range,
        targets=targets,
    )

    assert filtered_X.equals(expected_X(X))
    assert filtered_Y.equals(expected_Y(Y))
    assert start_col == expected_start_col
    assert end_col == expected_end_col(Y)


def test_run_model(prepared_universal_feature_set):
    # This test only checks that things run.
    (
        feature_infos,
        universal_feature_set,
        feature_metadata,
    ) = prepared_universal_feature_set
    Y = read_dataframe(os.path.join(TEST_DATA_DIR, "target_matrix.csv"))
    model_config = ModelConfig(
        name="Unbiased",
        features=["full_matrix", "partial_matrix", "full_table", "partial_table"],
        required_features=["full_matrix", "confounders"],
        related_dataset=None,
        relation="All",
        exempt=None,
    )

    ensemble = run_model(universal_feature_set, Y, model_config, 3)
    df = ensemble.format_results()
    assert df.notnull().all(axis=None)

    # Test related features
    model_config = ModelConfig(
        name="Related",
        features=["full_matrix", "partial_matrix", "full_table", "partial_table"],
        required_features=["full_matrix", "confounders"],
        related_dataset="related",
        relation="MatchRelated",
        exempt=None,
    )
    relation_table = format_related(
        {"Related": model_config},
        [FeatureInfo("related", os.path.join(TEST_DATA_DIR, "related.csv"))],
    )

    ensemble = run_model(
        universal_feature_set,
        Y,
        model_config,
        3,
        relation_table=relation_table,
        feature_metadata=feature_metadata,
    )
    df = ensemble.format_results()

    # There are only 4 features used, and one target has no related features
    assert (
        df.iloc[:4][
            [
                "gene",
                "model",
                "score0",
                "score1",
                "score2",
                "best",
                "feature0",
                "feature0_importance",
                "feature1",
                "feature1_importance",
                "feature2",
                "feature2_importance",
                "feature3",
                "feature3_importance",
            ]
        ]
        .notnull()
        .all(axis=None)
    )
    assert (
        df.iloc[4][
            [
                "gene",
                "model",
                "score0",
                "score1",
                "score2",
                "best",
                "feature0",
                "feature0_importance",
            ]
        ]
        .notnull()
        .all()
    )
