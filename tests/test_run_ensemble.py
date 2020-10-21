import os

from cds_ensemble.prepare_features import format_related
from cds_ensemble.run_ensemble import filter_run_ensemble_inputs, run_model
from cds_ensemble.parsing_utilities import read_dataframe, read_model_config
from cds_ensemble.data_models import ModelConfig, FeatureInfo
from .conftest import TEST_DATA_DIR, TEST_CONFIG_DIR


def test_filter_run_ensemble_inputs():
    pass


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
