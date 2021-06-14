import os

import pandas as pd
import pytest

from cds_ensemble.prepare_features import prepare_universal_feature_set
from cds_ensemble.data_models import FeatureInfo

TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "sample_files/data"
)
TEST_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "sample_files/config"
)


def parse_feature_df(file_name: str):
    """Returns the sample file called [file_name] as a Pandas DataFrame."""
    return pd.read_csv(os.path.join(TEST_DATA_DIR, file_name), index_col=0)


@pytest.fixture
def feature_info_file(tmpdir):
    feature_info = pd.DataFrame(
        [
            (dataset, os.path.join(TEST_DATA_DIR, dataset + ".csv"))
            for dataset in [
                "full_matrix",
                "full_table",
                "partial_matrix",
                "partial_table",
                "related",
            ]
        ],
        columns=["dataset", "filename"],
    )
    feature_info_path = tmpdir.join("feature_info.csv")
    feature_info.to_csv(feature_info_path, index=False)
    return feature_info_path


@pytest.fixture
def prepared_universal_feature_set():
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
    confounders_feature_info = FeatureInfo(
        "confounders", os.path.join(TEST_DATA_DIR, "confounders.csv"), normalize=False
    )
    feature_infos.append(confounders_feature_info)

    target_samples = pd.read_csv(
        os.path.join(TEST_DATA_DIR, "target_matrix.csv"), index_col=0
    ).index

    universal_feature_set, feature_metadata = prepare_universal_feature_set(
        target_samples, feature_infos
    )
    return feature_infos, universal_feature_set, feature_metadata
