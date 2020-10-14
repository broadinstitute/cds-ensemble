import os

import pandas as pd
import pytest
from click.testing import CliRunner

from cds_ensemble.__main__ import main
from .conftest import TEST_DATA_DIR, TEST_CONFIG_DIR


def test_prepare_y(tmpdir):
    input_path = os.path.join(TEST_DATA_DIR, "target_matrix.csv")

    output_path = tmpdir.join("output.ftr")

    runner = CliRunner()
    result = runner.invoke(
        main, ["prepare-y", "--input", input_path, "--output", output_path]
    )

    assert result.exit_code == 0
    df = pd.read_feather(output_path).set_index("Row.name")
    expected = pd.read_csv(input_path, index_col=0)
    expected.index.name = "Row.name"
    assert df.equals(expected)


def test_prepare_x(tmpdir):
    feature_info = pd.DataFrame(
        [
            (dataset, os.path.join(TEST_DATA_DIR, dataset + ".csv"))
            for dataset in [
                "full_matrix",
                "full_table",
                "partial_matrix",
                "partial_table",
            ]
        ],
        columns=["dataset", "filename"],
    )
    feature_info_path = tmpdir.join("feature_info.csv")
    feature_info.to_csv(feature_info_path, index=False)

    targets_path = os.path.join(TEST_DATA_DIR, "target_matrix.csv")
    output_path = tmpdir.join("output")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "prepare-x",
            "--targets",
            targets_path,
            "--model-config",
            os.path.join(TEST_CONFIG_DIR, "model_def.yml"),
            "--feature-info",
            feature_info_path,
            "--output",
            output_path,
            "--output-format",
            ".csv",
        ],
    )
    assert result.exit_code == 0

    targets = pd.read_csv(targets_path, index_col=0)
    df = pd.read_csv(output_path + "-Unbiased.csv", index_col=0)
    assert targets.index.size == df.index.size
