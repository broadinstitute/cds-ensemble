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


def test_prepare_x(tmpdir, feature_info_file):
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
            feature_info_file,
            "--output",
            output_path,
            "--output-format",
            ".csv",
        ],
    )
    assert result.exit_code == 0

    targets = pd.read_csv(targets_path, index_col=0)
    df = pd.read_csv(output_path + ".csv", index_col=0)
    assert targets.index.size == df.index.size


def test_fit_model(tmpdir, feature_info_file):
    runner = CliRunner()

    targets_path = os.path.join(TEST_DATA_DIR, "target_matrix.csv")

    model_def_file = os.path.join(TEST_CONFIG_DIR, "model_def.yml")
    # Get X
    output_path = tmpdir.join("output")
    runner.invoke(
        main,
        [
            "prepare-x",
            "--targets",
            targets_path,
            "--model-config",
            model_def_file,
            "--feature-info",
            feature_info_file,
            "--output",
            output_path,
            "--output-format",
            ".csv",
        ],
    )

    result = runner.invoke(
        main,
        [
            "fit-model",
            "--x",
            output_path + ".csv",
            "--y",
            targets_path,
            "--model-config",
            model_def_file,
            "--model",
            "Unbiased",
            "--output-dir",
            tmpdir,
        ],
    )
    assert result.exit_code == 0

    features = pd.read_csv(tmpdir.join("Unbiased_0_5_features.csv"))
    assert features.columns.to_list() == [
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
        "feature4",
        "feature4_importance",
        "feature5",
        "feature5_importance",
        "feature6",
        "feature6_importance",
        "feature7",
        "feature7_importance",
        "feature8",
        "feature8_importance",
        "feature9",
        "feature9_importance",
    ]

    predictions = pd.read_csv(tmpdir.join("Unbiased_0_5_predictions.csv"), index_col=0)
    target_df = pd.read_csv(targets_path, index_col=0)
    assert predictions.shape == target_df.shape
    assert set(predictions.index) == set(target_df.index)
    assert set(predictions.columns) == set(target_df.columns)
