import os

import pandas as pd
import pytest
from click.testing import CliRunner

from cds_ensemble.__main__ import main, prepare_x, prepare_y, fit_models
from .conftest import FIXTURE_DIR


def test_prepare_y(tmpdir):
    input_path = os.path.join(FIXTURE_DIR, "target_matrix.csv")

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
    pass
