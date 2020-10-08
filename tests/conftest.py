import os

import pandas as pd
import pytest

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "sample_files/data"
)

ALL_FILES = pytest.mark.datafiles(
    # os.path.join(FIXTURE_DIR, "confounders.csv"),
    os.path.join(FIXTURE_DIR, "full_matrix.csv"),
    os.path.join(FIXTURE_DIR, "partial_matrix.csv"),
    os.path.join(FIXTURE_DIR, "full_table.csv"),
    os.path.join(FIXTURE_DIR, "partial_table.csv"),
)


def parse_all_feature_dfs(datafiles):
    return {
        os.path.splitext(os.path.basename(file_name))[0]: pd.read_csv(
            file_name, index_col=0
        )
        for file_name in datafiles.listdir()
    }


def parse_feature_df(datafiles):
    """Returns the first file in datafiles (which is a pytest-datafile fixture) as a
    Pandas DataFrame. This must be used in conjunction with a @pytest.mark.datafile
    decorator.
    """
    return pd.read_csv(datafiles.listdir()[0], index_col=0)
