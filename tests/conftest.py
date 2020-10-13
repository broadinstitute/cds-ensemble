import os

import pandas as pd
import pytest

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "sample_files/data"
)


def parse_feature_df(file_name: str):
    """Returns the sample file called [file_name] as a Pandas DataFrame."""
    return pd.read_csv(os.path.join(FIXTURE_DIR, file_name), index_col=0)
