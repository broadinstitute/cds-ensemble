import pandas as pd
import pytest


@pytest.fixture
def data_matrices():
    return {
        filename: pd.read_csv(f"sample_files/data/{filename}.csv")
        for filename in [
            "confounders",
            "full_matrix",
            "full_table",
            "partial_matrix",
            "partial_table",
        ]
    }
