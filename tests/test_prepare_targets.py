import numpy as np
import pandas as pd
import pytest

from cds_ensemble.prepare_targets import prepare_targets

TARGETS = pd.DataFrame(
    {
        "SOX10 (6663)": [0, 0.5, 0.5],
        "NRAS (4893)": [0.6, 0.6, 0.7],
        "BRAF (673)": [0.3, 0.4, 0.4],
    },
    index=["sample-1", "sample-2", "sample-3"],
)


@pytest.mark.parametrize(
    "top_variance_filter,gene_filter,expected",
    [
        pytest.param(None, None, TARGETS),
        pytest.param(
            1,
            None,
            pd.DataFrame(
                {"SOX10 (6663)": [0, 0.5, 0.5]},
                index=["sample-1", "sample-2", "sample-3"],
            ),
        ),
        pytest.param(
            None,
            ["NRAS"],
            pd.DataFrame(
                {"NRAS (4893)": [0.6, 0.6, 0.7]},
                index=["sample-1", "sample-2", "sample-3"],
            ),
        ),
    ],
)
def test_prepare_targets(top_variance_filter, gene_filter, expected):
    actual = prepare_targets(TARGETS, top_variance_filter, gene_filter)
    assert actual.equals(expected)
