from typing import List, Optional

import pandas as pd

from .parsing_utilities import GENE_LABEL_FORMAT, split_gene_label_str


def prepare_targets(
    df: pd.DataFrame,
    top_variance_filter: Optional[int],
    gene_filter: Optional[List[str]],
) -> pd.DataFrame:
    if gene_filter is not None:
        if df.columns.str.match(GENE_LABEL_FORMAT).all():
            # ex: A1BG (1)
            gene_filter = [
                c for c in df.columns if split_gene_label_str(c)[0] in gene_filter
            ]
            if len(gene_filter) == 0:
                raise ValueError("No matching genes found")
        df = df.filter(items=gene_filter, axis="columns")

    if top_variance_filter is not None:
        df = df.filter(
            items=df.var(axis=0)
            .sort_values(ascending=False)[:top_variance_filter]
            .index,
            axis="columns",
        )

    df.index.name = "Row.name"

    return df
