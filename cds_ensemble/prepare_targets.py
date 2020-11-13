from typing import List, Optional

import pandas as pd

from .parsing_utilities import split_gene_label_series


def prepare_targets(
    df: pd.DataFrame,
    top_variance_filter: Optional[int],
    gene_filter: Optional[List[str]],
) -> pd.DataFrame:
    if gene_filter is not None:
        gene_symbol, _ = split_gene_label_series(df.columns)
        columns = pd.DataFrame({"column_name": df.columns, "gene_symbol": gene_symbol})
        filtered_columns = columns[columns["gene_symbol"].isin(gene_filter)][
            "column_name"
        ]
        if len(filtered_columns) == 0:
            raise ValueError("No matching genes found")
        df = df.filter(items=filtered_columns, axis="columns")

    if top_variance_filter is not None:
        df = df.filter(
            items=df.var(axis=0)
            .sort_values(ascending=False)[:top_variance_filter]
            .index,
            axis="columns",
        )

    df.index.name = "Row.name"

    return df
