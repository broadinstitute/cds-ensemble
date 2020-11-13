import re
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from .data_models import ModelConfig, FeatureInfo
from .exceptions import MalformedGeneLabelException

GENE_LABEL_FORMAT = r"^[A-Z\d\-]+ \(\d+\)$"
GENE_LABEL_FORMAT_GROUPS = r"^([A-Z\d\-]+) \((\d+)\)$"


def read_dataframe(file_path: str, set_index: bool = True) -> pd.DataFrame:
    """Reads a file as a feather, TSV, or CSV file based on extensions

    Args:
        file_path (str): Path of file to read
        set_index (bool): Whether to set the first column as the index

    Returns:
        pd.DataFrame: DataFrame stored at file_path
    """
    _, file_extension = os.path.splitext(file_path)
    if file_extension in {".ftr", ".feather"}:
        df = pd.read_feather(file_path)
        if set_index:
            df = df.set_index(df.columns[0])
        return df

    return pd.read_csv(
        file_path,
        sep="\t" if file_extension == ".tsv" else ",",
        index_col=0 if set_index else None,
    )


def read_dataframe_row_headers(file_path: str) -> pd.Series:
    """Reads a file as a feather, TSV, or CSV file based on extensions, and
    returns only the first column

    Args:
        file_path (str): Path of file to read

    Returns:
        pd.Series: Series with the first column of the file
    """
    _, file_extension = os.path.splitext(file_path)
    if file_extension in {".ftr", ".feather"}:
        df = pd.read_feather(file_path, columns=[0])
    else:
        df = pd.read_csv(
            file_path, sep="\t" if file_extension == ".tsv" else ",", usecols=[0]
        )
    return df.squeeze()


def read_model_config(file_path: str) -> Dict[str, ModelConfig]:
    with open(file_path) as f:
        raw_model_configs = yaml.load(f, Loader=yaml.SafeLoader)

    model_configs: Dict[str, ModelConfig] = {}
    for model_name, model_config in raw_model_configs.items():
        try:
            model_configs[model_name] = ModelConfig(
                name=model_name,
                features=model_config["Features"],
                required_features=model_config["Required"],
                related_dataset=model_config.get("Related"),
                relation=model_config["Relation"],
                exempt=model_config.get("Exempt"),
            )
        except KeyError as e:
            raise ValueError(f"Definition for model {model_name} is missing {str(e)}")

    return model_configs


def read_feature_info(file_path: str, confounders: Optional[str]) -> List[FeatureInfo]:
    df = read_dataframe(file_path, set_index=False)
    feature_infos = []
    for i, row in df.iterrows():
        try:
            if not os.path.exists(row["filename"]):
                raise ValueError(f"File { row.filename } not found")
            feature_infos.append(
                FeatureInfo(
                    dataset=row["dataset"],
                    filename=row["filename"],
                    normalize=row["dataset"] == confounders,
                )
            )
        except KeyError as e:
            raise ValueError(f"Feature info file is missing {str(e)} column")

    return feature_infos


def split_gene_label_str(gene_label: str) -> Tuple[str, int]:
    """Extracts the gene symbol and entrez ID from a gene label

    Args:
        s (str): Gene label of the form GENE_SYMBOL (ENTREZ_ID)

    Returns:
        Tuple[str, int]: gene symbol, entrez ID
    """
    match = re.match(GENE_LABEL_FORMAT_GROUPS, gene_label)
    if match is None:
        raise MalformedGeneLabelException(gene_label)
    symbol, entrez_id = match.groups()
    return symbol, int(entrez_id)


def split_gene_label_series(col: pd.Series) -> Tuple[pd.Series, pd.Series]:
    split_gene_label = col.str.extract(GENE_LABEL_FORMAT_GROUPS)
    return (
        split_gene_label[0],
        pd.to_numeric(split_gene_label[1], errors="coerce").astype("Int64"),
    )
