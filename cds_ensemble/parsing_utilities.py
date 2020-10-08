import os
from typing import List

import pandas as pd
import yaml

from .models import ModelConfig, FeatureInfo


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
    returns only the "Row.name" column

    Args:
        file_path (str): Path of file to read

    Returns:
        pd.Series: Series with the "Row.name" column of the file
    """
    _, file_extension = os.path.splitext(file_path)
    if file_extension in {".ftr", ".feather"}:
        return pd.read_feather(file_path, columns=["Row.name"])

    return pd.read_csv(
        file_path, sep="\t" if file_extension == ".tsv" else ",", usecols=["Row.name"]
    )


def read_model_config(file_path: str) -> List[ModelConfig]:
    with open(file_path) as f:
        raw_model_configs = yaml.load(f, Loader=yaml.SafeLoader)

    model_configs: List[ModelConfig] = []
    for model_name, model_config in raw_model_configs.items():
        try:
            model_configs.append(
                ModelConfig(
                    name=model_name,
                    features=model_config["Features"],
                    required_features=model_config["Required"],
                    relation=model_config["Relation"],
                )
            )
        except KeyError as e:
            raise ValueError(f"Definition for model {model_name} is missing {str(e)}")

    return model_configs


def read_feature_info(file_path: str) -> List[FeatureInfo]:
    df = read_dataframe(file_path, set_index=False)
    feature_infos = [
        FeatureInfo(dataset=row["dataset"], filename=row["filename"])
        for i, row in df.iterrows()
    ]

    return feature_infos
