from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import pandas as pd


@dataclass
class ModelConfig:
    def __init__(
        self,
        name: str,
        features: List[str],
        required_features: List[str],
        relation: str,
    ):
        self.name = name
        self.features = features
        self.required_features = required_features
        self.relation = relation


@dataclass
class FeatureInfo:
    def __init__(
        self, dataset: str, filename: str, data_format: Literal["matrix", "table"]
    ):
        self.dataset_name = dataset
        self.file_name = filename
        self.data_format = data_format
        self.data: Optional[pd.DataFrame] = None
        self.one_hot_mapping: Optional[Dict[str, str]]

    def set_dataframe(self, df: pd.DataFrame):
        self.data = df
