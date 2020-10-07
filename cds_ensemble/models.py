from typing import Dict, List, Optional

import pandas as pd


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


class FeatureInfo:
    def __init__(self, dataset: str, filename: str):
        self.dataset_name = dataset
        self.file_name = filename
        self.data: Optional[pd.DataFrame] = None

    def set_dataframe(self, df: pd.DataFrame):
        self.data = df
