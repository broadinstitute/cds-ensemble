from typing import Dict, List, Optional

import pandas as pd


class ModelConfig:
    def __init__(
        self,
        name: str,
        features: List[str],
        required_features: List[str],
        relation: str,
        exempt: Optional[List[str]],
    ):
        self.name = name
        self.features = features
        self.required_features = required_features
        self.relation = relation
        self.exempt = exempt


class FeatureInfo:
    def __init__(self, dataset: str, filename: str, normalize: bool = True):
        self.dataset_name = dataset
        self.file_name = filename
        self.normalize = normalize
        self.data: Optional[pd.DataFrame] = None

    def set_dataframe(self, df: pd.DataFrame):
        self.data = df
