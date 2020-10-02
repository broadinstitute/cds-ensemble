from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set

import pandas as pd


@dataclass
class FeatureInfo:
    def __init__(
        self,
        dataset: str,
        filename: str,
        source_data: str,
        source_version: int,
        source_file: str,
        format: Literal["matrix", "table"],
    ):
        self.dataset_name = dataset
        self.file_name = filename
        self.data_format = format
        self.data: Optional[pd.DataFrame] = None
        self.one_hot_mapping: Optional[Dict[str, str]]

    def set_dataframe(self, df: pd.DataFrame):
        self.data = df


def normalize_col(col: pd.Series):
    col = col - col.mean()
    col = col / col.std()
    return col


def standardize_col_name(df: pd.DataFrame, dataset_name: str):
    cols = df.columns

    renamed_columns = df.add_suffix("_" + dataset_name).columns.str.replace(
        r"[\s-]", "_"
    )

    feature_metadata = pd.DataFrame(
        {"feature_id": renamed_columns, "feature_name": cols, "dataset": dataset_name}
    )

    df = df.rename(
        columns=dict(
            zip(feature_metadata["feature_name"], feature_metadata["feature_id"])
        )
    )

    return df, feature_metadata


def one_hot_encode_and_standardize_col_name(df: pd.DataFrame, dataset_name: str):
    feature_metadata = pd.DataFrame(columns=["feature_id", "feature_name", "dataset"])
    one_hot_dfs: List[pd.DataFrame] = []

    for col in df.columns:
        one_hot = pd.get_dummies(df[col], columns=[col])
        one_hot, dataset_feature_metadata = standardize_col_name(one_hot, dataset_name)
        dataset_feature_metadata["feature_name"] = col

        one_hot_dfs.append(one_hot)
        feature_metadata = feature_metadata.append(dataset_feature_metadata)

    df = pd.concat(one_hot_dfs, axis="columns")
    return df, feature_metadata


def prepare_universal_feature_set(
    target_samples: Set[str],
    feature_infos: List[FeatureInfo],
    # confounders, # TODO
):
    feature_metadata = pd.DataFrame(columns=["feature_id", "feature_name", "dataset"])
    for feature_info in feature_infos:
        print(feature_info.dataset_name)
        df = pd.read_feather(feature_info.file_name).set_index("Row.name")

        if feature_info.data_format == "table":
            num = df.select_dtypes(include="number")
            num = num.apply(normalize_col, result_type="broadcast")
            num, dataset_feature_metadata = standardize_col_name(
                num, feature_info.dataset_name
            )
            feature_metadata = feature_metadata.append(dataset_feature_metadata)

            # One-hot encode the non-numeric columns
            cat = df.select_dtypes(exclude="number")
            if len(cat.columns) > 0:
                one_hot, feature_metadata = one_hot_encode_and_standardize_col_name(
                    cat, feature_info.dataset_name
                )
                feature_metadata = feature_metadata.append(dataset_feature_metadata)
                df = num.merge(one_hot, left_index=True, right_index=True)
            else:
                df = num
        else:
            df = df.apply(normalize_col, result_type="broadcast")

            df, dataset_feature_metadata = standardize_col_name(
                df, feature_info.dataset_name
            )

            feature_metadata = feature_metadata.append(dataset_feature_metadata)

        feature_info.set_dataframe(df.filter(items=target_samples, axis="index"))
    combined_features = pd.concat(
        [feature_info.data for feature_info in feature_infos],
        axis="columns",
        join="outer",
    )

    combined_features = combined_features.astype(float)

    # drops samples that have all missing values
    combined_features = combined_features.dropna(how="all", axis=0)
    # drops variables that are all missing
    combined_features = combined_features.dropna(how="all", axis=1)

    combined_features = combined_features.fillna(0)

    return combined_features, feature_metadata


def subset_by_model_config(
    model_config,
    feature_infos: List[FeatureInfo],
    confounders: Optional[str],
    combined_features: pd.DataFrame,
    feature_metadata: pd.DataFrame,
):
    features_to_use = model_config["Features"]
    if confounders is not None:
        features_to_use.append(confounders)

    model_feature_metadata = feature_metadata[
        feature_metadata["dataset"].isin(features_to_use)
    ]
    model_features = combined_features.filter(
        items=model_feature_metadata["feature_id"], axis="columns"
    )

    model_required_samples: Optional[Set[str]] = None
    for feature_info in feature_infos:
        if feature_info not in model_config["Required"]:
            continue

        if model_required_samples is None:
            model_required_samples = set(feature_info.data.index)
        else:
            model_required_samples = model_required_samples.intersection(
                set(feature_info.data.index)
            )

    if model_required_samples is not None:
        model_features = model_features.filter(
            items=model_required_samples, axis="index"
        )

    return model_features, model_feature_metadata


def prepare_features(
    model_configs,
    target_samples: Set[str],
    feature_infos: List[FeatureInfo],
    confounders: Optional[str],
):
    features_in_any_model: Set[str] = set()
    for model_config in model_configs.values():
        features_in_any_model.update(model_config["Features"])
        if model_config["Confounders"]:
            features_in_any_model.add("Confounders")

    subsetted_feature_infos = [
        feature_info
        for feature_info in feature_infos
        if feature_info.dataset_name in features_in_any_model
    ]

    combined_features, feature_metadata = prepare_universal_feature_set(
        target_samples, subsetted_feature_infos
    )

    # Subset all?
    all_model_features, all_model_feature_metadata = subset_by_model_config(
        model_configs["Unbiased"],
        feature_infos,
        confounders,
        combined_features,
        feature_metadata,
    )

    return all_model_features, all_model_feature_metadata
