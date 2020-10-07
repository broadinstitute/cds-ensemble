from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .models import ModelConfig, FeatureInfo


def normalize_col(col: pd.Series) -> pd.Series:
    """Normalizes column by z-score. Ignores one-hot encoded columns.

    Args:
        col (pd.Series): [description]

    Returns:
        pd.Series: [description]
    """
    if col.isin([0, 1, np.nan]).all():
        return col
    col = (col - col.mean()) / col.std()
    return col


def standardize_col_name(
    df: pd.DataFrame, dataset_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardizes column names (by replacing whitespace and hyphens with underscores
    and appending the dataset name) and returns map of dataframe with mapping

    Args:
        df (pd.DataFrame): DataFrame whose columns will be renamed
        dataset_name (str): Dataset name to append to column names

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Copy of df with renamed columns, DataFrame
        with columns ["feature_id": new column name, "feature_name": old column name,
        "dataset": dataset name]
    """
    renames = {
        old: new
        for old, new in zip(
            df.columns,
            df.add_suffix("_" + dataset_name).columns.str.replace(r"[\s-]", "_"),
        )
    }

    feature_metadata = pd.DataFrame(
        {
            "feature_id": renames.values(),
            "feature_name": renames.keys(),
            "dataset": dataset_name,
        }
    )

    df = df.rename(columns=renames)

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


def prepare_numeric_features(
    df: pd.DataFrame, dataset_name: str, normalize: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardizes the column names of a DataFrame with only numeric values, and
    normalizes column values if specified.

    Args:
        df (pd.DataFrame): [description]
        dataset_name (str): [description]
        normalize (bool): Whether to normalize the columns by z-score

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: [description]
    """
    if normalize:
        df = df.apply(normalize_col, result_type="broadcast")
    df, feature_metadata = standardize_col_name(df, dataset_name)
    return df, feature_metadata


def prepare_categorical_features(
    df: pd.DataFrame, dataset_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encodes columns and standardizes the column names of a DataFrame with
    only non-numeric columns.

    Args:
        df (pd.DataFrame): [description]
        dataset_name (str): [description]

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: [description]
    """
    feature_metadata: Optional[pd.DataFrame] = None
    one_hot_dfs: List[pd.DataFrame] = []

    for col in df.columns:
        one_hot = pd.get_dummies(df[col], columns=[col])
        one_hot, single_feature_metadata = standardize_col_name(one_hot, dataset_name)
        # Set feature_name to original column name, not one-hot encoded column name
        single_feature_metadata["feature_name"] = col

        one_hot_dfs.append(one_hot)

        if feature_metadata is None:
            feature_metadata = single_feature_metadata
        else:
            feature_metadata = feature_metadata.append(single_feature_metadata)

    df = pd.concat(one_hot_dfs, axis="columns")
    return df, feature_metadata


def prepare_single_dataset_features(
    df: pd.DataFrame, dataset_name: str, normalize: bool = True
):
    """Standardizes the column names of a DataFrame, normalizes numeric columns values
    if specified, and one-hot encodes non-numeric column values.

    Args:
        df (pd.DataFrame): [description]
                dataset_name (str): [description]
        dataset_name (str): [description]
        normalize (bool): Whether to normalize the columns by z-score

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: [description]
    """
    # Process numeric
    numeric_subset = df.select_dtypes(include="number")
    numeric_subset, numeric_subset_feature_metadata = prepare_numeric_features(
        numeric_subset, dataset_name, normalize
    )
    # If all columns in df are numeric, return
    if numeric_subset.columns.equals(df.columns):
        return numeric_subset, numeric_subset_feature_metadata

    # Process non-numeric columns
    categorical_subset = df.select_dtypes(exclude="number")
    (
        categorical_subset,
        categorical_subset_feature_metadata,
    ) = one_hot_encode_and_standardize_col_name(categorical_subset, dataset_name)

    # Merge processed numeric and categorical data and feature info DataFrames
    df = numeric_subset.merge(categorical_subset, left_index=True, right_index=True)
    feature_metadata = numeric_subset_feature_metadata.append(
        categorical_subset_feature_metadata
    )
    return df, feature_metadata


def prepare_universal_feature_set(
    target_samples: Set[str],
    feature_infos: List[FeatureInfo],
    # confounders, # TODO
):
    feature_metadatas: List[pd.DataFrame] = []
    for feature_info in feature_infos:
        print(feature_info.dataset_name)
        df = pd.read_feather(feature_info.file_name).set_index("Row.name")

        df, single_dataset_feature_metadata = prepare_single_dataset_features(
            df, feature_info.dataset_name
        )
        feature_info.set_dataframe(df.filter(items=target_samples, axis="index"))
        feature_metadatas.append(single_dataset_feature_metadata)
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

    feature_metadata = pd.concat(feature_metadatas, axis="rows")

    return combined_features, feature_metadata


def subset_by_model_config(
    model_config: ModelConfig,
    feature_infos: List[FeatureInfo],
    confounders: Optional[str],
    combined_features: pd.DataFrame,
    feature_metadata: pd.DataFrame,
):
    features_to_use = model_config.features
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
        if feature_info not in model_config.required_features:
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
    model_configs: List[ModelConfig],
    target_samples: Set[str],
    feature_infos: List[FeatureInfo],
    confounders: Optional[str],
):
    features_in_any_model: Set[str] = set()
    for model_config in model_configs:
        features_in_any_model.update(model_config.features)
        # TODO: confounders
        # if model_config.confounders:
        #     features_in_any_model.add("Confounders")

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
        next(
            model_config
            for model_config in model_configs
            if model_config.name == "Unbiased"
        ),
        feature_infos,
        confounders,
        combined_features,
        feature_metadata,
    )

    return all_model_features, all_model_feature_metadata
