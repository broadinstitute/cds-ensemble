from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .data_models import ModelConfig, FeatureInfo
from .parsing_utilities import (
    GENE_LABEL_FORMAT,
    read_dataframe,
    split_gene_label_series,
)


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
    gene_symbol, entrez_id = split_gene_label_series(feature_metadata["feature_name"])
    feature_metadata["gene_symbol"] = gene_symbol
    feature_metadata["entrez_id"] = entrez_id

    df = df.rename(columns=renames)

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
    if numeric_subset.columns.size == df.columns.size:
        return numeric_subset, numeric_subset_feature_metadata

    # Process non-numeric columns
    categorical_subset = df.select_dtypes(exclude="number")
    (
        categorical_subset,
        categorical_subset_feature_metadata,
    ) = prepare_categorical_features(categorical_subset, dataset_name)

    # Merge processed numeric and categorical data and feature info DataFrames
    df = pd.concat([numeric_subset, categorical_subset], axis="columns")
    feature_metadata = numeric_subset_feature_metadata.append(
        categorical_subset_feature_metadata
    )
    return df, feature_metadata


def prepare_universal_feature_set(
    target_samples: pd.Series, feature_infos: List[FeatureInfo]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardizes and merges features from `feature_infos` into one pd.DataFrame.

    Args:
        target_samples (pd.Series): Target samples/rows that are allowed in the merged
            dataset
        feature_infos (List[FeatureInfo]): List of FeatureInfos that have the file
            names of the feature datasets to merge. DataFrames will be added.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Merged features DataFrame, mapping from
        original dataset/feature to new column name in merged DataFrame
    """
    feature_metadatas: List[pd.DataFrame] = []
    for feature_info in feature_infos:
        df = read_dataframe(feature_info.file_name)

        df, single_dataset_feature_metadata = prepare_single_dataset_features(
            df, feature_info.dataset_name, feature_info.normalize
        )
        feature_info.set_dataframe(df.filter(items=target_samples.values, axis="index"))
        feature_metadatas.append(single_dataset_feature_metadata)

    dataframes_to_combine = [feature_info.data for feature_info in feature_infos]

    combined_features = pd.concat(dataframes_to_combine, axis="columns", join="outer")

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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Subsets the combined features to only the features listed in the model
    definition, and also filters samples to those found in the required features
    datasets.

    Args:
        model_config (ModelConfig): The model definition
        feature_infos (List[FeatureInfo]): FeatureInfos with processed data, used to
            filter required samples
        confounders (Optional[str]): Confounders dataset to include, if provided
        combined_features (pd.DataFrame): DataFrame of all processed features
        feature_metadata (pd.DataFrame): Corresponding metadata

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Filtered combined features,
            filtered metadata
    """
    # All features listed in the model definitions Features or Required, and also the
    # confounders, if provided
    features_to_use = model_config.features
    for feature in model_config.required_features:
        if feature not in features_to_use:
            features_to_use.append(feature)
    if confounders is not None and confounders not in features_to_use:
        features_to_use.append(confounders)

    # Filter metadata by datasets used and filter matching columns in combined_features
    model_feature_metadata = feature_metadata[
        feature_metadata["dataset"].isin(features_to_use)
    ]
    model_features = combined_features.filter(
        items=model_feature_metadata["feature_id"], axis="columns"
    )

    # For features that are listed as required, filter out samples that are not in the
    # required dataset's samples
    model_required_samples: Optional[pd.Index] = None
    for feature_info in feature_infos:
        if feature_info.dataset_name not in model_config.required_features:
            continue

        if model_required_samples is None:
            model_required_samples = feature_info.data.index
        else:
            model_required_samples = model_required_samples.intersection(
                feature_info.data.index
            )
    if model_required_samples is not None:
        model_features = model_features.filter(
            items=model_required_samples, axis="index"
        )

    return model_features, model_feature_metadata


def prepare_features(
    model_configs: Dict[str, ModelConfig],
    target_samples: pd.Series,
    feature_infos: List[FeatureInfo],
    confounders: Optional[str],
) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Processes and merges features for each of the models in `model_configs`

    Args:
        model_configs (Dict[str, ModelConfig]): Model definitions for which to process
            features
        target_samples (pd.Series): Samples in the target, i.e. for which there exists
            actual values to compare with predicted values
        feature_infos (List[FeatureInfo]): FeatureInfos with dataset file names
        confounders (Optional[str]): Name of confounders

    Returns:
        List[Tuple[str, pd.DataFrame, pd.DataFrame]]: List of (Processed and combined
            features, mapping of original dataset/columns to processed columns) for all
            models defined
    """
    features_in_any_model: Set[str] = set()
    for _, model_config in model_configs.items():
        features_in_any_model.update(model_config.features)
        features_in_any_model.update(model_config.required_features)
        if confounders is not None:
            features_in_any_model.add(confounders)

    subsetted_feature_infos = [
        feature_info
        for feature_info in feature_infos
        if feature_info.dataset_name in features_in_any_model
    ]

    combined_features, feature_metadata = prepare_universal_feature_set(
        target_samples, subsetted_feature_infos
    )

    models_features_and_metadata: List[Tuple[str, pd.DataFrame, pd.DataFrame]] = []

    for _, model_config in model_configs.items():
        model_features, model_feature_metadata = subset_by_model_config(
            model_config,
            feature_infos,
            confounders,
            combined_features,
            feature_metadata,
        )
        if model_features is not None:
            models_features_and_metadata.append(
                (model_config.name, model_features, model_feature_metadata)
            )

    return models_features_and_metadata


def format_related(
    model_configs: Mapping[str, ModelConfig], feature_infos: Iterable[FeatureInfo]
) -> pd.DataFrame:
    related_datasets = set(
        model_config.related_dataset
        for model_config in model_configs.values()
        if model_config.related_dataset is not None
    )

    if len(related_datasets) == 0:
        raise ValueError("No related dataset found for any of the model definitions")
    elif len(related_datasets) > 1:
        raise ValueError("Multiple related datasets found for model definitions")

    related_dataset_name = related_datasets.pop()

    try:
        related_feature_info = next(
            feature_info
            for feature_info in feature_infos
            if feature_info.dataset_name == related_dataset_name
        )
    except StopIteration:
        raise ValueError(
            f'No dataset "{related_dataset_name}" found in feature infos file'
        )

    unprocessed_related_table = read_dataframe(
        related_feature_info.file_name, set_index=False
    )

    if not (
        unprocessed_related_table["target"].str.match(GENE_LABEL_FORMAT).all()
        and unprocessed_related_table["partner"].str.match(GENE_LABEL_FORMAT).all()
    ):
        raise ValueError(
            "Related dataset target or partner columns not in 'GENE_SYMBOL (entrez id)' format"
        )

    target_gene_symbol, target_entrez_id = split_gene_label_series(
        unprocessed_related_table["target"]
    )
    partner_gene_symbol, partner_entrez_id = split_gene_label_series(
        unprocessed_related_table["partner"]
    )

    return pd.DataFrame(
        {
            "target_gene_symbol": target_gene_symbol,
            "target_entrez_id": target_entrez_id,
            "partner_gene_symbol": partner_gene_symbol,
            "partner_entrez_id": partner_entrez_id,
        }
    )
