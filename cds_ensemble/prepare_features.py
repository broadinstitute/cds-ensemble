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


def get_valid_samples_for_model(
    model_config: ModelConfig,
    feature_infos: List[FeatureInfo],
    combined_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filters samples to those found in the required features datasets.

    Args:
        model_config (ModelConfig): The model definition
        feature_infos (List[FeatureInfo]): FeatureInfos with processed data, used to
            filter required samples
        combined_features (pd.DataFrame): DataFrame of all processed features

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Filtered combined features,
            filtered metadata
    """
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
        return combined_features.index.isin(model_required_samples)

    return combined_features.index.isin(combined_features.index)


def prepare_features(
    model_configs: Dict[str, ModelConfig],
    target_samples: pd.Series,
    feature_infos: List[FeatureInfo],
    confounders: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Processes and merges features for each of the models in `model_configs`

    Args:
        model_configs (Dict[str, ModelConfig]): Model definitions for which to process
            features
        target_samples (pd.Series): Samples in the target, i.e. for which there exists
            actual values to compare with predicted values
        feature_infos (List[FeatureInfo]): FeatureInfos with dataset file names
        confounders (Optional[str]): Name of confounders

    Returns:
        [Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]: (Processed and combined
            features, mapping of original dataset/columns to processed columns, valid
            samples for each model)
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

    missing_features = features_in_any_model - set(
        feature_info.dataset_name for feature_info in subsetted_feature_infos
    )

    if len(missing_features) > 0:
        raise ValueError(
            f"Features {missing_features} listed in model definitions, but not found in feature info table."
        )

    combined_features, feature_metadata = prepare_universal_feature_set(
        target_samples, subsetted_feature_infos
    )

    for _, model_config in model_configs.items():
        valid_samples_for_models = {
            model_config.name: get_valid_samples_for_model(
                model_config, feature_infos, combined_features
            )
            for _, model_config in model_configs.items()
        }

        model_valid_samples = pd.DataFrame(
            data=valid_samples_for_models, index=combined_features.index
        )

    return combined_features, feature_metadata, model_valid_samples


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

    target_gene_symbol, target_entrez_id = split_gene_label_series(
        unprocessed_related_table["target"]
    )
    partner_gene_symbol, partner_entrez_id = split_gene_label_series(
        unprocessed_related_table["partner"]
    )

    related = pd.DataFrame(
        {
            "target_gene_symbol": target_gene_symbol,
            "target_entrez_id": target_entrez_id,
            "partner_gene_symbol": partner_gene_symbol,
            "partner_entrez_id": partner_entrez_id,
        }
    )

    related = related.dropna(axis="index", how="any")
    if related.shape[0] == 0:
        raise ValueError(
            "Related table has 'target' or 'partner' column not in GENE_SYMBOL (ENTREZ_ID) format."
        )
    return related
