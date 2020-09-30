import os
import pathlib
import yaml

from typing import List, Optional

import click
import pandas as pd

from prepare_targets import prepare_targets
from prepare_features import prepare_features, FeatureInfo


@click.group()
def main():
    pass


@main.command()
@click.option("--input", required=True, help="The file to filter and reformat")
@click.option("--output", required=True, help="Path where to write the output")
@click.option(
    "--top-variance-filter",
    type=int,
    help="If specified, will only keep the top N targets, ranked by variance",
)
@click.option("--gene-filter", help="If specified, will only keep the listed genes")
def prepare_y(
    input: str,
    output: str,
    top_variance_filter: Optional[int],
    gene_filter: Optional[str],
):
    if not os.path.exists(input):
        click.secho(f"File {input} not found")
        return

    if top_variance_filter is not None and top_variance_filter < 1:
        click.secho("Top variance filter must be >= 1", fg="red")
        return

    try:
        if input.endswith(".ftr") or input.endswith(".feather"):
            df = pd.read_feather(input)
            df.set_index(df.columns[0])
        elif input.endswith(".tsv"):
            df = pd.read_csv(input, index_col=0, sep="\t")
        else:
            df = pd.read_csv(input, index_col=0)
    except:
        click.secho(f"Could not read {input} as CSV", fg="red")
        return

    try:
        df = df.astype(float)
    except:
        click.secho(f"Values in {input} must all be numbers", fg="red")
        return

    try:
        gene_filter_list: Optional[List[str]] = None

        if gene_filter is not None:
            gene_filter_list = [gene.strip() for gene in gene_filter.split(",")]

        # Filter targets based on variance and/or gene
        filtered_df = prepare_targets(df, top_variance_filter, gene_filter_list)

        # Make output parent directories if they don't already exist
        pathlib.Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True)

        # Reset index because feather does not support indexes, then output as feather
        filtered_df.reset_index().to_feather(output)
    except ValueError as e:
        click.secho(str(e), fg="red")


@main.command()
@click.option(
    "--model-config",
    required=True,
    type=str,
    help="The file with model configurations (need to define format of this below) TODO",
)
@click.option(
    "--output",
    required=True,
    type=str,
    help="Full path to where to write the merged matrix",
)
@click.option("--targets", required=True, help="Matrix of the targets we are modeling")
@click.option(
    "--feature-info",
    required=True,
    help="Table containing feature datasets required and filename columns",
)
@click.option("--confounders", help="Table with target dataset specific QC, e.g. NNMD")
@click.option(
    "--output-format",
    type=click.Choice([".ftr", ".csv"], case_sensitive=False),
    help="Which format to write the output in (.ftr or .csv)",
)
@click.option(
    "--output-related",
    help='if specified, write out a file which can be used with "cds-ensemble fit-model --feature-subset ..." to select only related features for each target.',
)
def prepare_x(
    model_config: str,
    output: str,
    targets: str,
    feature_info: str,
    confounders: Optional[str],
    output_format: Optional[str],
    output_related: Optional[str],
):
    for p in [model_config, targets, feature_info]:
        if not os.path.exists(p):
            click.secho(f"File {p} not found", fg="red")
            return

    with open(model_config) as f:
        # TODO ?
        parsed_model_config = yaml.load(f, Loader=yaml.SafeLoader)

    target_samples = set(pd.read_feather(targets, columns=["Row.name"])["Row.name"])
    feature_infos = [
        FeatureInfo(**fi)
        for fi in pd.read_csv(feature_info, sep="\t").to_dict(orient="records")
    ]

    # TODO handle related
    all_model_features, all_model_feature_metadata = prepare_features(
        parsed_model_config, target_samples, feature_infos, confounders
    )

    if output_format == ".csv":
        all_model_features.to_csv(output)
    else:
        all_model_features.reset_index().to_feather(output)

    all_model_feature_metadata.reset_index().to_feather("feature_metadata.ftr")


@main.command()
@click.option("--x", required=True)
@click.option("--y", required=True)
@click.option("--valid_samples_file")
@click.option("--feature-subset")
@click.option("--target-range")
@click.option("--targets")
def fit_models():
    pass


if __name__ == "__main__":
    main()
