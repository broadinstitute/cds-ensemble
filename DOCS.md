# cds-ensemble Documentation

## Table of Contents
- [prepare-y](#prepare-y)
- [prepare-x](#prepare-x)
- [fit-model](#fit-model)

## prepare-y
```shell
cds-ensemble prepare-y \
--input [INPUT_FILE_PATH] \
--output [OUTPUT_FILE_PATH] \
--top-variance-filter [TOP_N_VARIANCE] \
--gene-filter [LIST_OF_GENES]
```

Prepares the targets to be consumed by `fit-models`. Concretely, filters if one is specified and outputs a feather file. Only one filter may be applied.

### Options
- `--input` \
    CSV, TSV, or feather file containing targets. The table in the file must have targets as the column headers (genes in the format `<GENE_SYMBOL> (<ENTREZ_ID>)`) and samples as the row headers.
- `--output` \
    Path to output the prepared targets. Will output as a feather file. Should have either `.feather` or `.ftr` extension.
- `--top-variance-filter` (optional) \
    If specified, will only keep the top N targets, ranked by variance.
- `--gene-filter` (optional) \
    If specified, will only keep the listed genes. List should be a comma-separated list of gene symbols with no spaces, e.g. `SOX10,NRAS,BRAF`

### Examples
```shell
cds-ensemble prepare-y --input targets.csv --output prepared_targets
cds-ensemble prepare-y --input targets.tsv --output prepared_targets --top-variance-filter 10
cds-ensemble prepare-y --input targets.ftr --output prepared_targets --gene-filter SOX10,NRAS,BRAF
```

[Top](#cds-ensemble-documentation)

## prepare-x
```shell
cds-ensemble prepare-x \
--targets [TARGET_FILE_PATH] \
--model-config [MODEL_CONFIG_FILE_PATH]\
--feature-info [FEATURE_INFO_FILE_PATH]\
--output [OUTPUT_FILE_PATH]\
--confounders [CONFOUNDERS_DATASET_NAME]\
--output-format [.csv | .ftr]\
--output-related [RELATED_OUTPUT_FILE_PATH]
```

Prepares the features to be consumed by `fit-models`

### Parameters
- `--targets` \
  Matrix of the targets we are modeling

- `--model-config` \
  The file with model configurations (need to define format of this below) TODO

- `--feature-info` \          Table containing feature datasets required and
                               filename columns

- `--output` \
  File name (without extension) for where to write the merged matrix

- `--confounders` (optional) \
  Table with target dataset specific QC, e.g. NNMD

- `--output-format` (optional | default: .ftr) \
  Which format to write the output in (.ftr or .csv)

- `--output-related` (optional) \
  If specified, write out a file which can be used with "cds-ensemble fit-model --feature-subset ..." to select only related features for each target.


### Examples
```shell
cds-ensemble prepare-x --model-config model-config.yaml --output x_output --targets prepared_targets.ftr --feature-info feature-info.csv --output-related related
```

[Top](#cds-ensemble-documentation)
