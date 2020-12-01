# cds-ensemble Documentation

## Table of Contents
- [Commands](#Commands)
  - [prepare-y](#prepare-y)
  - [prepare-x](#prepare-x)
  - [fit-model](#fit-model)
- [File formats](#file-formats)
  - [Model definitions](#model-definitions)
  - [Feature information](#feature-information)
  - [Features](#features)
  - [Related features](#related-features)
  - [Targets](#targets)

## Commands
### prepare-y
```shell
cds-ensemble prepare-y \
--input [INPUT_FILE_PATH] \
--output [OUTPUT_FILE_PATH] \
--top-variance-filter [TOP_N_VARIANCE] \
--gene-filter [LIST_OF_GENES]
```

Prepares the targets to be consumed by `fit-model`. Concretely, filters if one is specified and outputs a feather file. Only one filter may be applied.

#### Options
- `--input` \
    CSV, TSV, or feather file containing targets. The table in the file must have targets as the column headers (genes in the format `<GENE_SYMBOL> (<ENTREZ_ID>)`) and samples as the row headers.
- `--output` \
    Path to output the prepared targets. Will output as a feather file. Should have either `.feather` or `.ftr` extension.
- `--top-variance-filter` (optional) \
    If specified, will only keep the top N targets, ranked by variance.
- `--gene-filter` (optional) \
    If specified, will only keep the listed genes. List should be a comma-separated list of gene symbols with no spaces, e.g. `SOX10,NRAS,BRAF`

#### Examples
```shell
cds-ensemble prepare-y --input targets.csv --output prepared_targets
cds-ensemble prepare-y --input targets.tsv --output prepared_targets --top-variance-filter 10
cds-ensemble prepare-y --input targets.ftr --output prepared_targets --gene-filter SOX10,NRAS,BRAF
```

[Top](#cds-ensemble-documentation)

### prepare-x
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

Prepares the features to be consumed by `fit-model`. Outputs file in the format `[OUTPUT_FILE_PATH].[OUTPUT_FORMAT]`, and also outputs a feature metadata file with path `[OUTPUT_FILE_PATH]_feature_metadata.[OUTPUT_FORMAT]` and valid samples per model file with path `[OUTPUT_FILE_PATH]_valid_samples.[OUTPUT_FORMAT]`. `fit-models` assumes those files exist.

#### Options
- `--targets` \
  Matrix of the targets we are modeling

- `--model-config` \
  The file with model configurations (need to define format of this below) TODO

- `--feature-info` \
  Table containing feature datasets required and filename columns

- `--output` \
  File name (without extension) for where to write the merged matrix

- `--confounders` (optional) \
  Table with target dataset specific QC, e.g. NNMD

- `--output-format` (optional | default: .ftr) \
  Which format to write the output in (.ftr or .csv)

- `--output-related` (optional) \
  If specified, write out a file which can be used with "cds-ensemble fit-model --feature-subset ..." to select only related features for each target.


#### Examples
```shell
cds-ensemble prepare-x --model-config model-config.yaml --output x_output --targets prepared_targets.ftr --feature-info feature-info.csv --output-related related
```

[Top](#cds-ensemble-documentation)


### fit-model
```shell
cds-ensemble fit-model \
--x [PREPARED_FEATURES_FILE_PATH] \
--y [PREPARED_TARGETS_FILE_PATH] \
--model-config [MODEL_CONFIG_FILE_PATH] \
--model [MODEL_NAME] \
--target-range [START_INDEX] [END_INDEX] \
--related-table [RELATED_FEATURES_FILE_PATH]
```

Prepares the features to be consumed by `fit-model`

#### Options
- `--x` \
  Matrix of prepared features (output of `prepare-x`)

- `--y` \
  Matrix of the targets we are modeling (output of `prepare-y`)

- `--model-config` \
  The file with model configurations (need to define format of this below) See [Model definition](#model-definitions) for file format.

- `--model` \
  Name of model to fit (as defined in the model configurations)

- `task-mode` (optional | default: regress) \
  `regress` or `classify`

- `n-folds` (optional | default: 3) \
  Number of cross validation folds

- `related-table` (optional\*) \
  Table with relationships for genes, used for models with `Relation: MatchRelated`. See [Related features](#related-features) for file format.
  \*required if model has `Relation: MatchRelated`


- `feature-metadata` (optional) \
  File path for the feature metadata file. If not specified, will try to use the file matching the `--x` parameter.

- `model-valid-samples` (optional) \
  File path for the valid samples per model file. If not specified, will try to use the file matching the `--x` parameter.

- `feature-subset-file` (optional) \
  File with list of features to include, separated by new lines.

- `valid-samples-file` (optional) \
  File with list of samples to include, separated by new lines.

- `target-range` (optional) \
  The range of targets to fit models for, in the form `[start index] [end index]`. If not specified, will fit models for all targets. Cannot be combined with `--targets`

- `targets` (optional) \
  List of targets (either `[HUGO symbol]`, or `[HUGO symbol] ([Entrez ID])`) to fit models for. If not specified, will fit models for all targets. Cannot be combined with `--target-range`

- `output-dir` (optional) \
  Which directory/folder to output files to. Folder must exist.


#### Examples
```shell
cds-ensemble fit-model --x prepared_features.ftr --y prepared_targets.ftr --model-config model-config.yaml --model MatchRelated --target-range 0 10 --related-table related.ftr
```

[Top](#cds-ensemble-documentation)

## File formats
### Model definitions
The `model-config` file used in `prepare-x` and `fit-model`. This should be a yaml file containing a list of model definitions of the following format:
```yaml
[model name]:
  Features: [list of feature/dataset names that matches feature information file]
  Required: [list of features that are required, i.e. exclude samples that are not in each of these datasets]
  Relation: [All, MatchTarget, or MatchRelated]
  Related: [optional, dataset defining relations if Relation = MatchRelated]
  Exempt: [list of features that should be included per feature-fitting, regardless of relation]

```
#### Example
```yaml
---
Unbiased:
  Features:
  - full_matrix
  - full_table
  - partial_matrix
  - partial_table
  Required:
  - full_matrix
  Relation:    All
  Exempt:
Related:
  Features:
  - full_matrix
  - full_table
  - partial_matrix
  - partial_table
  Required:
  - full_matrix
  Relation:    MatchRelated
  Related: related
  Exempt:
  - partial_table
```
[Top](#cds-ensemble-documentation)


### Feature information
Table of dataset name (that matches features specified in model definitions) to file path. Files can be CSV, TSV, or Feather, and `prepare-x` will infer file type based on file extension.
#### Example
```csv
dataset,filename
full_matrix,full_matrix.csv
full_table,full_table.tsv
partial_matrix,partial_matrix.ftr
partial_table,partial_table.feather
related,related.csv
confounders,confounders.csv
```
[Top](#cds-ensemble-documentation)


### Features
Feature datasets/matrices should have samples as the row header/index and feature names as the column headers. Features that are gene names that should be matched when relation is MatchTarget or MatchRelated should be in the form `[HUGO symbo] ([Entrez ID])`
#### Example
```csv
,SOX10 (6663),NRAS (4893),BRAF (673)
sample-1,0,0.5,0.5
sample-2,0.6,0.6,0.7
sample-3,0.3,0.4,0.4
```
[Top](#cds-ensemble-documentation)


### Related features
The related features table should have two columns, `target` and `partner`, with each row representing a pair of related genes. The genes should be in the same format as in the features (`[HUGO symbo] ([Entrez ID])`). Columns with that contain only NA, 0, or 1 will be interpretted as binary columns. Non-numeric columns with be interpretted as categorical.
#### Example
```csv
,SOX10 (6663),NRAS (4893),BRAF (673),binary,categorical
sample-1,0,0.5,0.5,,red,
sample-2,0.6,0.6,0.7,1,green,
sample-3,0.3,0.4,0.4,0,blue
```
[Top](#cds-ensemble-documentation)

### Targets
The targets matrix should have samples as the index and targets as the row headers. Targets should be genes in the same format as above. All values must be numeric.
#### Example
```csv
,SOX10 (6663),NRAS (4893),BRAF (673)
sample-1,0.4,0,0.9
sample-2,0,0.8,0.9
sample-3,0.4,0.5,0.3
```
[Top](#cds-ensemble-documentation)
