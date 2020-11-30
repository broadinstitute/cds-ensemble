# cds-ensemble

cds-ensemble is a command line tool for running model fitting portion of the ensemble prediction pipeline.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install cds-ensemble.

```bash
pip install cds-ensemble
```

## Usage

[See detailed documentation here](/DOCS.md)

```
cds-ensemble prepare-y --input targets.csv --output prepared_targets
cds-ensemble prepare-x --model-config model-config.yaml --output prepared_features --targets prepared_targets.ftr --feature-info feature-info.csv --output-related related
cds-ensemble fit-model --x prepared_features.ftr --y prepared_targets.ftr --model-config model-config.yaml --model MatchRelated --target-range 0 10 --related-table related.ftr
```

## Contributing
### Setup
Run the setup script. This installs pip requirements and creates pre-commit hooks.

```bash
sh setup.sh
```

## License
