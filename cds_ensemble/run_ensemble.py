import gc
import math
import random
import re
from itertools import chain
from time import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from typing_extensions import TypedDict

from .data_models import ModelConfig
from .exceptions import MalformedGeneLabelException
from .parsing_utilities import split_gene_label_str


def filter_run_ensemble_inputs(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    model_config: ModelConfig,
    feature_metadata: pd.DataFrame,
    model_valid_samples: pd.DataFrame,
    valid_samples: Optional[pd.Series],
    feature_subset: Optional[pd.Series],
    target_range: Optional[Tuple[int, int]],
    targets: Optional[List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    datasets = model_config.features + model_config.required_features
    features = feature_metadata[feature_metadata["dataset"].isin(datasets)][
        "feature_id"
    ]
    X = X.filter(items=features, axis="columns")

    samples = model_valid_samples[model_valid_samples[model_config.name]].index
    X = X.filter(items=samples, axis="index")
    Y = Y.filter(items=samples, axis="index")

    if valid_samples is not None:
        X = X.filter(items=valid_samples, axis="index")
        Y = Y.filter(items=valid_samples, axis="index")

    if feature_subset is not None:
        X = X.filter(items=feature_subset, axis="columns")

    if target_range is None and targets is None:
        start_col = 0
        end_col = Y.shape[1]
    elif target_range is not None and targets is None:
        start_col, end_col = target_range
        Y = Y.iloc[:, start_col:end_col]
    elif target_range is None and targets is not None:
        start_col = 0
        end_col = Y.shape[1]
        Y = Y.filter(items=targets, axis="columns")
    else:
        raise ValueError("Only one of target_range and targets can be specified")

    Y = Y.dropna(how="all", axis=0)
    Y = Y.dropna(how="all", axis=1)

    return X, Y, start_col, end_col


def soft_roc_auc(ytrue, ypred):
    if all(ytrue > 0.5) or not any(ytrue > 0.5):
        return 0.5
    return roc_auc_score((ytrue > 0.5).astype(np.int), ypred)


def varfilter(X, threshold):
    return np.std(np.array(X), axis=0) > threshold


class QuantileKFold(KFold):
    """Quantile K-Folds cross-validator
    Sorts continuous values and then bins them into N/k number of bins,
    such that each bin contains k values.
    Values in each bin are then assigned a random fold 1:k
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Always ignored, exists for compatibility.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.
        groups : object
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        # y = check_array(y, ensure_2d=False, dtype=None)
        df = pd.DataFrame(data=y)
        df.columns = ["value"]
        df["position"] = range(0, df.shape[0])
        df = df.sort_values(by=["value"])

        fold_assignment = {}
        for i in range(0, math.ceil(df.shape[0] / self.n_splits)):
            list = [*range(0, self.n_splits)]
            random.shuffle(list)
            fold_assignment[i] = list
        fold_assignment = [*chain(*fold_assignment.values())]
        df["fold"] = [fold_assignment[i] for i in range(0, df.shape[0])]

        def split_pairs(df, nfolds):
            for i in range(nfolds):
                train_pos = df[df["fold"] != i]["position"]
                test_pos = df[df["fold"] == i]["position"]
                yield [train_pos.tolist(), test_pos.tolist()]

        split_generator = split_pairs(df=df, nfolds=self.n_splits)
        return split_generator


def single_fit(
    column,
    X,
    Y,
    model_types,
    splitter,
    scoring,
    nfeatures=10,
    rounding=False,
    return_models=False,
):
    y = Y[column]
    if rounding:
        if (y > 0.5).sum() == 0 or (y > 0.5).sum() == len(y):
            raise ValueError(
                "Column %s has %i of %i possible true values\n%r"
                % (column, (y > 0.5).sum(), len(y), Y[column])
            )
    else:
        if y.std() == 0:
            raise ValueError("Column %s has 0 variance\n%r" % (column, y))
    if y.isnull().any():
        raise ValueError(
            "Column %s of y contains %i nulls" % (column, y.isnull().sum())
        )
    target_models = [val["ModelClass"](**val["kwargs"]) for val in model_types]
    scores = []
    features = []
    prediction = []
    for model, x in zip(target_models, X):
        if x.isnull().any().any():
            raise ValueError(
                "Feature set for model %r contains nulls. Axial sums of nulls:\n%r\n\n%r"
                % (model, x.isnull().sum(), x.isnull().sum(axis=1))
            )
        if rounding:
            splits = splitter.split(y > 0.5, y > 0.5)
        else:
            splits = splitter.split(y, y)
        score = []
        malformed_gene_labels = []
        model_prediction = pd.Series(np.nan, index=Y.index, name=column)
        for train, test in splits:
            try:
                model.fit(x.iloc[train], y.iloc[train])
                ypred = model.predict(x.iloc[test])
            except MalformedGeneLabelException as e:
                print(e)
                malformed_gene_labels.append(e.gene_label)
                continue
            except Exception as e:
                print("error fitting model %r for column %s" % (model, column))
                print("train indices:\n %r\n" % train)
                print("test indices:\n %r\n" % test)
                print("train features: \n%r\n" % x.iloc[train])
                print("test features: \n%r\n" % x.iloc[test])
                print("train column: \n%r\n" % y.iloc[train])
                print("test column: \n%r\n" % y.iloc[test])
                raise e
            model_prediction.iloc[test] = ypred[:]
            score.append(scoring(y.iloc[test], ypred))

        scores.append(score)
        prediction.append(model_prediction)
        model.fit(x, y)
        try:
            features.append(model.get_feature_series(nfeatures))
        except AttributeError:
            features.append(pd.Series(dtype=np.float))
        gc.collect()
    best_index = np.argmax(np.mean(scores, axis=1))
    if not return_models:
        target_models = [np.nan for i in range(len(scores))]
    return {
        "models": target_models,
        "best": best_index,
        "scores": scores,
        "features": features,
        "predictions": prediction,
    }


##############################################################
#################### E N S E M B L E #########################
##############################################################


class EnsembleRegressor:
    def __init__(
        self,
        model_types,
        nfolds=3,
        scoring=soft_roc_auc,
        Splitter=StratifiedKFold,
        rounding=False,
    ):

        """
        model_types: [{'Name': str, 'ModelClass': class,  'kwargs': dict}] Model classes will be initiated with the
            dict of keyword arguments
        """
        self.model_types = model_types
        self.best_indices = {}
        self.trained_models = {}
        self.scores = {}
        self.important_features = {}
        self.nfolds = nfolds
        self.splitter = Splitter(n_splits=nfolds, shuffle=True)
        self.scoring = scoring
        self.rounding = rounding
        self.predictions = None

    def check_x(self, X):
        xerror = ValueError(
            "X must be a list or array with a feature set dataframe of matching indices for each model \
            present in the ensemble, passed\n%r"
            % X
        )
        if not len(X) == len(self.model_types):
            print("X not the same length as models\n")
            raise xerror
        for df in X[1:]:
            if not all(df.index == X[0].index):
                raise xerror

    def fit(self, X, Y, report_freq=20):
        """
        X: [{ModelClass: dataframe}
        Y: dataframe
        """
        self.check_x(X)
        assert isinstance(Y, pd.DataFrame)
        if not all(Y.index == X[0].index):
            raise ValueError(
                "Y must be a dataframe with index matching the indices in X"
            )
        columns = Y.columns
        n = len(self.model_types)
        outputs = {
            "models": {},
            "best": {},
            "scores": {},
            "features": {},
            "predictions": {},
        }
        start_time = time()
        curr_time = start_time
        for i, col in enumerate(columns):
            ind = Y.index[Y[col].notnull()]
            output = single_fit(
                column=col,
                X=[x.loc[ind] for x in X],
                Y=Y.loc[ind],
                model_types=self.model_types,
                splitter=self.splitter,
                scoring=self.scoring,
                rounding=self.rounding,
            )
            for key in outputs.keys():
                outputs[key][col] = output[key]
            t = time()
            if t - curr_time > report_freq:
                print(
                    "%f elapsed, %i%% complete, %f estimated remaining"
                    % (
                        t - start_time,
                        int(100 * (i + 1) / len(columns)),
                        (t - start_time) * (len(columns) - i - 1) * 1.0 / (i + 1),
                    )
                )
                curr_time = t
        self.trained_models.update(outputs["models"])
        self.best_indices.update(outputs["best"])
        self.scores.update(outputs["scores"])
        self.important_features.update(outputs["features"])
        predictions = [
            {col: val[j] for col, val in outputs["predictions"].items()}
            for j in range(n)
        ]
        if self.predictions is None:
            self.predictions = [pd.DataFrame(v) for v in predictions]
        else:
            for i in range(len(self.model_types)):
                self.predictions[i] = self.predictions[i].join(
                    outputs["predictions"][i]
                )

    def predict(self, X):
        self.check_x(X)
        return pd.DataFrame(
            {
                column: self.trained_models[column][self.best_indices[column]].predict(
                    X[self.best_indices[column]]
                )
                for column in self.columns
            },
            index=X[0].index,
        )

    def format_results(self):
        columns = ["gene", "model"]
        for i in range(self.nfolds):
            columns.append("score%i" % i)
        columns.append("best")
        for i in range(10):
            columns.extend(["feature%i" % i, "feature%i_importance" % i])

        melted = pd.DataFrame(columns=columns)
        for gene in self.trained_models.keys():
            for i in range(len(self.model_types)):
                row = {
                    "gene": gene,
                    "model": self.model_types[i]["Name"],
                    "best": self.best_indices[gene] == i,
                }
                for j in range(self.nfolds):
                    row["score%i" % j] = self.scores[gene][i][j]
                for j in range(10):
                    try:
                        row["feature%i" % j] = self.important_features[gene][i].index[j]
                        row["feature%i_importance" % j] = self.important_features[gene][
                            i
                        ].iloc[j]
                    except IndexError:
                        row["feature%i" % j] = np.nan
                        row["feature%i_importance" % j] = np.nan
                melted = melted.append(row, ignore_index=True)
        return melted

    def save_results(self, feat_outfile, pred_outfile):
        melted = self.format_results()
        melted.to_csv(feat_outfile, index=None)

        # This all outputs to a single file. Should this be multiple files, or is there
        # only one model type/predictions?
        for model, pred in zip(self.model_types, self.predictions):
            pred.to_csv(pred_outfile, index_label="Row.name")


##############################################################
###################  O T H E R  ##############################
##############################################################


def gene_feature_filter(df, gene_name):
    genes = [x.split("_")[0].split(" ")[0] for x in df.columns]
    mask = np.array([s == gene_name for s in genes], dtype=np.bool)
    return mask


class SelfFeatureForest(RandomForestRegressor):
    """
    Uses only features matching ("like") the target (+ `reserved_columns` which are always included).
    Uses the target's name with everything after the first space stripped off to match columns in the feature set.
    A custom target name can be passed with the call to "fit" to be used for finding related features.
    """

    def __init__(self, reserved_columns=[], **kwargs):
        self.reserved_columns = reserved_columns
        RandomForestRegressor.__init__(self, **kwargs)
        self.feature_names = []

    def fit(self, X, y, name=None, **kwargs):
        if name is None:
            name = y.name
        self.name = name
        mask = X.columns.isin(self.reserved_columns)
        mask = mask | gene_feature_filter(X, self.name.split(" ")[0])
        self.feature_names = X.columns[mask].tolist()
        features = X.loc[:, self.feature_names]
        RandomForestRegressor.fit(
            self, features.values, y.loc[features.index].values, **kwargs
        )
        # RandomForestRegressor.fit(self, features.as_matrix(), y.loc[features.index].values, **kwargs)

    def predict(self, X, **kwargs):
        features = X.loc[:, self.feature_names]
        return RandomForestRegressor.predict(self, features.values, **kwargs)
        # return RandomForestRegressor.predict(self, features.as_matrix(), **kwargs)

    def get_feature_series(self, n_features=None):
        if n_features is None:
            n_features = len(self.feature_names)
        imp = pd.Series(self.feature_importances_, index=self.feature_names)
        return imp.sort_values(ascending=False)[:n_features]


class RelatedFeatureForest(SelfFeatureForest):
    """
    Uses a two-column list of related features to select only features related to the target (+ `reserved_columns` which are always included).
    Uses the target's name with everything after the first space stripped off to match to related features.
    A custom target name can be passed with the call to "fit" to be used for finding related features.
    """

    def __init__(
        self,
        relations: pd.DataFrame,
        feature_metadata: pd.DataFrame,
        reserved_columns: Optional[List[str]] = None,
        **kwargs
    ):
        SelfFeatureForest.__init__(self, reserved_columns, **kwargs)
        self.relations = relations
        self.feature_metadata = feature_metadata

    def fit(self, X, y, name: Optional[str] = None, **kwargs):
        if name is None:
            name = y.name
        self.name = name

        _, target_entrez_id = split_gene_label_str(self.name)
        related_entrez_ids = set(
            self.relations["partner_entrez_id"][
                self.relations["target_entrez_id"] == target_entrez_id
            ]
        )
        related_entrez_ids.add(target_entrez_id)

        related_features = self.feature_metadata["feature_id"][
            self.feature_metadata["entrez_id"].isin(related_entrez_ids)
        ].tolist()

        columns = (
            related_features + self.reserved_columns
            if self.reserved_columns is not None
            else related_features
        )
        self.feature_names = columns
        features = X[columns]

        RandomForestRegressor.fit(
            self, features, y.loc[features.index].values, **kwargs
        )


class KFilteredForest(RandomForestRegressor):
    """
    Selects the top `k` features with highest correlation with the target and variance greater than `var_threshold`
    """

    def __init__(self, k=1000, var_threshold=0, **kwargs):
        self.k = k
        RandomForestRegressor.__init__(self, **kwargs)
        self.filter = SelectKBest(score_func=f_regression, k=k)
        self.var_threshold = var_threshold

    def fit(self, X, y, **kwargs):
        self.mask1 = varfilter(X, self.var_threshold)
        x = X.loc[:, X.columns[self.mask1]]
        if x.shape[1] > self.k:
            self.filter.fit(x, np.array(y))
            self.mask2 = self.filter.get_support()
            x = x.loc[:, x.columns[self.mask2]]
        self.feature_names = x.columns.tolist()
        RandomForestRegressor.fit(self, x.values, y, **kwargs)
        # RandomForestRegressor.fit(self, x.as_matrix(), y, **kwargs)

    def predict(self, X, **kwargs):
        x = X.loc[:, self.feature_names]
        return RandomForestRegressor.predict(self, x.values, **kwargs)
        # return RandomForestRegressor.predict(self, x.as_matrix(), **kwargs)

    def get_feature_series(self, n_features):
        if n_features is None:
            n_features = len(self.feature_names)
        imp = pd.Series(self.feature_importances_, index=self.feature_names)
        return imp.sort_values(ascending=False)[:n_features]

class BogusEnsemble:
    def __init__(self, targets, samples, model, top_n, nfolds):
        self.nfolds = nfolds
        self.model = model
        self.top_n = top_n
        self.targets = targets
        self.samples = samples

    def save_results(self, feature_file_path, predictions_file_path):
        # make the features_file
        # sample:
        # gene,model,score0,score1,score2,score3,score4,best,feature0,feature0_importance,feature1,feature1_importance,feature2,feature2_importance,feature3,feature3_importance,feature4,feature4_importance,feature5,feature5_importance,feature6,feature6_importance,feature7,feature7_importance,feature8,feature8_importance,feature9,feature9_importance
        # RGS2 (5997),Related,0.06471196187331774,0.1146279381673232,0.1086815442125918,-0.014651457705020965,0.02726712572331791,True,CasActivity_crispr_confounders,0.06187336389318614,ScreenFPR_crispr_confounders,0.04370822804451761,ScreenMedianEssentialDepletion_crispr_confounders,0.04282315156351761,ScreenNNMD_crispr_confounders,0.025410613940554865,ScreenROCAUC_crispr_confounders,0.021000058979961267,ScreenMADNonessentials_crispr_confounders,0.01891194956566079,MON1A_(84315)_RNAseq,0.01332746956748942,ScreenMADEssentials_crispr_confounders,0.011934924192498666,MON1A_(84315)_CN,0.011929959401096441,CHD3_(1107)_RNAseq,0.011822485364727377

        features_table_cols = ["gene", "model"]
        for i in range(self.nfolds):
            features_table_cols.append(f"score{i}")
        features_table_cols.append("best")
        for i in range(self.top_n):
            features_table_cols.extend([f"feature{i}", f"feature{i}_importance"])
        rows = []
        for target in self.targets:
            row = {col: pd.NA for col in features_table_cols}
            row["best"] = "False"
            row["gene"] = target
            row["model"] = self.model
            rows.append(row)
        df = pd.DataFrame(rows, columns=features_table_cols)
        df.to_csv(feature_file_path, index=False)

        # make the predictions file
        # sample: Row.name,RGS2 (5997),RGS20 (8601),RGS21 (431704),RGS22 (26166),RGS3 (5998),RGS4 (5999),RGS5 (8490),RGS6 (9628),RGS7 (6000),RGS7BP (401190)
        # ACH-000208,-0.13226353896347068,0.010234398971580492,-0.01062930635446206,0.062382314447098214,-0.0014926357860092733,0.03907420537432578,0.16500149683242857,-0.008528366762044833,0.009945912533090833,0.0067523898875571755
        # ACH-000381,-0.06529517223163701,0.009006151985379358,0.001709141434118836,-0.033790842006468466,-0.008362480948481315,0.1316203055955582,0.1105027989178931,-0.04808614467392679,0.07743513404641876,-0.013084708152628658
        predictions_cols = ["Row.name"] + self.targets
        rows = []
        for sample in self.samples:
            row = {col: pd.NA for col in predictions_cols}
            row["Row.name"] = sample
            rows.append(row)
        df = pd.DataFrame(rows, columns=predictions_cols)
        df.to_csv(predictions_file_path, index=False)



class PandasForest(RandomForestRegressor):
    """
    A simple wrapper for RandomForestRegressor that plays nice with dataframes and series instead of numpy arrays
    """

    def fit(self, X, y, **kwargs):
        self.feature_names = X.columns.tolist()
        RandomForestRegressor.fit(self, X, y, **kwargs)

    def get_feature_series(self, n_features):
        if n_features is None:
            n_features = len(self.feature_names)
        imp = pd.Series(self.feature_importances_, index=self.feature_names)
        return imp.sort_values(ascending=False)[:n_features]


def run_model(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    model: ModelConfig,
    nfolds: int,
    task="regress",
    relation_table=None,
    feature_metadata=None,
) -> EnsembleRegressor:
    """Fit models for specified columns of Y using a selection of feature subsets from X.

    Function parameters should be documented in the ``Args`` section. The name
    of each parameter is required. The type and description of each parameter
    is optional, but should be included if not obvious.

    Args:
        param1 (int): The first parameter.
        param2 (:obj:`str`, optional): The second parameter. Defaults to None.
            Second line of description should be indented.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if successful, False otherwise.

        The return type is optional and may be specified at the beginning of
        the ``Returns`` section followed by a colon.

        The ``Returns`` section may span multiple lines and paragraphs.
        Following lines should be indented to match the first line.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    Raises:
        AttributeError: The ``Raises`` section is a list of all exceptions
            that are relevant to the interface.
        ValueError: If `param2` is equal to `param1`.

    """
    print("aligning features")
    shared_lines = list(set(X.index) & set(Y.index))
    assert len(shared_lines) > 0, "no shared lines found: \n\n features %r\n\n "

    Y = Y.loc[shared_lines]
    X = X.loc[shared_lines]
    print("Number of shared cell lines: " + str(len(shared_lines)))
    if len(shared_lines) < 10:
        print("Warning: tiny number of samples. Returning bogus result to avoid crashing")
        return BogusEnsemble(Y.columns.to_list(), X.index.to_list(), model.name, 10, nfolds)


    if model.exempt is not None:
        constant_features = [
            s
            for s in X.columns
            if any(s.endswith(re.sub(r"[\s-]", "_", end)) for end in model.exempt)
        ]
    else:
        constant_features = []

    ModelInputs = TypedDict(
        "ModelInputs",
        {
            "Name": str,
            "ModelClass": Optional[RandomForestRegressor],
            "kwargs": Optional[Dict[str, Any]],
        },
        total=False,
    )
    new_model: ModelInputs = {"Name": model.name}

    if (model.relation == "All") and (X.shape[1] <= 1000):
        new_model["ModelClass"] = PandasForest
        new_model["kwargs"] = dict(max_depth=8, n_estimators=100, min_samples_leaf=5)
    if (model.relation == "All") and (X.shape[1] > 1000):
        new_model["ModelClass"] = KFilteredForest
        new_model["kwargs"] = dict(max_depth=8, n_estimators=100, min_samples_leaf=5)
    elif model.relation == "MatchTarget":
        new_model["ModelClass"] = SelfFeatureForest
        new_model["kwargs"] = dict(
            reserved_columns=constant_features,
            max_depth=8,
            n_estimators=100,
            min_samples_leaf=5,
        )
    elif model.relation == "MatchRelated":
        new_model["ModelClass"] = RelatedFeatureForest
        new_model["kwargs"] = dict(
            reserved_columns=constant_features,
            relations=relation_table,
            feature_metadata=feature_metadata,
            max_depth=8,
            n_estimators=100,
            min_samples_leaf=5,
        )

    models = [new_model]

    if len(X) != len(Y):
        raise RuntimeError(
            "length of X and Y do not match (shapes %r and %r)" % (X.shape, Y.shape)
        )
    Xtrain = [X]
    assert len(Xtrain) == len(models), (
        "number of models %i does not match number of feature sets %i"
        % (len(models), len(Xtrain))
    )
    for i, x in enumerate(Xtrain):
        assert x.shape[1] > 0, "feature set %i does not have any columns" % i
        assert all(
            x.index == Y.index
        ), "feature set %i index does not match Y index\n\n%r" % (i, x.iloc[:5, :5])

    print("creating TDA ensemble")
    if task == "classify":
        ensemble = EnsembleRegressor(
            model_types=models,
            nfolds=nfolds,
            rounding=True,
            Splitter=StratifiedKFold,
            scoring=soft_roc_auc,
        )
    elif task == "regress":
        ensemble = EnsembleRegressor(
            model_types=models,
            nfolds=nfolds,
            rounding=False,
            Splitter=QuantileKFold,
            scoring=r2_score,
        )
    else:
        raise ValueError('task must be "classify" or "regress"')
    ensemble.fit(X=Xtrain, Y=Y)
    print("Finished fitting")

    return ensemble
