import gc
import json
import math
import os
import random
from itertools import chain
from time import time
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, make_scorer, r2_score
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    mutual_info_regression,
    f_regression,
)
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr

from .data_models import ModelConfig


def filter_run_ensemble_inputs(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    valid_samples: Optional[pd.Series],
    feature_subset: Optional[pd.Series],
    target_range: Optional[Tuple[int, int]],
    targets: Optional[List[str]],
):
    # TODO
    if target_range is None:
        start_col = 0
        end_col = Y.shape[1]
    else:
        if targets is not None:
            raise ValueError("Only one of target_range and targets can be specified")
        start_col, end_col = target_range
        Y = Y.filter(items=Y.columns[start_col:end_col], axis="columns")

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
        n = 0
        model_prediction = pd.Series(np.nan, index=Y.index, name=column)
        for train, test in splits:
            try:
                model.fit(x.iloc[train], y.iloc[train])
                ypred = model.predict(x.iloc[test])
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
            n += 1
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
        self.columns = None
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

    def fit(self, X, Y, columns=None, report_freq=20):
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
        if columns is None:
            columns = Y.columns
        self.columns = Y.columns
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

    def save_results(self, feat_outfile, pred_outfile):
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
        melted.to_csv(feat_outfile, index=None)
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
        reserved_columns=[],
        relations=pd.DataFrame(columns=["target", "partner"]),
        **kwargs
    ):
        SelfFeatureForest.__init__(self, reserved_columns, **kwargs)
        self.relations = relations

    def fit(self, X, y, name=None, **kwargs):
        if name is None:
            name = y.name
        self.name = name
        self.related = self.relations.partner[
            self.relations.target == name.split(" ")[0]
        ].tolist()
        self.related = list(set(self.related))
        if not self.name in self.related:
            self.related.append(self.name)
        mask = X.columns.isin(self.reserved_columns)
        for partner in self.related:
            mask = mask | gene_feature_filter(X, partner)
        self.feature_names = X.columns[mask]
        features = X[self.feature_names]
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
    X,
    Y,
    model: ModelConfig,
    nfolds,
    start_col=0,
    end_col=None,
    task="regress",
    relation_table=None,
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

    if end_col is None:
        end_col = Y.shape[1]
    print("aligning features")
    shared_lines = list(set(X.index) & set(Y.index))
    assert len(shared_lines) > 0, "no shared lines found: \n\n features %r\n\n "
    Y = Y.loc[shared_lines]
    X = X.loc[shared_lines]
    print("Number of shared cell lines: " + str(len(shared_lines)))

    if model.exempt is not None:
        constant_features = [
            s for s in X.columns if any(s.endswith(end) for end in model.exempt)
        ]
    else:
        constant_features = []

    new_model = {"Name": model.name}

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
            relations=relation_table[["target", "partner"]],
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

    columns = Y.columns[start_col:end_col]
    ensemble.fit(Xtrain, Y, columns)
    print("Finished fitting")

    return ensemble


def run_ensemble(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    model_config: ModelConfig,
    task_mode: str,
    nfolds: int,
    target_range: Tuple[int, int],
    related_table: Optional[pd.DataFrame],
):
    return run_model(
        X,
        Y,
        model=model_config,
        nfolds=nfolds,
        task=task_mode,
        relation_table=related_table,
        start_col=target_range[0],
        end_col=target_range[1],
    )
