################################################################
#
# Robust Synthetic Control
#
# Implementation based on:
# http://www.jmlr.org/papers/volume19/17-777/17-777.pdf
#
################################################################
import numpy as np
import pandas as pd

from sklearn.exceptions import NotFittedError

from tslib.src.models.tsSVDModel import SVDModel
from tslib.src.models.tsALSModel import ALSModel
from tslib.src import tsUtils


class RobustSyntheticControl(object):

    # seriesToPredictKey:       (string) the series of interest (key)
    # kSingularValuesToKeep:    (int) the number of singular values to retain
    # M:                        (int) the number of columns for the matrix
    # probObservation:          (float) the independent probability of observation of each entry in the matrix
    # modelType:                (string) SVD or ALS. Default is "SVD"
    # svdMethod:                (string) the SVD method to use (optional)
    # otherSeriesKeysArray:     (array) an array of keys for other series which will be used to predict

    def __init__(
        self,
        seriesToPredictKey=None,
        kSingularValues=5,
        p=1.0,
        modelType="svd",
        svdMethod="numpy",
        otherSeriesKeysArray=[],
    ):
        self.seriesToPredictKey = seriesToPredictKey
        self.kSingularValues = kSingularValues
        self.p = p
        self.modelType = modelType
        self.svdMethod = svdMethod
        self.otherSeriesKeysArray = otherSeriesKeysArray

        self.N = 1  # each series is on its own row
        self.model = None
        self.control = None  # these are the synthetic control weights

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return dict(
            seriesToPredictKey=self.seriesToPredictKey,
            kSingularValues=self.kSingularValues,
            p=self.p,
            modelType=self.modelType,
            svdMethod=self.svdMethod,
            otherSeriesKeysArray=self.otherSeriesKeysArray,
        )

    def create_model(self, X_train):
        M = len(X_train)
        if self.modelType == "als":
            self.model = ALSModel(
                self.seriesToPredictKey,
                self.kSingularValues,
                self.N,
                M,
                probObservation=self.p,
                otherSeriesKeysArray=self.otherSeriesKeysArray,
                includePastDataOnly=False,
            )
        else:  # default: SVD
            self.model = SVDModel(
                self.seriesToPredictKey,
                self.kSingularValues,
                self.N,
                M,
                probObservation=self.p,
                svdMethod="numpy",
                otherSeriesKeysArray=self.otherSeriesKeysArray,
                includePastDataOnly=False,
            )

    # X_train: (Pandas dataframe) a key-value Series
    # y_train is ignored but required for compatibility with some sklearn methods
    # Note that the keys provided in the constructor MUST all be present
    # The values must be all numpy arrays of floats.
    def fit(self, X_train, y_train=None):
        # Model must be (re)created here for compatibility with scikit API
        self.create_model(X_train)
        self.model.fit(X_train)

    # otherKeysToSeriesDFNew:     (Pandas dataframe) needs to contain all keys provided in the model;
    #                               all series/array MUST be of length >= 1,
    #                               If longer than 1, then the most recent point will be used (for each series)
    def predict(self, otherKeysToSeriesDFNew):
        if self.model is None:
            raise NotFittedError("Cannot call predict() before fit()")
        prediction = np.dot(
            self.model.weights, otherKeysToSeriesDFNew[self.otherSeriesKeysArray].T
        )
        return prediction

    def rmse(self, y_pred, y_true):
        return np.sqrt(np.mean((y_pred - y_true) ** 2))

    # Score = -RMSE
    def score(self, X, y_true=None):
        if y_true is None:
            y_true = X[self.seriesToPredictKey]
        y_pred = self.predict(X)
        return -self.rmse(y_pred, y_true)

    # return the synthetic control weights
    def getControl(self):
        if self.model is None:
            raise NotFittedError("Cannot get model weights before calling fit()")
        else:
            return self.model.weights
