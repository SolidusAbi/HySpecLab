import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEquidistantSelection(TransformerMixin, BaseEstimator):
    def __init__(self, n_features_to_select=None):
        self.n_features_to_select = n_features_to_select
        self.selected_features_idx = []

    def get_params(self, deep=True):
        return {"n_features_to_select": self.n_features_to_select}       

    def fit(self, X, y=None):
        n_features = X.shape[1]
        self.selected_features_idx = np.linspace(0, n_features-1, self.n_features_to_select, dtype=int)
        return self

    def transform(self, X, y=None, **kwargs):
        return X[:, self.selected_features_idx]


class FeatureSelection(TransformerMixin, BaseEstimator):
    def __init__(self, n_features=None, selected_features=None):
        self.n_features = n_features
        self.selected_features = selected_features

    # def get_params(self, deep=True):
    #     params = dict()
    #     for feature in list(self.__dict__.keys()):
    #         params[feature] = getattr(self, feature)
            
    #     return params

    # def set_params(self, **parameters):
    #     for parameter, value in parameters.items():
    #         setattr(self, parameter, value)
    #     return self

    def fit(self, X, y=None):
        self.n_features = X.shape[1]
        # self.selected_features_idx = np.linspace(0, n_features-1, self.n_features_to_select, dtype=int)
        return self

    def transform(self, X, y=None, **kwargs):
        if self.selected_features:
            feature_idx = format(self.selected_features, "b").zfill(self.n_features)
            feature_idx = np.array(list(map(int, feature_idx)), dtype=np.bool_)
            return X[:, feature_idx]
        
        return X