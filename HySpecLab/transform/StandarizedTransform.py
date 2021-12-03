
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

class StandarizedTransform(TransformerMixin, BaseEstimator):
    '''
        Z-Score normalization, aka Standarized normalization, is a common strategy of 
        normalizing data that avoids this outlier issue.
    '''
    def __init__(self):
        super(StandarizedTransform, self).__init__()

    def fit(self, X, y=None):
        '''
            Parameters
            ----------
            X : array, shape (n_samples, n_features)
                Data from which to calculate the mean and std that will be used
                to generate a standardized data. 

            y : any, default=None
                Ignored. This parameter exists only for compatibility with
                sklearn.pipeline.Pipeline.

            Returns
            -------
            self : object
                Returns the instance itself.
        '''
        self.means_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X, y=None, **kwargs):
        check_is_fitted(self)
        return (X - self.means_) / self.std_

    def inverse_transform(self, X):
        std_inv = 1 / (self.std_ + 1e-16)
        mean_inv = -self.means_ * std_inv

        return (X - mean_inv) / std_inv