import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

class Norm1Transform(TransformerMixin, BaseEstimator):
    '''
        X is noramlized in order to have 1-Normed vector space, i.e, the samples
        are normalized in order the Norm ||X|| is 1.
    '''
    def __init__(self, n_features_to_select=None):
        super(Norm1Transform, self).__init__()

    def fit(self, X, y=None):
        '''
            Parameters
            ----------
            X : array, shape (n_samples, n_features)
                Data from which to compute the Norm which will be used in order to generate a 1-normed vector space.

            y : any, default=None
                Ignored. This parameter exists only for compatibility with
                sklearn.pipeline.Pipeline.

            Returns
            -------
            self : object
                Returns the instance itself.
        '''
        self.norms_ = np.sqrt(np.sum(X**2, axis=1))
        return self

    def transform(self, X, y=None, **kwargs):
        check_is_fitted(self)
        return X / self.norms_[:,np.newaxis]