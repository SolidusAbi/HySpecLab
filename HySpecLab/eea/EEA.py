from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
import scipy.linalg as splin
import numpy as np



class EEA(ABC, BaseEstimator):
    def __init__(self, n_endmembers:int):
        super(EEA, self).__init__()
        self.n_endmembers = n_endmembers

    @abstractmethod
    def endmembers(self):
        pass

    def _proj_subspace(self, X, d):
        '''
            Project data onto a subspace d. This method uses SVD. SVD and PCA are
            also equal in the case of zero-mean data.

            Params
            -----
            X : array, shape (n_features, n_samples)
                Data that will be projected in the subspace d.

            d : int
                Space where the zero-mean data will be projected.

            Return
            ------
                (Data projected onto subspace d, Tranform matrix which project data onto d-space).
        '''
        _, N = X.shape
        Ud  = splin.svd(np.dot(X,X.T)/float(N))[0][:,:d]    # computes the d-projection matrix 
        return (np.dot(Ud.T, X), Ud)
