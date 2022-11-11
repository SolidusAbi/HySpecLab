from sklearn.utils.validation import check_is_fitted
from imblearn.under_sampling import NearMiss
import numpy as np
from . import EEA

from SVDD import BaseSVDD

class SVDD(EEA):
    '''
        Support Vector Data Descriptor (SVDD) to extract a set of endmembers 
        (elementary spectrum) from a given hyperspectral image.

        This implementation uses Radial Basis Function (RBF) as kernel in SVDD. 
        The sigma controls how tightly the SVDD models the support boundaries.
    '''

    def __init__(self, sigma: float, d:int, n_samples: int, C=100.0) -> None:
        '''
            Parameters
            ----------
                sigma: float
                    Kernel coefficient for RBF kernel. This parameters controls 
                    how tightly the SVDD models the support boundaries.
                
                d: int
                    Projection to a d-subspace

                n_samples: int
                    The number of samples which will be chosen from original X.
                    This undersampling is applied using NearMiss 1 in order to obtain
                    the maximum volume from Hyperspectral data.

                C: float
                    Regularization parameter. The strength of the regularization is
                    inversely proportional to C. Must be strictly positive. The penalty
                    is a squared l2 penalty

                
        '''
        super(SVDD, self).__init__(0)
        self.sigma = sigma
        self.n_samples = n_samples
        self.d = d
        self.C = C

    def fit(self, X, y):
        '''
            Parameters
            ----------
        '''

        sampling_strategy = {class_id:self.n_samples for class_id in np.unique(y)}
        nm = NearMiss(sampling_strategy=sampling_strategy, version=1)

        X_resampled, _ = nm.fit_resample(X, y)

        Xd, Ud = self._proj_subspace(X_resampled.T, self.d)

        svdd = BaseSVDD(C=self.C, gamma=self.sigma, kernel='rbf', display='off')
        svdd.fit(Xd.T)

        # svdd.plot_boundary(Xd.T)

        Xp = np.dot(Ud,Xd[:self.d,:]).T # Xd projected into the original space
        self.endmembers_ = np.clip(Xp[svdd.support_vector_indices], a_min=0, a_max=1) # Force to be sure that values goes from 0 to 1
        self.idx_ = svdd.support_vector_indices        

    def endmembers(self):
        '''
            Return the endmembers estimated from X.
        '''
        check_is_fitted(self)
        return self.endmembers_ # Check