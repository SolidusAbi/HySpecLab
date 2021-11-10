import numpy as np
import torch

from .InterbandRedundancySelector import InterbandRedundancySelector 
from torch.nn.functional import one_hot
from imblearn.under_sampling import NearMiss

from IPDL.functional import matrix_estimator
from IPDL.InformationTheory import MatrixBasedRenyisEntropy as renyis

class InterbandRedundacyMutualInformationSelector(InterbandRedundancySelector):
    '''
        Feature selector that removes the features with high-multicollinearity in 
        adjacent features based on Variational Inflation Factor (VIF). Between the 
        adjacent features with high-multicollinearity, are chosen those which Mutual
        Information (MI) is bigger.

        The MI estimation is based on kernels, using Radial Basis Function (RBF), and
        compare the MI between the feature X^f and target y. The optimal sigma value
        is estimated based on Silverman's Rule. This selector has to be used such as
        supervised learning.

        Parameters
        ----------
        threshold : float, default=5.0
            Features with a VIF bigger than this threshold will be removed. The default
            is to keep all features with a VIF lower than 5, i.e. remove the features that
            have a low-multicollinearity with the adjacent features.

        undersampling : int, default=256
            Number of samples per class in order to optimize the MI computation. This value
            has to be empirically tested for your dataset. The algorithm applied for undersampling
            is NearMiss3 from Imbalanced-Learn.

        gamma : float, default=0.5
            Gamma value for Silverman's rule used to estimate the optimal sigma value for RBF kernel.

        Attributes
        ----------
        vif_ : array, shape (n_features, n_features)
            VIF computed.

        mi_estimation_ : array, shape(n_fetures,)
            Kernel-Based MI estimation.

        mask_ : array, shape (n_features,)
            1-D array which contains the selected features.
    '''
    def __init__(self, threshold=5, undersampling=256, gamma=.5):
        super(InterbandRedundacyMutualInformationSelector, self).__init__(threshold)
        self.undersampling = undersampling
        self.gamma = gamma

    def fit(self, X, y):
        '''
            Run score function on (X, y) and get the appropriate features.

            Parameters
            ----------
            X : array, shape (n_samples, n_features)
                Data from which to compute the VIF between different features, where `n_samples` is
                the number of samples and `n_features` is the number of features.

            y : array, shape (n_samples, ) 
                The target values (class labels in classification, real numbers in
                regression).

            Returns
            -------
            self : object
                Returns the instance itself.
        '''
        n_features = X.shape[1]

        self.vif_ = self._vif_estimation(X)
        self.mi_ = self._mi_estimation(X, y)

        features_available = np.arange(0, n_features)
        self.mask_ = np.zeros(features_available.size, dtype=np.uint)

        features_selected_idx = np.linspace(0, features_available.size, 5, dtype=np.uint)[1:-1]
        while(features_available.size > 0):
            features_selected = features_available[features_selected_idx]
            
            features_high_collinear = tuple(map(lambda x: self._clusterize(x, features_available), features_selected))
            #Update the features selected based on MI estimation
            features_selected = [cluster[np.argmax(self.mi_[cluster])] for cluster in features_high_collinear]
            self.mask_[features_selected] = 1
            
            features_high_collinear = np.unique(np.concatenate(features_high_collinear))

            # Remove features selected and those which have high-multicollinearity
            features_available = features_available[ np.logical_not(np.in1d(features_available, features_high_collinear, assume_unique=True)) ]

            features_selected_idx = np.linspace(0, features_available.size, 5, dtype=np.uint)[1:-1]

        return self

    def _mi_estimation(self, X, y):
        '''
            Kernel-based Mutual Information estimation. Using the implementation from
            IPDL library based on PyTorch.

            Parameters
            ----------
                X : array, shape (n_samples, n_features)
                    Data from which to compute the VIF between different features, where `n_samples` is
                    the number of samples and `n_features` is the number of features.

                y : array, shape (n_samples, ) 
                    The target values (class labels in classification, real numbers in
                    regression).
        '''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        sampling_strategy = {class_id:self.undersampling for class_id in np.unique(y)}
        undersampler = NearMiss(sampling_strategy=sampling_strategy, version=3)
        x_resampled, y_resampled = undersampler.fit_resample(X, y)

        x_tensor = torch.tensor(x_resampled)
        y_tensor = one_hot(torch.tensor(y_resampled))

        n_features = x_tensor.size(1)
        
        sigma_y = self._silverman(y_tensor.size(0), y_tensor.size(1)) 
        _, Ay = matrix_estimator(y_tensor.to(device), sigma=sigma_y)

        mi_estimation = np.zeros(x_tensor.size(1))
        sigma_x = self._silverman(y_tensor.size(0), 1) 
        for feature in range(n_features):
            Kx, Ax = matrix_estimator(x_tensor[:, feature].reshape(-1, 1).to(device), sigma=sigma_x)
            mi_estimation[feature] = renyis.mutualInformation(Ax, Ay).cpu()

        return mi_estimation

    def _silverman(self, n, d):
        '''
            Sigma estimation for RBF kernel using Silverman's rule.

            Parameters
            ----------
                n : int, 
                    Number of samples.

                d : int, 
                    Dimensionality of the samples.
        '''
        return self.gamma * (n ** (-1 / (4 + d))) 