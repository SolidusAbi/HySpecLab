from .InterbandRedundancySelector import InterbandRedundancySelector 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

class InterbandRedundacyAUCSelector(InterbandRedundancySelector):
    '''
        Feature selector that removes the features with high-multicollinearity in 
        adjacent features based on Variational Inflation Factor (VIF). Between the 
        adjacent features with high-multicollinearity, are chosen those which Area 
        Under ROC Curve (AUC) is bigger.

        Parameters
        ----------
        threshold : float, default=5.0
            Features with a VIF bigger than this threshold will be removed. The default
            is to keep all features with a VIF lower than 5, i.e. remove the features that
            have a low-multicollinearity with the adjacent features.

        Attributes
        ----------
        vif_ : array, shape (n_features, n_features)
            VIF computed.

        auc_estimation_ : array, shape(n_fetures,)
            Kernel-Based MI estimation.

        mask_ : array, shape (n_features,)
            1-D array which contains the selected features.
    '''
    def __init__(self, threshold=5):
        super(InterbandRedundacyAUCSelector, self).__init__(threshold)

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
        self.auc_ = self._roc_auc_estimation(X, y)

        features_available = np.arange(0, n_features)
        self.mask_ = np.zeros(features_available.size, dtype=np.uint)

        features_selected_idx = np.linspace(0, features_available.size, 5, dtype=np.uint)[1:-1]
        while(features_available.size > 0):
            features_selected = features_available[features_selected_idx]
            
            features_high_collinear = tuple(map(lambda x: self._clusterize(x, features_available), features_selected))
            #Update the features selected based on AUC estimation
            features_selected = [cluster[np.argmax(self.auc_[cluster])] for cluster in features_high_collinear]
            print(features_selected)
            self.mask_[features_selected] = 1
            
            features_high_collinear = np.unique(np.concatenate(features_high_collinear))

            # Remove features selected and those which have high-multicollinearity
            features_available = features_available[ np.logical_not(np.in1d(features_available, features_high_collinear, assume_unique=True)) ]

            features_selected_idx = np.linspace(0, features_available.size, 5, dtype=np.uint)[1:-1]

        return self

    def _roc_auc_estimation(self, X, y):
        '''
            Compute the Area Under ROC Curve (AUC) per feature.

            Parameters
            ----------
            X : array, shape (n_samples, n_features)
                Data from which to compute the AUC between different features, where `n_samples` is
                the number of samples and `n_features` is the number of features.

            y : array, shape (n_samples, ) 
                The target values (class labels in classification, real numbers in
                regression).

            Returns
            -------
            roc_auc : array, shape (n_features, )
                AUC computed between the features and the target.
        '''
        _, d = X.shape
        roc_auc = np.zeros(d)
        for feature in range(d):
            _X = np.expand_dims(X[:, feature], axis=1)
            clf = LogisticRegression(solver="liblinear", random_state=0).fit(_X, y)
            roc_auc[feature] = roc_auc_score(y, clf.predict_proba(_X), multi_class='ovr')
        return roc_auc