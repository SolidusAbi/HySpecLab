import numpy as np
from imblearn.under_sampling.base import BaseUnderSampler
# from imblearn.utils import Substitution
# from imblearn.utils._docstring import _n_jobs_docstring
from spectral.algorithms import spectral_angles
from sklearn.cluster import KMeans
from sklearn.utils import _safe_indexing
from spectral.algorithms import spectral_angles

def spectral_angles_pixel(x, ref):
    '''
        For pixel input, the original function is prepare for image
    '''
    return spectral_angles(x[np.newaxis,:], ref)[0]

# @Substitution(
#     sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
#     n_jobs=_n_jobs_docstring,
# )
## Por ahora no usar!!! Está en pruebas!!

class HyperSpectralUnderSampler(BaseUnderSampler):
    ''' 
        Class to perform HyperSpectral data under-sampling.

        Under-sample the different class(es) by K-Mean unsupervised clustering approach. The K-Means clustering 
        is applied independently to each group of labeled pixels in order to obtain K clusters per group. In order 
        to reduce the original training dataset, such centroids are employed to identify the most representative pixels of
        each class by using the Spectral Angle [2] algorithm. For each cluster centroid, only the S most similar
        samples are selected.

        Parameters
        ----------
        n_clusters: int, default=100
            The number of centroids used in K-Mean clustering (K).
        
        samples_per_class: int, default=10
            The number of most similiar signals to select (S)

        {random_state}

        References
        ----------
          [1] Martinez, B., Leon, R., Fabelo, H., Ortega, S., Piñeiro, J. F., Szolna, A., ... & M Callico, G. (2019). Most
          relevant spectral bands identification for brain cancer detection using hyperspectral imaging.Sensors, 19(24), 5481.
          
          [2] Rashmi, S.; Addamani, S.; Ravikiran, A. Spectral Angle Mapper algorithm for remote sensing image classification. 
          IJISET Int. J. Innov. Sci. Eng. Technol. 2014, 1, 201–20
    '''
    def __init__(self, *, sampling_strategy="auto", n_clusters=100, samples_per_cluster=10, random_state=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = n_clusters
        self.S = samples_per_cluster
        self.random_state = random_state

    def _spectral_angles_pixel(self, x, ref):
        '''
            For pixel input, the original function is prepare for image
        '''
        return spectral_angles(x[np.newaxis,:], ref)[0]

    def get_most_relevant_samples_idx(self, x, labels, centroids, n_samples_per_centroid=10):
        if len(x.shape) != 2:
            assert 'X shape error!'

        if x.shape != labels.shape:
            assert 'x and labels do not match in dimension'

        if np.unique(labels).size != centroids.shape[0]:
            assert 'The number of labels and centroids does not match'
        
        sam_result = spectral_angles_pixel(x, centroids)
        selected_samples_idx = np.array([], dtype=np.uint)

        for label in np.unique(labels):
            cluster_samples_idx = np.argwhere(labels==label).flatten()
            n_samples_per_centroid = n_samples_per_centroid if cluster_samples_idx.size >= n_samples_per_centroid else cluster_samples_idx.size
            ind = np.argpartition(sam_result[cluster_samples_idx, label], -n_samples_per_centroid)[-n_samples_per_centroid:].astype(np.uint)
            selected_samples_idx = np.concatenate((selected_samples_idx, cluster_samples_idx[ind])).astype(np.uint)

        return selected_samples_idx

    def _fit_resample(self, X, y):
        class_label = np.unique(y)
        selected_samples_idx = np.array([], dtype=np.uint)
        for i in class_label:
            class_samples_idx = np.argwhere(y==i).flatten()
            _x = X[class_samples_idx]
            kmeans = KMeans(n_clusters=self.K, random_state=self.random_state).fit(_x)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_

            selected_samples_class_idx = self.get_most_relevant_samples_idx(_x, labels, centroids, self.S)
            selected_samples_idx = np.concatenate((selected_samples_idx, class_samples_idx[selected_samples_class_idx])).astype(np.uint)

        self.sample_indices_ = selected_samples_idx

        return _safe_indexing(X, self.sample_indices_), _safe_indexing(y, self.sample_indices_)