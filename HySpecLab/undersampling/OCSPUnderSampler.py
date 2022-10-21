from imblearn.under_sampling.base import BaseUnderSampler
from sklearn.utils import _safe_indexing
import numpy as np
import torch

class OCSPUnderSampler(BaseUnderSampler):
    '''
        Orthogonal Complement Subspace Projection (OCSP) to undersample 
        the HyperSpectral dataset. The OCSP will be used to select samples from large-size classes
        as proposed in the original paper

        References
        ----------
            [1] Li, J., Du, Q., Li, Y., & Li, W. (2018). Hyperspectral image classification with imbalanced
            data based on orthogonal complement subspace projection. IEEE Transactions on Geoscience and Remote
            Sensing, 56(7), 3838-3851.

        Parameters
        ----------
            n_samples: int
                Number of samples to select from each class. The number of samples
                have to be bigger than 1000.
            
            random_state: int
                Random state to use for reproducibility of results (default: None)
            
            cuda: bool
                Whether to use GPU or not (default: False)
    '''
    def __init__(self, n_samples, random_state=None, cuda=False):
        super().__init__()

        if n_samples < 1000:
            raise ValueError('n_samples must be greater than 1000')

        self.n_samples = n_samples
        self.random_state = random_state
        self.cuda = cuda

    def _fit_resample(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        class_label = np.unique(y)
        self.mask_idx = np.zeros(len(X))

        # Convert to torch tensor because it is possible to use GPU
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()

        if self.cuda:
            X = X.cuda()
            y = y.cuda()

        for label in class_label:
            _mask_idx = np.zeros(len(X), dtype=bool)
            class_samples_idx = torch.where(y == label)[0].cpu()
            idx = rng.choice(len(class_samples_idx), size=100, replace=False)

            _mask_idx[class_samples_idx[idx]] = True
            class_samples_idx = np.delete(class_samples_idx, idx)
            
            for it in range((self.n_samples // 100)-1):
                x_f = X[_mask_idx] # Selected samples
                x_r = X[class_samples_idx] # Remaining samples
                
                max_idx = self._get_most_relevant_samples(x_f, x_r, 100)

                _mask_idx[class_samples_idx[max_idx]] = True
                class_samples_idx = np.delete(class_samples_idx, max_idx) # remove the selected samples from the remaining samples
            
            if self.n_samples % 100 != 0:
                x_f = X[_mask_idx] # Selected samples
                x_r = X[class_samples_idx] # Remaining samples
                max_idx = self._get_most_relevant_samples(x_f, x_r, self.n_samples % 100)
                _mask_idx[class_samples_idx[max_idx]] = True
                
            self.mask_idx = np.logical_or(self.mask_idx, _mask_idx)

        return _safe_indexing(X.cpu().numpy(), self.mask_idx), _safe_indexing(y.cpu().numpy(), self.mask_idx)

    def _get_most_relevant_samples(self, x_f: np.ndarray, x_r: np.ndarray, n_samples: int):
        '''
            Get the most relevant samples from the remaining samples

            Parameters
            ----------
                x_f: np.ndarray
                    Selected samples

                x_r: np.ndarray
                    Remaining samples

                n_samples: int
                    Number of samples to select from the remaining samples

            Returns
            -------
                np.ndarray
                    The indices of the most relevant samples from x_r
        '''
        P = self._OCSP(x_f.T)
        projected_x_r = (P@x_r.T).T
        norms = torch.linalg.norm(projected_x_r, dim=1)
        return torch.argsort(norms)[-n_samples:].cpu() # Get the largest projection

    def _OCSP(self, X: torch.Tensor):
        # Pseudo inverse estimated by solve() because it is
        # faster and more stable than computing the inverse explicitly
        pseudo_inverse = torch.linalg.solve(X.T@X, X.T) # a.k.a, torch.inverse(X.T@X) @ X.T 
        return torch.eye(X.shape[0], device=X.device) - X@pseudo_inverse