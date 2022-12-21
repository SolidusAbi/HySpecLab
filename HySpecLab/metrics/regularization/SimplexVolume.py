import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from sklearn.decomposition import PCA

class SimplexVolumeLoss(_Loss):
    '''
        Simplex Volume Regularization defined by the endmembers E. It is the implementation
        proposed in [1].

        Parameters
        ----------
            X: Array-like, shape (n_samples, n_bands)
                The input samples.
            
            n_endmembers: int
                Number of endmembers.

        References
        ----------
            [1] Miao, L., & Qi, H. (2007). Endmember extraction from highly mixed data using
            minimum volume constrained nonnegative matrix factorization. IEEE Transactions on
            Geoscience and Remote Sensing, 45(3), 765-777.

    '''

    def __init__(self, X, n_endmembers, random_state=42) -> None:
        super(SimplexVolumeLoss, self).__init__()
        self.c = n_endmembers
        self.mu = X.reshape(-1, X.shape[-1]).mean(axis=0, keepdims=True).T # (bands, 1)
        self.mu = torch.tensor(self.mu).type(torch.float32) if not isinstance(self.mu, torch.Tensor) else self.mu

        self.U = PCA(n_components=n_endmembers-1, random_state=random_state).fit(X.reshape(-1, X.shape[-1])).components_.T # (bands, endmembers-1)
        self.U = torch.tensor(self.U).type(torch.float32)

        self.B = torch.vstack((torch.zeros((n_endmembers-1,)), torch.eye(n_endmembers-1)))
        self.C = torch.zeros((n_endmembers, n_endmembers))
        self.C[0, :] = 1

        self.tau = 1 / np.math.factorial(self.c - 1) # tau = 1 / (c-1)!

    def forward(self, endmembers: torch.Tensor) -> torch.Tensor:
        '''
            Parameters
            ----------
                endmembers: torch.Tensor, shape (n_endmembers, n_bands)
                    The endmembers to be regularized. 
        '''
        Z = self.C + self.B@self.U.T@(endmembers.T-self.mu)
        return (0.5*self.tau)*(torch.linalg.det(Z).pow(2))

    def cuda(self):
        self.mu = self.mu.cuda()
        self.U = self.U.cuda()
        self.B = self.B.cuda()
        self.C = self.C.cuda()
        return self

    def to(self, device:str):
        self.mu = self.mu.to(device)
        self.U = self.U.to(device)
        self.B = self.B.to(device)
        self.C = self.C.to(device)
        return self