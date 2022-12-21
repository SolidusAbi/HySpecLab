import torch
from torch.nn.modules.loss import _Loss
from torch.nn.functional import cross_entropy, normalize

class SimilarityLoss(_Loss):
    def __init__(self, n_endmembers, temperature=1e-1, reduction='mean'):
        super(SimilarityLoss, self).__init__(reduction=reduction)
        self.temperature = temperature
        self.n_endmembers = n_endmembers

    def forward(self, X):
        '''
            Parameters
            ----------
                X: torch.Tensor, shape=(n_endmembers, n_features)
        '''
        labels = torch.arange(0, self.n_endmembers).to(X.device)
        
        # Cosine Similarity
        X = normalize(X, dim=1)
        similarity_matrix = torch.matmul(X, X.T) 

        logit = similarity_matrix / self.temperature
        return cross_entropy(logit, labels, reduction=self.reduction)