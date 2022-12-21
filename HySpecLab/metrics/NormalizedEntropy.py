import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.nn.functional import softmax, log_softmax

class NormalizedEntropy(_Loss):
    '''
        Normalized Shannon's Entropy

        Parameters
        ----------
            S : int
                Number of classes or endmembers, used for the normalization. 
    '''
    def __init__(self, S: int, reduction:str='mean') -> None:
        super(NormalizedEntropy, self).__init__(reduction=reduction)
        self.max_entropy = np.log(S)
        self.reduction = torch.mean if reduction == 'mean' else torch.sum

    def forward(self, x: torch.Tensor):
        '''
            Args:
                x: torch.Tensor, shape (N, S): 
                    This matrix contains the logit values of the classes or endmembers.
        '''
        return self.reduction(-torch.sum(softmax(x, dim=1) * log_softmax(x, dim=1), dim=1)/self.max_entropy)