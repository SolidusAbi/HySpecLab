import torch 
from torch import Tensor
from torch.nn.modules.loss import _Loss

def total_variational(x: Tensor) -> Tensor:
    return torch.diff(x, 1).abs().sum(axis=1)

class TotalVariationalReg(_Loss):
    r'''
        Total Variational Regularization factor.
        This is a custom autograd function for computing the total variational
        regularization factor.
        
        The input tensor is assumed to be a 1D tensor. The output tensor is a scalar.
        
        Parameters
        ----------
            reduction: str
                Specifies the reduction to apply to the output: 'mean': the weighted mean of
                 the output is taken, 'sum': the output will be summed.
    '''
    def __init__(self, reduction: str = 'mean') -> None:
        super(TotalVariationalReg, self).__init__()
        self.reduction = torch.sum if reduction == 'sum' else torch.mean
    
    def forward(self, x: Tensor) -> Tensor:
        return self.reduction(total_variational(x))


        