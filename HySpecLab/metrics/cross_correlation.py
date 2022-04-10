# Based on TorchIR implementation

import torch 
from torch import Tensor
from torch.nn.modules.loss import _Loss


class StableStd(torch.autograd.Function):    
    r'''
        Stable standard deviation for Normalized Cross Correlation.
        This is a custom autograd function for computing the standard deviation
        of a tensor.

        The input tensor is assumed to be a 1D tensor.

        The output tensor is a scalar.
    '''
    @staticmethod
    def forward(ctx, tensor):
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        res = torch.std(tensor).detach()
        ctx.result = res.detach()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.tensor.detach()
        result = ctx.result.detach()
        e = 1e-6
        assert tensor.numel() > 1
        return (
            (2.0 / (tensor.numel() - 1.0))
            * (grad_output.detach() / (result.detach() * 2 + e))
            * (tensor.detach() - tensor.mean().detach())
        )

def ncc(x1, x2, e=1e-10):
    r'''
        Normalized cross correlation.

        Args:
        -----
            x1 (Tensor): 1D tensor.
            x2 (Tensor): 1D tensor.
            e (float): epsilon for numerical stability.
    '''
    assert x1.shape == x2.shape, "Inputs are not of equal shape"
    cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
    std = StableStd.apply(x1) * StableStd.apply(x2)
    ncc = cc / (std + e)
    return ncc

class NCC(_Loss):
    r'''
        Normalized cross correlation as Pytorch loss function.

        Args:
        -----
            target: Tensor, shape(N, D)
            pred: Tensor, shape(N, D)   
    '''
    def __init__(self):
        super(NCC, self).__init__()

    def forward(self, target: Tensor, pred: Tensor)-> Tensor:
        # Negative for minimization
        return -ncc(target, pred)