import torch
from torch.nn.modules.loss import _Loss

class UnmixingLoss(_Loss):
    '''
        Loss function for reconstruction based on Frobenius norm.
    '''
    def __init__(self, reduction='mean'):
        super(UnmixingLoss, self).__init__(reduction=reduction)
        self.reduction = torch.mean if reduction == 'mean' else torch.sum

    def forward(self, input, target):
        return self.reduction(torch.norm(target-input, dim=1))