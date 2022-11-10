import torch
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _Loss

def sid(x: Tensor, y: Tensor) -> Tensor:
    return cross_entropy(x, y, reduction='none') + cross_entropy(y, x, reduction='none')

class SID(_Loss):
  r'''
    Spectral Information Divergence (SID)
    
    Code inspired by: https://github.com/dv-fenix/HyperspecAE/blob/main/src/train_objectives.py

    Parameters
    ----------
      reduction: str
        Specifies the reduction to apply to the output: 'mean': the weighted mean of
        the output is taken, 'sum': the output will be summed.
  '''
  def __init__(self, reduction: str='mean') -> None:
    super(SID, self).__init__()
    self.reduction = torch.sum if reduction == 'sum' else torch.mean

  def forward(self, inputs: Tensor, targets: Tensor):
    r'''
      Parameters
      ----------
        input: Tensor
          Input tensor of shape (batch_size, n_bands)
          
        target: Tensor
          Target tensor of shape (batch_size, n_bands)

      Returns
      -------
        Tensor
          Spectral similarity from information theory perspective
    ''' 
    
    return self.reduction(sid(inputs, targets))