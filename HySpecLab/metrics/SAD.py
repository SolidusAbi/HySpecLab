import torch
from torch import Tensor
from torch.nn.functional import hardtanh
from torch.nn.modules.loss import _Loss

def sad(x: Tensor, y: Tensor, n_bands: int) -> Tensor:
    r'''
      Parameters
      ----------
        x: Tensor, shape (batch_size, n_bands)          
        y: Tensor, shape (batch_size, n_bands)

      Returns
      -------
        Tensor, shape (batch_size,)
          Angle between the input and the target
    ''' 
    inputs_norm = torch.norm(x, dim=1)
    targets_norm = torch.norm(y, dim=1)

    summation = torch.bmm(x.view(-1, 1, n_bands), y.view(-1, n_bands, 1)).squeeze()

    # Using Hard Tanh to force values between [-1, 1] because $cos^{-1}(x)$ where $x \in [-1,1]$
    return torch.acos(hardtanh(summation / (inputs_norm * targets_norm)))


class SAD(_Loss):
  r'''
    Spectral Angle Distance (SAD) Objective
    Code inspired by: https://github.com/dv-fenix/HyperspecAE/blob/main/src/train_objectives.py

    Parameters
    ----------
      n_bands: int
        Number of bands in the hyperspectral image

      reduction: str
        Specifies the reduction to apply to the output: 'mean': the weighted mean of
        the output is taken, 'sum': the output will be summed.
  '''
  def __init__(self, n_bands: int=156, reduction: str='mean') -> None:
    super(SAD, self).__init__()
    self.n_bands = n_bands
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
          Angle between the input and the target
    ''' 
    
    return self.reduction(sad(inputs, targets, self.n_bands))