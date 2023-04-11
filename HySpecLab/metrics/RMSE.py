import torch
from torch.nn.functional import mse_loss

def rmse(x: torch.Tensor, y: torch.Tensor, dim=1):
    '''
        Root Mean Squared Error (RMSE)

        Parameters
        ----------
            x: torch.Tensor, shape (batch_size, n_bands)
                input tensor.

            y: torch.Tensor, shape (batch_size, n_bands)
                target tensor.
            
            dim: int
                dimension to compute the mean. if 'None' the mean 
                is computed over all the elements.
    '''
    mse = mse_loss(x, y, reduction='none')
    if dim is None:
        return torch.sqrt(mse.mean())
    else:
        return torch.sqrt(mse.mean(dim))