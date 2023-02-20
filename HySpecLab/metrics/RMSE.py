from torch.nn.functional import mse_loss

def rmse(x: torch.Tensor, y: torch.Tensor):
    '''
        Root Mean Squared Error (RMSE)

        Parameters
        ----------
            x: torch.Tensor, shape (batch_size, n_bands)
                input tensor.

            y: torch.Tensor, shape (batch_size, n_bands)
                target tensor.
    '''
    return torch.sqrt(mse_loss(x, y, reduction='none').mean(dim=1))