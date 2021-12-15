import torch
from torch import Tensor
from enum import Enum

class NOISE_TYPE(Enum):
    uniform=0,
    normal=1

def fill_noise(x:Tensor, noise_type:NOISE_TYPE):
    '''
        Fills tensor `x` with noise of type `noise_type`.
    '''
    if noise_type == NOISE_TYPE.uniform:
        x.uniform_()
    elif noise_type == NOISE_TYPE.normal:
        x.normal_() 
    else:
        assert False

def get_noise(shape: tuple, batch_size: int, noise_type=NOISE_TYPE.uniform, var=1./10):
    '''
        Returns a pytorch.Tensor of size (B, C, H, W) initialized in a specific way.
        
        Arguments
        ---------
            shape: Tuple, shape (C, H, W)
                Contains the dimension of the noise generated

            batch_size: int
                Number of elementes which define the Tensor shape (batch_size, C, H, W)

            method: NOISE_METHOD
                Method to generate the random Tensor

            noise_type: NOISE_TYPE
                Define the noise used for initialize the tensor. 
            
            std: float
                A factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    '''
    spatial_size = shape[1:]
    channel_size = shape[0]
    out_shape = [batch_size, channel_size, spatial_size[0], spatial_size[1]]
    output = torch.zeros(out_shape)
    
    fill_noise(output, noise_type)
      
    return output * var