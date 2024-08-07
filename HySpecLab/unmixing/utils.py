import torch
from torch import Tensor
from enum import Enum

from collections import deque
from itertools import islice

## General
def slide(iterable, size):
    '''
        Iterate through iterable using a sliding window of several elements.
        Important: It is a generator!.
        
        Creates an iterable where each element is a tuple of `size`
        consecutive elements from `iterable`, advancing by 1 element each
        time. For example:
        >>> list(sliding_window_iter([1, 2, 3, 4], 2))
        [(1, 2), (2, 3), (3, 4)]
        
        source: https://codereview.stackexchange.com/questions/239352/sliding-window-iteration-in-python
    '''
    iterable = iter(iterable)
    window = deque(islice(iterable, size), maxlen=size)
    for item in iterable:
        yield tuple(window)
        window.append(item)
    if window:  
        # needed because if iterable was already empty before the `for`,
        # then the window would be yielded twice.
        yield tuple(window)

## UnDIP 
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


def restoration(endmembers: Tensor, abundance: Tensor) -> Tensor:
    '''
        Obtain the HySpectral Cube, Y = EA + N, where E represents the endmembers and
        A is the abudance map from this endmembers.

        Arguments
        ---------
            endmembers: torch.Tensor, shape (1, n_bands, n_endmembers)
                Contains the endmembers extracted from the HyperSpectral dataset.

            saliency: torch.Tensor, shape (B, n_endmembers, H, W)
                Output from UnDIP model.
    '''
    B, _, H, W = abundance.shape
    _, n_bands, _ = endmembers.shape
    return torch.matmul(endmembers, abundance.flatten(start_dim=2)).reshape((B, n_bands, H, W))
    