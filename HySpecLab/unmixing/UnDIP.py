from collections import deque
from itertools import islice

from torch.nn.modules import activation, linear
from torch import nn, Tensor
import torch

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


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encode = nn.Sequential()

        self.decode = nn.Sequential()

    def get_encode(self, dropout=False):
        encode = []
        for module in self.encode:
            if dropout or isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Sigmoid, nn.ReflectionPad2d,
                                nn.Tanh, nn.Identity, nn.MaxPool2d, nn.BatchNorm1d, nn.BatchNorm2d)):
                encode.append(module)

        return nn.Sequential(*encode)

    def get_decode(self, dropout=False):
        decode = []
        for module in self.decode:
            if dropout or isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Sigmoid, 
                                nn.Tanh, nn.Identity, nn.MaxPool2d, nn.BatchNorm1d, 
                                nn.BatchNorm2d, nn.Upsample, nn.ConvTranspose2d)):
                decode.append(module)

        return nn.Sequential(*decode)

    def is_valid_activation_fuction(self, activation_funct: nn.Module) -> bool:
        return isinstance(activation_funct, (activation.LeakyReLU, activation.ReLU, 
                                            activation.Sigmoid, activation.Tanh, 
                                            linear.Identity))

    def encode_decode_activation(self, activation_func) -> list:
        r'''
            This function obtain the encoder and decoder activation function.
        '''
        if isinstance(activation_func, (list, tuple)):
            if len(activation_func) != 2:
                raise ValueError("activation_func as a list has to contain 2 activation function, the encoder and decoder activation function")
            
            return activation_func
        else:
            return activation_func, activation_func

    def weight_init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)

class AutoEncoderConv(AutoEncoder):
    def __init__(self, in_ch, btl_ch, out_ch, activation_func = nn.ReLU(inplace=True), skip_connection=0, reflection=False, batch_norm = True, **kwargs):
        super(AutoEncoderConv, self).__init__()

        encode_act_func, decode_act_func = self.encode_decode_activation(activation_func)
        encode_act_func = encode_act_func if self.is_valid_activation_fuction(encode_act_func) else nn.ReLU(inplace=True)
        decode_act_func = decode_act_func if self.is_valid_activation_fuction(decode_act_func) else nn.ReLU(inplace=True)


        self.encode = nn.Sequential(
            nn.Dropout2d(0.2),
            *(  nn.ReflectionPad2d(1), nn.Conv2d(in_ch, btl_ch, 3, stride=1, padding=0) ) if (reflection) 
                else ( nn.Conv2d(in_ch, btl_ch, 3, stride=1, padding=1), ),
            encode_act_func,
            *(  nn.MaxPool2d(3, stride=2, padding=1), nn.BatchNorm2d(btl_ch) ) if (batch_norm) 
                else ( nn.MaxPool2d(3, stride=2, padding=1), )
        )

        n_in_decode = (btl_ch + skip_connection) if skip_connection else btl_ch
        decode_act_func if self.is_valid_activation_fuction(decode_act_func) else nn.ReLU(inplace=True),
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Dropout2d(0.2),
            nn.Conv2d(n_in_decode, out_ch, 3, padding=1),
            *(  decode_act_func, nn.BatchNorm2d(out_ch) ) if (batch_norm) 
                else ( decode_act_func, )
        )

        for m in self.modules():
            self.weight_init(m)


from torch import Tensor
from collections import OrderedDict

class Level(nn.Module):
    def __init__(self, in_ch, btl_ch, out_ch, skip_channels = 0, dropout = True, 
                batch_norm = True, deeper = None, **kwargs):
        super(Level, self).__init__()

        self.in_channels = in_ch
        self.out_channels = out_ch if deeper else in_ch

        if skip_channels:
            self.skip_ = nn.Sequential(
                nn.Conv2d(in_ch, skip_channels, kernel_size=1, stride=1),
                *( nn.ReLU(inplace=True), nn.BatchNorm2d(skip_channels) ) if batch_norm
                    else ( nn.ReLU(inplace=True), )
            )

        ae = AutoEncoderConv(in_ch, btl_ch, out_ch, skip_connection=skip_channels,
                             batch_norm=batch_norm, **kwargs)
        if not deeper:
            self.f = ae.get_encode(dropout=dropout)
        else:
            if not isinstance(deeper, Level):
                raise ValueError('deeper parameters must be a Level class!')
            
            self.f = nn.Sequential(
                ae.get_encode(dropout=dropout),
                deeper
            )

        self.decoder = ae.get_decode(dropout=dropout)

    def forward(self, x: Tensor):
        x = torch.cat([self.skip_(x), self.decoder[0](self.f(x))], dim=1) if hasattr(self, 'skip_') else self.f(x)
        return self.decoder[1:](x) if hasattr(self, 'skip_') else self.decoder(x)


class UnDIP(nn.Module):
    ''' 
        HyperSpectral Unmixing using Deep Image Prior (UnDIP)

        Parameters
        ----------
            in_channels
            out_channels
            skip_channels

        Reference
        ---------
            [1] UnDIP: Hyperspectral Unmixing Using Deep Image Prior (10.1109/TGRS.2021.3067802)
    '''
    def __init__(self, n_endmembers:int, out_channels:int, sdae_dims:list, skip_channels:list, **kwargs) -> None:
        ''' 
        
        '''
        if not(isinstance(sdae_dims, list)) or not(isinstance(skip_channels, list)):
            raise ValueError('Parameters must be list')

        if len(skip_channels) != len(sdae_dims) -1:
            raise ValueError('The "skip_channels" must contains the numbers of elements in "dims" grouped in pairs')

        super(UnDIP, self).__init__()
        
        self.prior = None
        for idx, (btl_channel, in_channel) in enumerate(slide(sdae_dims[::-1], 2)):
            out_channel = out_channels if idx == len(sdae_dims)-2 else in_channel
            self.prior = Level(in_channel, btl_channel, out_channel, skip_channels=skip_channels[::-1][idx],
                            deeper=(self.prior if self.prior else None), **kwargs)


        self.unmix = nn.Sequential(*[
            nn.Conv2d(out_channels, n_endmembers, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=1),
        ])

    def forward(self, x : Tensor) -> Tensor:
        x = self.prior(x)
        return self.unmix(x)