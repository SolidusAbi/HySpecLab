import torch
from torch import Tensor
from torch import nn
from collections import OrderedDict

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Encoder, self).__init__()

        self.encode = nn.Sequential(OrderedDict([
                ('pad_0', nn.ReflectionPad2d(1)),
                ('conv_0', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)), 
                ('act_0', nn.ReLU()),
                ('pooling', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)), 
                ('conv_1', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)), 
                ('act_1', nn.ReLU()),
                ('bn', nn.BatchNorm2d(out_channels)),
            ])
        )

    def forward(self, x: Tensor):
        return self.encode(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Decoder, self).__init__()

        self.decode = nn.Sequential(OrderedDict([
                ('upsampling', nn.Upsample(scale_factor=2, mode='nearest')),
                ('pad_0', nn.ReflectionPad2d(1)),
                ('conv_0', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)),
                # ('act_0', nn.ReLU()),
                # ('conv_1', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
                ('act_1', nn.ReLU()),
                ('bn', nn.BatchNorm2d(out_channels)),
            ])
        )

    def forward(self, x: Tensor):
        return self.decode(x)

class Level(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels = 0, deeper = None):
        super(Level, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels if deeper else in_channels

        if skip_channels:
            self.skip_ = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, skip_channels, kernel_size=1, stride=1)),
                    ('act', nn.ReLU()),
                    ('bn', nn.BatchNorm2d(skip_channels))
                ])
            )

        if not deeper:
            self.f = Encoder(in_channels, out_channels)
            self.decoder = Decoder(out_channels + skip_channels, in_channels)
        else:
            if not isinstance(deeper, Level):
                raise ValueError('Meh!')
            
            self.f = nn.Sequential(
                Encoder(in_channels, deeper.in_channels),
                deeper
            )

            self.decoder = Decoder(deeper.out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor):
        x = torch.cat([self.skip_(x), self.decoder.decode[0](self.f(x))], dim=1) if hasattr(self, 'skip_') else self.f(x)
        return self.decoder.decode[1:](x) if hasattr(self, 'skip_') else self.decoder(x)

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
    def __init__(self, in_channels, out_channels, skip_channels, n_endmembers=4) -> None:
        ''' 
        
        '''
        super(UnDIP, self).__init__()
        if not(isinstance(in_channels, list)) or not(isinstance(out_channels, list)) or not(isinstance(skip_channels, list)):
            raise ValueError('Parameters must be list')

        if len(in_channels) != len(out_channels) != len(skip_channels):
            raise ValueError('The parameters must contain the samme number of elements')
        
        out_channels_inv = out_channels[::-1]
        skip_channels_inv = skip_channels[::-1]
        for idx, in_channel in enumerate(in_channels[::-1]):
            self.prior = Level(in_channel, out_channels_inv[idx], skip_channels=skip_channels_inv[idx], deeper=(self.prior if hasattr(self, 'prior') else None) )

        self.unmix = nn.Sequential(OrderedDict([
            ('conv_0', nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1)),
            ('act_0', nn.LeakyReLU(negative_slope=.1)),
            ('bn_0', nn.BatchNorm2d(out_channels[0])),
            ('ee_conv', nn.Conv2d(out_channels[0], n_endmembers, kernel_size=3, stride=1, padding=1)),
            ('ee_act', nn.Softmax(dim=1)),
        ]))

    def forward(self, x : Tensor) -> Tensor:
        x = self.prior(x)
        return self.unmix(x)