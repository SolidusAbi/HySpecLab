import torch
from torch import nn
from Sparse import KWinners

class KFeatureSelector(nn.Module):
    '''
        Feature selector by a dynamic threshold, 'KWinners' approach.
    '''
    def __init__(self, in_features: int, k: int) -> None:
        super(KFeatureSelector, self).__init__()
        self.model = nn.Sequential(*[
            nn.Conv1d(in_features,in_features,1, groups=in_features, bias=False),
            nn.Flatten(1),
            KWinners(in_features, 25)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x[:,:,None]

        return self.model(x)

    