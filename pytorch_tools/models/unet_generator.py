import torch
import torch.nn as nn

from .base import ConvBNReLU, NNCatConv

class encoder(nn.Module):
    def __init__(self, input_channels,
                 dims=[16,32,64,128,256], rep=None,
                 block=ConvBNReLU,
                 down_param={'keep': {'down': False}, 'down': {'down': True}},
                 params=None):
        super().__init__()
        if rep is None: rep = [1 for i in range(len(dims))]
        else: assert len(dims) == len(rep)

        if params is None: params = [{} for i in range(len(dims))]
        else: assert len(dims) == len(params)

        self.backbone = nn.ModuleDict()
        self.features_dims = dims

        for k in range(len(dims)):
            in_dim = input_channels if k == 0 else ou_dim
            ou_dim = dims[k]      # Output dimension for the layer

            block_params = params[k]
            # First layer never gets downsampled
            down = down_param['keep'] if k == 0 else down_param['down']
            if rep[k] == 1:
                seq = block(in_dim, ou_dim, **down, **block_params)
            else:
                seq = nn.Sequential()
                seq.append(block(in_dim, ou_dim, **down, **block_params))

                in_dim = ou_dim
                for i in range(rep[k]-1):
                    seq.append(block(in_dim, ou_dim, **down_param['keep'], **block_params))

            name = f'layer_{k}' if k > 0 else 'stem'
            self.backbone.update({name: seq})

    def forward(self, x):
        features = []
        for key in self.backbone.keys():
            x = self.backbone[key](x)
            features.append(x)

        return features


class decoder(nn.Module):
    def __init__(self, features_dims=[16,32,64,128,256], decoder_dims=None,
                 block = NNCatConv,
                 params=None):
        super().__init__()
        
        self.upstream = nn.ModuleList()

        assert len(features_dims) > 1
        if decoder_dims is None: decoder_dims =  [16*2**k for k in reversed(range(len(features_dims)-1))]
        else: assert len(features_dims)-1 == len(decoder_dims)

        if params is None: params = [{} for i in range(len(decoder_dims))]
        else: assert len(decoder_dims) == len(params)

        in_dim0 = features_dims[-1]    # tensor to be upsampled
        for k, ou_dim in enumerate(decoder_dims):
            in_dim1 = features_dims[-(2+k)]    # tensor coming from skip connection
            block_params = params[k]
            self.upstream.append(block(in_dim0, in_dim1, ou_dim, **block_params))

            in_dim0 = ou_dim    # input to next layer is output of previous layer

    def forward(self, features):
        x = features[-1]
        for k in range(len(self.upstream)):
            x = self.upstream[k](x, features[-(2+k)])

        return x
