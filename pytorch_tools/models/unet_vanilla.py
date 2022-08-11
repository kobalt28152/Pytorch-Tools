import torch.nn as nn

from .base import ConvBNReLU_2, encoder_block, decoder_block,


def build_encoder(input_channels, depth, base_features):
    """ Utility function to create the encoder """

    encoder = []
    in_dim = input_channels
    for k in range(depth):
        out_dim = base_features * 2**k    # Double out_dim at each step
        if k == 0:
            encoder.append(encoder_block(in_dim, out_dim, reduce=False))
        else:
            encoder.append(encoder_block(in_dim, out_dim, reduce=True))
        in_dim = out_dim

    return nn.ModuleList(encoder)


def build_decoder(depth, base_features, **kwargs):
    """ Utility function to create the decoder """

    decoder = []
    for k in reversed(range(1, depth)):
        in_dim, out_dim = base_features * 2**(k), base_features * 2**(k-1)
        decoder.append(decoder_block(in_dim, out_dim))

    return nn.ModuleList(decoder)



class vanilla_encoder(nn.Module):
    """ Encoder used in vanilla UNet """

    def __init__(self, input_channels, depth=5, base_features=16):
        """
        Parameters
        ----------
        input_channels : int
            number of input channels
        depth : int
            depth of the encoder; number of sucessive 'encoder_block's
        base_features : int
            number of output channels for the first block (doubled after each
            block)
        """
        super().__init__()
        self.depth = depth
        self.backbone = build_encoder(input_channels, depth, base_features)

    def forward(self, x):
        features = []
        for i in range(self.depth):
            x = self.backbone[i](x)
            features.append(x)

        return features


class vanilla_decoder(nn.Module):
    """ Decoder used in vanilla UNet """

    def __init__(self, depth=5, base_features=16):
        """
        Parameters
        ----------
        depth : int
            depth of the encoder; number of features given by the encoder.
        base_features : int
            number of output channels for the first block (doubled after each
            block)
        """
        super().__init__()
        self.up_stream = build_decoder(depth, base_features)
        self.depth = depth - 1    # decoder depth always -1

    def forward(self, features):
        x = features[-1]
        for i in range(self.depth):
            x = self.up_stream[i](x, features[self.depth-1-i])

        return x
