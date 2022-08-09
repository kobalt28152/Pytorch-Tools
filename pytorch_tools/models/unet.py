import torch
import torch.nn as nn

from .functional import build_encoder, build_decoder
from .base import ConvBNReLU_2

class encoder_block(nn.Module):
    """ Encoder block

    Input                                  Output
             |            |---------------> a
        x -> | conv_block |  |
             |            |  .-> maxpool -> b
    """

    def __init__(self, conv_block, in_dim, out_dim):
        super().__init__()
        self.conv = conv_block(in_dim, out_dim)
        self.maxpool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.conv(x)
        y = self.maxpool(x)

        return x, y

class decoder_block(nn.Module):
    """ Decoder block

    Input                                        Output
      x -> convT ----> cat --> |            |
                       ^       | conv_block | --> a
                       |       |            |
      y ---------------·
    """

    def __init__(self, conv_block, in_dim, out_dim):
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.conv = conv_block(in_dim, out_dim)

    def forward(self, x, y):
        x = self.convT(x)
        x = torch.cat([x, y], dim=1)    # (N x C x H x W): conctar along C
        x = self.conv(x)

        return x

class UNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, depth=4, base_features=16,
            encoder_block=encoder_block, decoder_block=decoder_block, conv_block=ConvBNReLU_2,
            center_block=ConvBNReLU_2, **kwargs):
        """
        Parameters
        ----------
        input_channels : int
            number of input channels; e.g. 3 for H x W x 3 images
        num_classes : int
            number of output classes
        depth : int
            encoder/decoder depth: # of encoding/decoding blocks
        base_features : int
            number of output channels for the first encoder block; this number
            is then doubled after each encoder block (e.g. 16 -> 32 -> 64 ...)
        """
        super().__init__()
        self.encoder = build_encoder(encoder_block, conv_block, input_channels, depth, base_features, **kwargs)
        self.center = center_block(base_features * 2**(depth-1), base_features * 2**(depth), **kwargs)
        self.decoder = build_decoder(decoder_block, conv_block, depth, base_features, **kwargs)

        self.head = nn.Conv2d(base_features, num_classes, kernel_size=1)

        self.depth = depth

    def forward(self, x):
        features = []
        for i in range(self.depth):
            s, x = self.encoder[i](x)
            features.append(s)

        x = self.center(x)

        for i in range(self.depth):
            x = self.decoder[i](x, features[self.depth-1-i])

        x = self.head(x)

        return x
