import torch
import torch.nn as nn

from .utils import get_encoder, get_decoder


class UNet(nn.Module):
    """ UNet

                        x --> D1 --> D2 --> D3 --> D4 --> D5 ---.
                              |      |      |      |            |
                              v      v      v      v            |
      <-- HEAD <-- ( U5 ) <-- U4 <-- U3 <-- U2 <-- U1 <---------·  

    """

    def __init__(self, encoder='vanilla', input_channels=3, num_classes=1, **kwargs):
        """
        Parameters
        ----------
        encoder : str
            encoder used; valido options are:
                vanilla, resnet18, resnet34, resnet50, resnet101
        input_channels : int
            number of input channels; e.g. 3 for H x W x 3 images
        num_classes : int
            number of output classes
        depth : int
            encoder/decoder depth (not including center block)
        base_features : int
            number of output channels for the first encoder block; this number
            is then doubled after each encoder block (e.g. 16 -> 32 -> 64 ...)
        """
        super().__init__()

        self.encoder = get_encoder(encoder, input_channels, **kwargs)
        self.decoder = get_decoder(encoder, **kwargs)

        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder(features)
        x = self.head(x)

        return x
