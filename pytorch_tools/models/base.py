import torch
import torch.nn as nn

class ConvBNReLU_2(nn.Module):
    """ Base convolution block (used throught out the UNet model)

    Conv -> BN -> ReLU -> Conv -> BN -> ReLU """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU()

        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        return x

class ConvReLUDrop_2(nn.Module):
    """ Base convolution block (used throught out the UNet model)

    Conv -> ReLU -> Dropout -> Conv -> ReLU """

    def __init__(self, in_dim, out_dim, p_drop=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU()
        self.drop = nn.Dropout2d(p=p_drop)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.act(x)

        return x

class encoder_block_var(nn.Module):
    """ Encoder block

    Encoder block with variable downsampler (e.g., maxpool) and variable
    convolutional block (e.g. ConvBNReLU -> ConvBNReLU)

            ---> Downs ---> Block --->    (*,out,H/2,W/2)
        in_dim        mid_dim       out_dim
    """

    def __init__(self, in_dim, out_dim, mid_dim=None, down=None, block=None, **kwargs):
        """
        Parameters
        ----------
        in_dim : int
            input dimension
        mid_dim : int
            dimension at the output of the downsample layer; input dimension
            for the convolutional block
        out_dim : int
            output dimension of the convolutional block.
        """
        super().__init__()
        if mid_dim is None: mid_dim = in_dim

        self.down = down(in_dim, mid_dim, **kwargs)
        self.conv = block(mid_dim, out_dim, **kwargs)


    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        x = self.conv(x)

        return x


class encoder_block(nn.Module):
    """ Encoder block used in vanilla UNet

    First encoder block does not use maxpool to reduce dimensionality.

        reduce=False: (*,in,H,W)   ---> ConvBNReLU --->                 (*,out,H,W)
        reduce=True:  (*,in,H,W)   ---> maxpool ---> ConvBNReLU --->    (*,out,H/2,W/2) """

    def __init__(self, in_dim, out_dim, reduce=True, **kwargs):
        super().__init__()
        self.reduce = reduce
        self.maxpool = nn.MaxPool2d((2, 2)) if reduce else None
        self.conv = ConvBNReLU_2(in_dim, out_dim, **kwargs)

    def forward(self, x):
        if self.reduce:
            x = self.maxpool(x)
        x = self.conv(x)

        return x


class decoder_block(nn.Module):
    """ Decoder block used in vanilla UNet

    Decoder block takes two inputs:
    - output from previous block (x)
    - feature from encoder block (f)

    x ---> convT ----> cat ----> ConvBNReLU --->
                        ^
                        |
    f ------------------·    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.conv = ConvBNReLU_2(in_dim, out_dim)

    def forward(self, x, f):
        x = self.convT(x)
        x = torch.cat([x, f], dim=1)    # (N x C x H x W): conctar along C
        x = self.conv(x)

        return x


class decoder_block_up(nn.Module):
    """ Decoder block with upsampling

    Decoder block takes two inputs:
    - output from previous block (x)
    - feature from encoder block (f)

    x ----> up ----> cat ----> ConvBNReLU --->
                       ^
                       |
    f -----------------·    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2.0)
        self.conv = ConvBNReLU_2(in_dim, out_dim)

    def forward(self, x, f):
        x = self.up(x)
        if f is not None:
            x = torch.cat([x, f], dim=1)    # (N x C x H x W): conctar along C
        x = self.conv(x)

        return x
