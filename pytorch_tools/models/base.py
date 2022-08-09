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
