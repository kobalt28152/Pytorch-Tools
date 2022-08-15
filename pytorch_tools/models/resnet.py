import torch
import torch.nn as nn

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from torchvision.models.feature_extraction import create_feature_extractor

from .base import NNCatConv

def resnet18_encoder(input_channels, pretrained=True, **kwargs):
    return_nodes = {
            'relu': 0,
            'layer1.1.relu_1': 1,
            'layer2.1.relu_1': 2,
            'layer3.1.relu_1': 3,
            'layer4.1.relu_1': 4
            }
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = resnet18(weights=weights)
    if input_channels != 3:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return create_feature_extractor(model, return_nodes)

def resnet34_encoder(input_channels, pretrained=True,**kwargs):
    return_nodes = {
            'relu': 0,
            'layer1.2.relu_1': 1,
            'layer2.3.relu_1': 2,
            'layer3.5.relu_1': 3,
            'layer4.2.relu_1': 4
            }
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = resnet34(weights=weights)
    if input_channels != 3:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return create_feature_extractor(model, return_nodes)

def resnet50_encoder(input_channels, pretrained=True,**kwargs):
    return_nodes = {
            'relu': 0,
            'layer1.2.relu_2': 1,
            'layer2.3.relu_2': 2,
            'layer3.5.relu_2': 3,
            'layer4.2.relu_2': 4
            }
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = resnet50(weights=weights)
    if input_channels != 3:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return create_feature_extractor(model, return_nodes)

def resnet101_encoder(input_channels, pretrained=True,**kwargs):
    return_nodes = {
            'relu': 0,
            'layer1.2.relu_2': 1,
            'layer2.3.relu_2': 2,
            'layer3.22.relu_2': 3,
            'layer4.2.relu_2': 4
            }
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = resnet101(weights=weights)
    if input_channels != 3:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return create_feature_extractor(model, return_nodes)


def get_dims(name):
    if name == 'resnet18' or name == 'resnet34':
        return [64, 64, 128, 256, 512]
    else:
    # elif name == 'resnet50' or name == 'resnet101':
        return [64, 256, 512, 1024, 2048]



# Encoder output always:
# dims = [16, 32, 64, 128, 256, 512]
# resnet18:  [64, 64, 128, 256, 512]
# resnet34:  [64, 64, 128, 256, 512]
# resnet50:  [64, 256, 512, 1024, 2048]
# resnet101: [64, 256, 512, 1024, 2048]
def build_decoder(name, **kwargs):
    """ Utility function to create the decoder """
    decoder = []
    resnet_dims = get_dims(name)                 # [64,64,128,256,512] or [64,256,512,1024,2048]
    output_dims =  [256, 128, 64, 32, 16]        # decoder out dims 

    # in_dim = resnet_dims[-1] + resnet_dims[-2]
    in_dim0 = resnet_dims[-1]
    for k, out_dim in enumerate(output_dims):
        in_dim1 = resnet_dims[-(2+k)] if 2+k <= len(resnet_dims) else 0
        decoder.append(NNCatConv(in_dim0, in_dim1, out_dim))

        in_dim0 = out_dim    # input to next layer is output of current layer

    return nn.ModuleList(decoder)


class resnet_decoder(nn.Module):
    """ Decoder used for pretrained backbones """

    def __init__(self, name, block=NNCatConv, params=None):
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

        self.upstream = nn.ModuleList()

        resnet_dims = get_dims(name)                 # [64,64,128,256,512] or [64,256,512,1024,2048]
        output_dims =  [256, 128, 64, 32, 16]        # decoder out dims 

        if params is None: params = [{} for i in range(len(output_dims))]
        else: assert len(output_dims) == len(params)

        in_dim0 = resnet_dims[-1]
        for k, out_dim in enumerate(output_dims):
            in_dim1 = resnet_dims[-(2+k)] if 2+k <= len(resnet_dims) else 0
            block_params = params[k]
            self.upstream.append(block(in_dim0, in_dim1, out_dim, **block_params))
            in_dim0 = out_dim    # input to next layer is output of current layer

    def forward(self, features):
        x = features[f'4']
        for i in range(len(self.upstream)):
            f = features[f'{3-i}'] if 3-i >= 0 else None
            x = self.upstream[i](x, f)

        return x
