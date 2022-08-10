import torch
import torch.nn as nn

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
from torchvision.models.feature_extraction import create_feature_extractor

from .base import decoder_block_up

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
    output_dims =  [16, 32, 64, 128, 256]        # decoder out dims 

    in_dim = resnet_dims[-1] + resnet_dims[-2]
    for k, out_dim in enumerate(reversed(output_dims)):
        decoder.append(decoder_block_up(in_dim, out_dim))
        tmp = resnet_dims[-(3+k)] if 3+k < len(resnet_dims)+1 else 0
        in_dim = out_dim + tmp

    return nn.ModuleList(decoder)


class resnet_decoder(nn.Module):
    """ Decoder used for pretrained backbones """

    def __init__(self, name, **kwargs):
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
        self.up_stream = build_decoder(name)

    def forward(self, features):
        x = features[f'4']
        for i in range(len(self.up_stream)):
            f = features[f'{3-i}'] if 3-i >= 0 else None
            x = self.up_stream[i](x, f)

        return x
