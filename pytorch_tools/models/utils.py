from .unet_vanilla import vanilla_encoder, vanilla_decoder
# from .unet_vanilla import drop_encoder
from .resnet import resnet18_encoder, resnet34_encoder, resnet50_encoder, resnet101_encoder
from .resnet import resnet_decoder
from .convnext import ConvNeXt, convnext_decoder


def get_encoder(name, input_channels=3, **kwargs):
    if name == 'vanilla':
        return vanilla_encoder(input_channels, **kwargs)
    # elif name == 'dropout':
        # return (input_channels, **kwargs)
    elif name == 'resnet18':
        return resnet18_encoder(input_channels, **kwargs)
    elif name == 'resnet34':
        return resnet34_encoder(input_channels, **kwargs)
    elif name == 'resnet50':
        return resnet50_encoder(input_channels, **kwargs)
    elif name == 'resnet101':
        return resnet101_encoder(input_channels, **kwargs)
    elif name == 'convnext-T':
        return ConvNeXt(input_channels, **kwargs)

    raise ValueError(f'Invalid name: {name}')

def get_decoder(name, **kwargs):
    if name == 'vanilla':
        return vanilla_decoder(**kwargs)
    elif name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
        return resnet_decoder(name, **kwargs)
    elif name in ['convnext-T', 'convnext-S', 'convnext-B', 'convnext-L', 'convnext-X']:
        return convnext_decoder(name, **kwargs)
    else:
        raise ValueError(f'Invalid name: {name}')
