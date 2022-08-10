from .unet_vanilla import vanilla_encoder, vanilla_decoder
from .resnet import resnet18_encoder, resnet34_encoder, resnet50_encoder, resnet101_encoder
from .resnet import resnet_decoder


def get_encoder(name, input_channels=3, **kwargs):
    if name == 'vanilla':
        return vanilla_encoder(input_channels, **kwargs)
    elif name == 'resnet18':
        return resnet18_encoder(input_channels, **kwargs)
    elif name == 'resnet34':
        return resnet34_encoder(input_channels, **kwargs)
    elif name == 'resnet50':
        return resnet50_encoder(input_channels, **kwargs)
    elif name == 'resnet101':
        return resnet101_encoder(input_channels, **kwargs)

    raise ValueError(f'Invalid name: {name}')

def get_decoder(name, **kwargs):
    if name == 'vanilla':
        return vanilla_decoder(**kwargs)
    else:
        return resnet_decoder(name, **kwargs)

    # raise ValueError(f'Invalid name: {name}')
