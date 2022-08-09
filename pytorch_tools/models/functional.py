import torch.nn as nn

def build_encoder(encoder_block, conv_block, input_channels, depth, base_features, **kwargs):
    """ Build an encoder from the base block 'encoder_block'

    Parameters
    ----------
    encoder_block : nn.Module
        base block of the encoder; 1 input, 2 outputs
    input_channels : int
        number of input channels
    depth : int
        depth of the encoder; number of sucessive 'encoder_block's
    base_features : int
        number of output channels for the first block (doubled after each
        block)

    Returns
    -------
    nn.Sequential
        encoder
    """

    encoder = nn.Sequential()
    in_dim = input_channels
    for k in range(depth):
        out_dim = base_features * 2**k    # Double out_dim at each step
        encoder.append(encoder_block(conv_block, in_dim, out_dim, **kwargs))
        in_dim = out_dim
        
    return encoder

def build_decoder(decoder_block, conv_block, depth, base_features, **kwargs):
    """ Build a decoder from the base block 'decoder_block'

    Parameters
    ----------
    decoder_block : nn.Module
        base block of the decoder; 2 inputs, 1 output
    depth : int
        depth of the encoder; number of sucessive 'encoder_block's
    base_features : int
        number of output channels for the first block (doubled after each
        block)

    Returns
    -------
    nn.Sequential
        decoder
    """
    decoder = nn.Sequential()
    for k in range(depth, 0, -1):
        in_dim, out_dim = base_features * 2**k, base_features * 2**(k-1)
        decoder.append(decoder_block(conv_block, in_dim, out_dim, **kwargs))
        
    return decoder
