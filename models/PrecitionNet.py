# Copyright 2022 CircuitNet. All rights reserved.

import torch
import torch.nn as nn

from collections import OrderedDict
from InceptionEncoder import IncepEncoder

def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                    or classname.find('Linear') != -1):
            
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    module.apply(init_func)

def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys

from UNetModule import UNet



class PredictionNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()

    def forward(self, x):
        pass
    def init_weights(self, pretrained=None, strict=True, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None. ")

class DRC_Prediction_Net(PredictionNet):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()
        self.uNet = UNet(n_channels=9, n_classes=1, bilinear=True, task="DRC")

    def forward(self, x):
        return self.uNet(x)

    def init_weights(self, pretrained=None, strict=True, **kwargs):
        super().init_weights(pretrained, strict)

class Congestion_Prediction_Net(PredictionNet):
    def __init__(self, in_channels=3, out_channels=1,**kwargs):
        super().__init__()
        self.uNet = UNet(n_channels=3, n_classes=1,bilinear=True, task="Congestion")

    def forward(self, x):
        return self.uNet(x)

    def init_weights(self, pretrained=None, strict=True, **kwargs):
        super().init_weights(pretrained, strict)
