import torch.nn as nn
from functools import partial

def alexnetMap_register_hook(model, save_activation):
    """Register forward hook to all conv layers in alexnetMap model."""
    """for name, m in model.rgb_features.named_modules():
        if isinstance(m, nn.Conv2d):
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'rgbFeat_'+name))
    for name, m in model.d_features.named_modules():
        if isinstance(m, nn.Conv2d):
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'dFeat_'+name))"""
    for name, m in model.features.named_modules():
        if isinstance(m, nn.Conv2d) and name != '15':
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'feat_'+name))


def alexnet_register_hook(model, save_activation):
    """Register forward hook to all conv layers in alexnetMap model."""
    for name, m in model.features.named_modules():
        if isinstance(m, nn.Conv2d) and name != '0':
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'feat_'+name))


def alexnet_ductran_register_hook(model, save_activation):
    """Register forward hook to all conv layers in alexnetMap model."""
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'feat_'+name))