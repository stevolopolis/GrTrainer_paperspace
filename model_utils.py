import torch.nn as nn
from functools import partial

from inference.models.grasp_model import GraspModel, ResidualBlock

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
    for name, m in model.rgb_features.named_modules():
        if isinstance(m, nn.Conv2d):
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'rgb_features.0'))
    for name, m in model.features.named_modules():
        if isinstance(m, nn.Conv2d) and name in ['0', '4', '7', '10']:
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'features.'+name))


def alexnet_register_hook(model, save_activation):
    """Register forward hook to all conv layers in alexnet-imagenet model."""
    for name, m in model.features.named_modules():
        if isinstance(m, nn.Conv2d):
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'feat_'+name))


def alexnet_ductran_register_hook(model, save_activation):
    """Register forward hook to all conv layers in alexnet-cornell model."""
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'feat_'+name))


def resnet_register_hook(model, save_activation):
    """Register forward hook to all conv layers in resnet-imagenet model."""
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and ('conv' in name) and not ('layer' in name and ('conv1' in name or '.0.' in name)):
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'feat_'+name))


def grconvnet_kumra_register_hook(model, save_activation):
    """Register forward hook to all conv layers in gr-convnet model."""
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and ('conv' in name) and not ('res' in name and 'conv1' in name):
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'feat_' + '_'.join(name.split('.'))))