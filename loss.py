import torch
import torch.nn as nn
import torch.nn.functional as F

import collections
from functools import partial

from parameters import Params
from model_utils import *

params = Params()

def MapLoss(output, map):
    output = torch.moveaxis(output, 1, -1)
    map = torch.moveaxis(map, 1, -1)

    confidence_loss = nll_loss(output[:, :, :, 5], map[:, :, :, 5])
    confidence_loss = torch.mean(confidence_loss)

    bbox_loss = logl1Loss(output[:, :, :, :5], map[:, :, :, :5])

    valid_pixel = map[:, :, :, 5] != 0.0
    valid_pixel = torch.unsqueeze(valid_pixel, dim=3)
    valid_pixel = torch.cat((valid_pixel, valid_pixel, valid_pixel, valid_pixel, valid_pixel), dim=3)
    bbox_loss = bbox_loss * valid_pixel
    bbox_loss = torch.mean(bbox_loss)

    return confidence_loss + bbox_loss * 2


def logl1Loss(output, target):
    left_loss = - (torch.log(1 + ((1/(1+target+1e-5))*(output-target))))
    right_loss = - (torch.log(1 + ((1/(1-target+1e-5))*(target-output))))
    nll = torch.where(output < target, left_loss, right_loss)

    return nll

def minLossTarget(output, target_candidates):
    """Returns the min. MSE loss (distance) between the output
    and all target candidates."""
    for i in range(len(output)):
        loss = torch.pow(output[i] - target_candidates[i], 2)
        loss = torch.sum(loss, 1)
        min_loss = torch.min(loss, 0)[1]
        min_target = torch.unsqueeze(target_candidates[i][min_loss], dim=0)
        if i == 0:
            min_loss_batch = min_target
        else:
            min_loss_batch = torch.cat((min_loss_batch, min_target), dim=0)

    return min_loss_batch


def nll_loss(output, target):
    return - (torch.log(output + 1e-5) * target + torch.log(1 - output + 1e-5) * (1 - target))


def DistillationLoss(img_in, model_s, model_t, model_s_type='alexnetMap', model_t_type='alexnet'):
    loss = 0
    feature_s = get_model_features(img_in, model_s, model_type=model_s_type)
    feature_t = get_model_features(img_in, model_t, model_type=model_t_type)
    for i, feat_t in enumerate(feature_t):
        if feature_t[feat_t].shape == feature_s[list(feature_s.keys())[i]].shape:
            loss += F.mse_loss(feature_t[feat_t], feature_s[list(feature_s.keys())[i]])

    return loss


def get_model_features(img_in, model, model_type='alexnetMap'):
    activations = {}
    def save_activation(name, mod, inp, out):
        activations[name] = out
    
    hook_activations(model, save_activation, model_type=model_type)

    model_forward_pass(img_in, model,model_type=model_type)

    return activations


def hook_activations(model, save_activation, model_type='alexetMap'):
    # Save activation maps of all encoder conv layers in alexnetMap model
    if model_type == 'alexnetMap':
        alexnetMap_register_hook(model, save_activation)
    # Save activation maps of all encoder conv layers in alexnet (Imagenet)
    elif model_type == 'alexnet':
        alexnet_register_hook(model, save_activation)
    # Save activation maps of all encoder conv layers in alexnet (Cornell)
    elif model_type == 'alexnet_ductran':
        alexnet_ductran_register_hook(model, save_activation)


def model_forward_pass(img_in, model, model_type='alexnetMap'):
    if model_type == 'alexnetMap':
        _ = model(img_in)
    else:
        img_in = img_in[:, :3, :, :]
        _ = model(img_in)