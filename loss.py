from re import I
import torch
import torch.nn.functional as F

from parameters import Params

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