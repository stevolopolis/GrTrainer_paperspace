from re import I
import torch
import torch.nn.functional as F

from parameters import Params

params = Params()

def MapLoss(output, map):
    output = torch.moveaxis(output, 1, -1)
    map = torch.moveaxis(map, 1, -1)

    # Best for CLS
    confidence_loss = nll_loss(output[:, :, :, 5], map[:, :, :, 5])
    #confidence_loss = torch.pow(output[:, :, :, 5] - map[:, :, :, 5], 2)
    #confidence_loss = torch.abs(output[:, :, :, 5] - map[:, :, :, 5])
    confidence_loss = torch.mean(confidence_loss)
    # L2 loss (grasp)
    #bbox_loss = torch.pow(output[:, :, :, :5] - map[:, :, :, :5], 2)
    # L1 loss (grasp)
    bbox_loss = torch.abs(output[:, :, :, :5] - map[:, :, :, :5])
    # Cross Entropy Loss (cls)
    """clsLoss = torch.nn.BCELoss(reduction='none')
    softmax = torch.nn.Softmax(dim=3)
    output_softmax = softmax(output[:, :, :, :5])
    #output_softmax = torch.moveaxis(output_softmax, -1, 1)
    label_softmax = map[:, :, :, :5]
    #label_softmax = torch.moveaxis(label_softmax, -1, 1)
    bbox_loss = clsLoss(output_softmax, label_softmax)
    #bbox_loss = torch.sum(bbox_loss, dim=3)"""
    valid_pixel = map[:, :, :, 5] != 0.0
    valid_pixel = torch.unsqueeze(valid_pixel, dim=3)
    valid_pixel = torch.cat((valid_pixel, valid_pixel, valid_pixel, valid_pixel, valid_pixel), dim=3)
    bbox_loss = bbox_loss * valid_pixel
    bbox_loss = torch.mean(bbox_loss)

    return confidence_loss + bbox_loss * 2


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


def BoundingLoss(output, target):
    """Returns L1 distance if output is counted as a valid grasp,
       and returns higher loss if not."""
    pass


def CLSLoss(output, target):
    """l1_loss = F.l1_loss(output, target)
    output_sum = torch.sum(output, dim=1)
    sum_diff = torch.abs(1 - output_sum)
    sum_cond = sum_diff < 1
    sum_loss = - torch.log(torch.where(sum_cond, sum_diff, torch.ones_like(sum_diff, dtype=torch.float32)))
    loss = l1_loss + sum_loss"""
    loss = nll_loss(output, target)

    return torch.sum(loss) / loss.size(0)


def WeightedL2Loss(output, target):
    mse = torch.pow(output - target, 2)
    cond = [1, 1, 0, 0, 0]
    loss = torch.where(cond, 2*mse, mse)

    return torch.sum(loss) / loss.size(0)


def BCEL1Loss(output, target):
    """Returns BCELoss for when output is in [0, 1], and
    returns L1Loss for when output is not in [0, 1].

    Implemented using the format of smooth_l1_loss.
    """
    smaller_than_one = output < .9
    greater_than_zero = output > .1

    condition = smaller_than_one == greater_than_zero
    loss = torch.where(condition, nll_loss(output, target), torch.abs(output - target) + .1)
    
    return torch.sum(loss) / loss.size(0)


def nll_loss(output, target):
    return - (torch.log(output + 1e-5) * target + torch.log(1 - output + 1e-5) * (1 - target))