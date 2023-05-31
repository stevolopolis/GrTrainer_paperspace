import torch
import numpy as np

from shapely.geometry import Polygon
from math import pi 

from parameters import Params

params = Params()


def map2grasp(map):
    map = np.array(map.cpu().detach())
    grasp_candidates = set()
    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j][5] == 0:
                continue
            grasp_x = (j + map[i][j][0]) / params.OUTPUT_SIZE
            grasp_y = (i + map[i][j][1]) / params.OUTPUT_SIZE
            grasp_theta = map[i][j][2]
            grasp_w = map[i][j][3] / params.OUTPUT_SIZE
            grasp_h = map[i][j][4] / 2 + 0.5
            grasp = (grasp_x, grasp_y, grasp_theta, grasp_w, grasp_h)
            grasp_candidates.add(grasp)

    grasp_candidates = torch.tensor(list(grasp_candidates))
    return grasp_candidates


def get_max_grasp(map):
    confidence_map = map[:, :, :, 5]
    flattened_confidence_map = torch.reshape(confidence_map, (confidence_map.shape[0], -1))
    max_idx = torch.max(flattened_confidence_map, 1)[1]
    max_x_idx = max_idx % confidence_map.shape[1]
    max_y_idx = max_idx // confidence_map.shape[2]

    return max_x_idx, max_y_idx


def bboxes_to_grasps(bboxes):
    """Converts bounding boxxes to grasp boxes."""
    # convert bbox to grasp representation -> tensor([x, y, theta, h, w])
    x = bboxes[:,0] + (bboxes[:,4] - bboxes[:,0])/2
    y = bboxes[:,1] + (bboxes[:,5] - bboxes[:,1])/2 
    theta = torch.atan((bboxes[:,3] -bboxes[:,1]) / (bboxes[:,2] -bboxes[:,0]))
    w = torch.sqrt(torch.pow((bboxes[:,2] -bboxes[:,0]), 2) + torch.pow((bboxes[:,3] -bboxes[:,1]), 2))
    h = torch.sqrt(torch.pow((bboxes[:,6] -bboxes[:,0]), 2) + torch.pow((bboxes[:,7] -bboxes[:,1]), 2))
    grasps = torch.stack((x, y, theta, h, w), 1)
    return grasps


def grasps_to_bboxes(grasps):
    """Converts grasp boxes to bounding boxes."""
    # convert grasp representation to bbox
    x = grasps[:, :, 0] * 1024
    y = grasps[:, :, 1] * 1024
    theta = torch.deg2rad(grasps[:, :,2] * 180 - 90)
    w = grasps[:, :, 3] * 1024
    h = grasps[:, :, 4] * 100
    
    x1 = x -w/2*torch.cos(theta) +h/2*torch.sin(theta)
    y1 = y -w/2*torch.sin(theta) -h/2*torch.cos(theta)
    x2 = x +w/2*torch.cos(theta) +h/2*torch.sin(theta)
    y2 = y +w/2*torch.sin(theta) -h/2*torch.cos(theta)
    x3 = x +w/2*torch.cos(theta) -h/2*torch.sin(theta)
    y3 = y +w/2*torch.sin(theta) +h/2*torch.cos(theta)
    x4 = x -w/2*torch.cos(theta) -h/2*torch.sin(theta)
    y4 = y -w/2*torch.sin(theta) +h/2*torch.cos(theta)
    bboxes = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), 2)
    return bboxes


def box_iou(bbox_value, bbox_target):
    """Returns the iou between <bbox_value> and <bbox_target>."""
    p1 = Polygon(bbox_value.view(-1,2).tolist())
    p2 = Polygon(bbox_target.view(-1,2).tolist())
    intersection = p1.intersection(p2).area
    iou = intersection / (p1.area + p2.area - intersection) 
    return iou


def get_correct_grasp_preds(output, target):
    """Returns number of correct predictions out of number of instances.
    
    A correct prediction is defined as a grasp prediction that meets
    the following criterions with at least one grasp candidate:
        - iou > 0.25
        - angle difference < 30
    """
    bbox_outputs = grasps_to_bboxes(output)
    bbox_targets = grasps_to_bboxes(target)

    pre_theta = output[:, :, 2] * 180 - 90
    target_theta = target[:, :, 2] * 180 - 90
    angle_diff = torch.abs(pre_theta - target_theta)
    correct = 0
    for i in range(len(target)):
        for j in range(len(bbox_targets[i])):
            # Check if leftover bbox candidates are repeated (for data_loader speedup)
            if j < len(bbox_targets[i]) - 1:
                if torch.sum(bbox_targets[i, j, :] != bbox_targets[i, j + 1, :]) == 0:
                    break
            # Catch value error caused by invalid box (i.e. model not outputing valid box)
            try:
                iou = box_iou(bbox_outputs[i][j], bbox_targets[i][j])
            except ValueError:
                break
            
            if angle_diff[i][j] < 30 and iou > 0.25:
                correct += 1
                break

    return correct, len(target)


def get_correct_grasp_preds_from_map(output, target):
    """Returns number of correct predictions out of number of instances.
    
    A correct prediction is defined as a grasp prediction that meets
    the following criterions with at least one grasp candidate:
        - iou > 0.25
        - angle difference < 30
    """
    correct = 0

    max_grasp_x, max_grasp_y = get_max_grasp(output)
    
    output_grasp = output[[i for i in range(len(max_grasp_y))], max_grasp_y, max_grasp_x, :5]
    # Model grasp to grasp parameters
    output_grasp[:, 0] = (max_grasp_x + output_grasp[:, 0]) / params.OUTPUT_SIZE
    output_grasp[:, 1] = (max_grasp_y + output_grasp[:, 1]) / params.OUTPUT_SIZE
    output_grasp[:, 3] = output_grasp[:, 3] / params.OUTPUT_SIZE
    output_grasp[:, 4] = output_grasp[:, 4] / 2 + 0.5
    
    output_bbox = grasps_to_bboxes(output_grasp)

    target_grasp = target[[i for i in range(len(max_grasp_y))], max_grasp_y, max_grasp_x, :5]
    # Model grasp to grasp parameters
    target_grasp[:, 0] = (max_grasp_x + target_grasp[:, 0]) / params.OUTPUT_SIZE
    target_grasp[:, 1] = (max_grasp_y + target_grasp[:, 1]) / params.OUTPUT_SIZE
    target_grasp[:, 3] = target_grasp[:, 3] / params.OUTPUT_SIZE
    target_grasp[:, 4] = target_grasp[:, 4] / 2 + 0.5

    target_bbox = grasps_to_bboxes(target_grasp)

    for i in range(len(target_bbox)):
        iou = box_iou(output_bbox[i], target_bbox[i])
        pre_theta = output_grasp[i][2] * 180 - 90
        target_theta = target_grasp[i][2] * 180 - 90
        angle_diff = torch.abs(pre_theta - target_theta)
        
        if angle_diff < 30 and iou > 0.25:
            correct += 1

    return correct, len(target)
