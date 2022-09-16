"""
This file contains all the functions needed for evaluating
a model.

All functions use the test dataset currently. If you wish to use the
train dataset, simply change 'params.TEST_PATH' to 'params.TRAIN_PATH'
on the DataLoader lines.
"""

import torch
import cv2
import random

import numpy as np
import torch.nn.functional as F

from parameters import Params
from data_loader import DataLoader
from utils import get_correct_preds, get_acc
from grasp_utils import get_correct_grasp_preds, grasps_to_bboxes, box_iou

params = Params()

def get_test_acc(model):
    """Returns the test accuracy and loss of a CLS model."""
    data_loader = DataLoader(params.TEST_PATH, 2, params.TRAIN_VAL_SPLIT)

    loss = 0
    correct = 0
    total = 0
    for (img, label) in data_loader.load_batch():
        output = model(img)
        loss += torch.nn.CrossEntropyLoss()(output, label).item()
        label = F.one_hot(label, num_classes=params.NUM_CLASS)
        batch_correct, batch_total = get_correct_preds(output, label)
        correct += batch_correct
        total += batch_total
    
    accuracy = get_acc(correct, total)

    return accuracy, round(loss / total, 3)


def get_grasp_acc(model):
    """Returns the test accuracy and loss of a Grasp model."""
    data_loader = DataLoader(params.TEST_PATH, 2, params.TRAIN_VAL_SPLIT)

    loss = 0
    correct = 0
    total = 0
    for (img, label, candidates) in data_loader.load_grasp():
        output = model(img)
        loss += torch.nn.MSELoss()(output, label).item()
        batch_correct, batch_total = get_correct_grasp_preds(output, [candidates])
        correct += batch_correct
        total += batch_total
        if batch_correct == 0:
            plot_grasp(img, output, label, candidates)
    
    accuracy = get_acc(correct, total)

    return accuracy, round(loss / total, 3)


def visualize_grasp(model):
    """Visualize the model's grasp predictions on test images one by one."""
    data_loader = DataLoader(params.TRAIN_PATH, 2, params.TRAIN_VAL_SPLIT)

    for (img, label, candidates) in data_loader.load_grasp():
        output = model(img)
        print(output, label)
        plot_grasp(img, output, label, candidates)
        

def plot_grasp(img, output, label, candidates):
    """Plots the relevant grasping boxes on the test images and prints out
    the maximum iou and minimum angle difference of the model's prediction.
    
    Grasping boxes visualized:
        - Model prediction (blue)
        - Training label (black)
        - 20% of candidate labels (green)
    Remarks: grasp plate positions are all colored RED

    """
    output_bbox = grasps_to_bboxes(output)
    target_bboxes = grasps_to_bboxes(candidates)
    label_bbox = grasps_to_bboxes(label)

    # Get iou
    max_iou = 0
    min_angle_diff = torch.tensor(0)
    for i in range(len(target_bboxes)):
        iou = box_iou(output_bbox[0], target_bboxes[i])
        pre_theta = output[0][2] * 180 - 90
        target_theta = candidates[i][2] * 180 - 90
        angle_diff = torch.abs(pre_theta - target_theta)
        
        if iou > max_iou:
            max_iou = iou
            min_angle_diff = angle_diff
    print(max_iou, min_angle_diff.item())

    img_vis = np.array(img.cpu())
    img_r = np.clip((img_vis[:, 0, :, :] * 0.229 + 0.485) * 255, 0, 255)
    img_g = np.clip((img_vis[:, 1, :, :] * 0.224 + 0.456) * 255, 0, 255)
    img_b = np.clip((img_vis[:, 2, :, :] * 0.225 + 0.406) * 255, 0, 255)
    img_d = img_vis[:, 2, :, :][0]
    
    img_bgr = np.concatenate((img_b, img_g, img_r), axis=0)
    img_bgr = np.moveaxis(img_bgr, 0, -1)
    img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
    
    draw_bbox(img_bgr, output_bbox[0], (255, 0, 0))
    draw_bbox(img_bgr, label_bbox[0], (0, 0, 0))
    for bbox in target_bboxes:
        # Choose some 20% random bboxes to show:
        if random.randint(0, 5) == 0:
            draw_bbox(img_bgr, bbox, (0, 255, 0))

    cv2.imshow('img', img_bgr)
    cv2.waitKey(0)


def draw_bbox(img, bbox, color):
    """Draw grasp boxes with the grasp-plate edges as RED and the
    other two edges as <color>."""
    x1 = int(bbox[0] / 1024 * params.OUTPUT_SIZE)
    y1 = int(bbox[1] / 1024 * params.OUTPUT_SIZE)
    x2 = int(bbox[2] / 1024 * params.OUTPUT_SIZE)
    y2 = int(bbox[3] / 1024 * params.OUTPUT_SIZE)
    x3 = int(bbox[4] / 1024 * params.OUTPUT_SIZE)
    y3 = int(bbox[5] / 1024 * params.OUTPUT_SIZE)
    x4 = int(bbox[6] / 1024 * params.OUTPUT_SIZE)
    y4 = int(bbox[7] / 1024 * params.OUTPUT_SIZE)
    cv2.line(img, (x1, y1), (x2, y2), color, 1)
    cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 1)
    cv2.line(img, (x3, y3), (x4, y4), color, 1)
    cv2.line(img, (x4, y4), (x1, y1), (0, 0, 255), 1)
