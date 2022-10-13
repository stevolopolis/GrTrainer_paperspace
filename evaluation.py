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
from PIL import Image, ImageDraw

from parameters import Params
from data_loader_v2 import DataLoader
from tqdm import tqdm
from utils import get_correct_preds, get_acc, get_correct_cls_preds_from_map
from grasp_utils import get_correct_grasp_preds, grasps_to_bboxes, box_iou, map2grasp, get_max_grasp

params = Params()

def get_cls_acc(model, dataset=params.TEST_PATH):
    """Returns the test accuracy and loss of a CLS model."""
    data_loader = DataLoader(dataset, 2, params.TRAIN_VAL_SPLIT, verbose=False)

    loss = 0
    correct = 0
    total = 0
    for (img, cls_map, label) in data_loader.load_batch():
        output = model(img)
        batch_correct, batch_total = get_correct_cls_preds_from_map(output, label)
        correct += batch_correct
        total += batch_total
    
    accuracy = get_acc(correct, total)

    return accuracy, round(loss / total, 3)


def get_grasp_acc(model):
    """Returns the test accuracy and loss of a Grasp model."""
    data_loader = DataLoader(params.TEST_PATH, 1, params.TRAIN_VAL_SPLIT)

    loss = 0
    correct = 0
    total = 0

    pbar = tqdm(total=data_loader.n_data)
    for (img, candidates) in data_loader.load_grasp_old():
        output = model(img)
        # Move grasp channel to the end
        output = torch.moveaxis(output, 1, -1)
        # Convert grasp map into single grasp prediction
        output_grasp = map2singlegrasp(output)
        
        batch_correct, batch_total = get_correct_grasp_preds(output_grasp, [candidates])
        correct += batch_correct
        total += batch_total

        # Uncomment if visualize grasp for one instance
        #plot_grasp(img, output_grasp, candidates)

        pbar.update(1)
    
    accuracy = get_acc(correct, total)

    return accuracy, round(loss / total, 3)


def get_confidence_map(img, output):
    """Visualize the model's grasp confidences on test images one by one."""
    # Formatting image for visualization
    img_vis = np.array(img.cpu())
    img_r = np.clip((img_vis[:, 0, :, :] * 0.229 + 0.485) * 255, 0, 255)
    img_g = np.clip((img_vis[:, 1, :, :] * 0.224 + 0.456) * 255, 0, 255)
    img_b = np.clip((img_vis[:, 2, :, :] * 0.225 + 0.406) * 255, 0, 255)
    img_d = img_vis[:, 2, :, :][0]
    img_bgr = np.concatenate((img_b, img_g, img_r), axis=0)
    img_bgr = np.moveaxis(img_bgr, 0, -1)
    # Get confidence map
    conf_map = output[0, 5, :, :]
    conf_map = conf_map * 255
    conf_map = torch.unsqueeze(conf_map, 0)
    conf_map = torch.clip(conf_map, 0, 255)
    conf_map = torch.moveaxis(conf_map, 0, -1)
    conf_map_rgb = torch.cat((conf_map*0.1, conf_map*0.1, conf_map), -1).detach().cpu().numpy()
    # Combine confidence heatmap with rgb image
    img_bgr = conf_map_rgb + img_bgr

    vis_img = np.clip(img_bgr, 0, 255)
    vis_img = np.ascontiguousarray(vis_img, dtype=np.uint8)

    return vis_img    


def visualize_cls(model):
    """Visualize the model's grasp predictions on test images one by one."""
    data_loader = DataLoader(params.TEST_PATH, 1, params.TRAIN_VAL_SPLIT)

    for i, (img, cls_map, label) in enumerate(data_loader.load_cls()):
        output = model(img)
        print(output[0, :, 112, 112])
        print(cls_map[0, :, 112, 112])

        model_cls_map = get_cls_map(img, output[0])
        true_cls_map = get_cls_map(img, cls_map[0])
        raw_img = denormalize_img(img)

        ref_cls_1 = get_cls_map(img, cls_map[0], color_roll=1)
        ref_cls_2 = get_cls_map(img, cls_map[0], color_roll=2)
        ref_cls_3 = get_cls_map(img, cls_map[0], color_roll=3)
        ref_cls_4 = get_cls_map(img, cls_map[0], color_roll=4)
        ref_cls_map = get_ref_map(ref_cls_1, ref_cls_2, ref_cls_3, ref_cls_4, label)

        vis_img = np.concatenate((model_cls_map, true_cls_map, raw_img, ref_cls_map), 1)
        cv2.imshow('vis', vis_img)
        cv2.waitKey(0)
        cv2.imwrite('vis/%s.png' % i, vis_img)


def visualize_grasp(model):
    """Visualize the model's grasp predictions on test images one by one."""
    data_loader = DataLoader(params.TEST_PATH, 1, params.TRAIN_VAL_SPLIT)

    #for (img, cls_map, label) in data_loader.load_cls():
    for i, (img, grasp_map) in enumerate(data_loader.load_grasp()):
        output = model(img)
        # Get confidence map
        conf_on_rgb = get_confidence_map(img, output)
        # Move grasp channel to the end
        output = torch.moveaxis(output, 1, -1)
        grasp_map = torch.moveaxis(grasp_map, 1, -1)
        # Convert grasp map into single grasp prediction
        output_grasp = map2singlegrasp(output)
        # Denoramlize grasps
        denormalize_grasp(grasp_map)
        # Convert grasp maps into grasp candidate tensors
        target_grasps = map2grasp(grasp_map[0])
        
        # Get grasps map on the rgb image
        model_grasp_map = get_grasp_map(conf_on_rgb, output_grasp, target_grasps, vis_truth=False)
        true_grasp_map = get_grasp_map(img, output_grasp, target_grasps, vis_model=False)

        vis_img = np.concatenate((model_grasp_map, true_grasp_map), 1)
        cv2.imshow('vis', vis_img)
        cv2.waitKey(0)
        cv2.imwrite('vis/%s.png' % i, vis_img)


def map2singlegrasp(output):
    """Convert output grasp map into single grasp prediction."""
    # Denoramlize grasps
    denormalize_grasp(output)
    # Get grasp output with max confidence
    max_grasp_x, max_grasp_y = get_max_grasp(output[0])
    output_grasp = output[0, max_grasp_y, max_grasp_x, :5]
    # Model grasp to grasp parameters
    output_grasp[0] = (max_grasp_x + output_grasp[0]) / params.OUTPUT_SIZE
    output_grasp[1] = (max_grasp_y + output_grasp[1]) / params.OUTPUT_SIZE
    output_grasp[3] = output_grasp[3] / params.OUTPUT_SIZE
    # Unsqeeze grasp tensor
    output_grasp = torch.unsqueeze(output_grasp, 0)

    return output_grasp


def denormalize_grasp(grasp_map):
    # Denormalize x-coord
    grasp_map[:, :, :, 0] *= params.OUTPUT_SIZE
    # Denormalize y-coord
    grasp_map[:, :, :, 1] *= params.OUTPUT_SIZE
    # Denormalize width
    grasp_map[:, :, :, 3] *= params.OUTPUT_SIZE


def get_ref_map(ref1, ref2, ref3, ref4, cls_idx):
    CLS = ['Chair (0)', 'Lamp (1)', 'figurines (2)', 'plants (3)', 'pen+pencil (4)']

    ref_map_1 = np.concatenate((ref1, ref2), axis=0)
    ref_map_2 = np.concatenate((ref3, ref4), axis=0)
    ref_map = np.concatenate((ref_map_1, ref_map_2), axis=1)
    ref_map = cv2.resize(ref_map, (ref1.shape[0], ref1.shape[1]))
    ref_map_with_text = Image.fromarray(ref_map)
    draw = ImageDraw.Draw(ref_map_with_text)
    draw.text((5, 5), CLS[(cls_idx+1) % 5])
    draw.text((ref_map.shape[0] // 2 + 5, 5), CLS[(cls_idx+2) % 5])
    draw.text((5, ref_map.shape[0] // 2 + 5), CLS[(cls_idx+3) % 5])
    draw.text((ref_map.shape[0] // 2 + 5, ref_map.shape[0] // 2 + 5), CLS[(cls_idx+4) % 5])

    return np.array(ref_map_with_text)


def get_cls_map(img, cls_map, color_roll=0):
    if not type(img) == np.ndarray:
        img_bgr = denormalize_img(img)
    else:
        img_bgr = img

    cls_map = torch.moveaxis(cls_map, 0, -1)
    conf_map = torch.unsqueeze(cls_map[:, :, 5], -1)
    conf_map = torch.cat((conf_map, conf_map, conf_map, conf_map, conf_map), 2)
    pred_map = cls_map[:, :, :5]
    pred_val_map = conf_map * pred_map
    weighted_pred_map, idx_map = torch.max(pred_val_map, 2)
    weighted_pred_map = weighted_pred_map / torch.max(weighted_pred_map)

    weighted_pred_map = torch.unsqueeze(weighted_pred_map, -1)
    weighted_pred_map = torch.cat((weighted_pred_map, weighted_pred_map, weighted_pred_map), -1)

    idx_map = torch.unsqueeze(idx_map, 2).type(torch.float32)
    sub_color = torch.full((idx_map.shape[0], idx_map.shape[1], 1), 0.2)
    main_color = torch.full((idx_map.shape[0], idx_map.shape[1], 1), 0.95)
    color1 = torch.cat((sub_color, sub_color, main_color), 2).to(params.DEVICE)
    color2 = torch.cat((sub_color, main_color, sub_color), 2).to(params.DEVICE)
    color3 = torch.cat((main_color, sub_color, sub_color), 2).to(params.DEVICE)
    color4 = torch.cat((sub_color, main_color, main_color), 2).to(params.DEVICE)
    color5 = torch.cat((main_color, sub_color, main_color), 2).to(params.DEVICE)
    color_list = [color1, color2, color3, color4, color5]
    idx_map = torch.where(idx_map == 0, color_list[(0+color_roll) % 5], idx_map)
    idx_map = torch.where(idx_map == 1, color_list[(1+color_roll) % 5], idx_map)
    idx_map = torch.where(idx_map == 2, color_list[(2+color_roll) % 5], idx_map)
    idx_map = torch.where(idx_map == 3, color_list[(3+color_roll) % 5], idx_map)
    idx_map = torch.where(idx_map == 4, color_list[(4+color_roll) % 5], idx_map)

    cls_color_map = weighted_pred_map * idx_map * 255
    colored_img = img_bgr + cls_color_map.detach().cpu().numpy()
    colored_img = np.clip(colored_img, 0, 255)

    return np.ascontiguousarray(colored_img, dtype=np.uint8)


def get_grasp_map(img, output, candidates, vis_model=True, vis_truth=True):
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
    
    if not type(img) == np.ndarray:
        img_bgr = denormalize_img(img)
    else:
        img_bgr = img
    
    if vis_model:
        for bbox in output_bbox:
            draw_bbox(img_bgr, bbox, (255, 0, 0), 1)
    if vis_truth:
        for bbox in target_bboxes:
            # Choose some 20% random bboxes to show:
            if random.randint(0, 5) == 0:
                draw_bbox(img_bgr, bbox, (255, 255, 255), 1)

    return img_bgr


def draw_bbox(img, bbox, color, width):
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
    cv2.line(img, (x1, y1), (x2, y2), color, width)
    cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 1)
    cv2.line(img, (x3, y3), (x4, y4), color, width)
    cv2.line(img, (x4, y4), (x1, y1), (0, 0, 255), 1)


def get_iou(output_grasps, target_grasps):
    output_bbox = grasps_to_bboxes(output_grasps)
    target_bboxes = grasps_to_bboxes(target_grasps)
    max_iou = 0
    min_angle = 180
    for i in range(len(target_bboxes)):
        iou = box_iou(output_bbox[0], target_bboxes[i])
        pre_theta = output_grasps[0][2] * 180 - 90
        target_theta = target_grasps[i][2] * 180 - 90
        angle_diff = torch.abs(pre_theta - target_theta)
        if angle_diff < min_angle and iou > max_iou:
            max_iou = iou
            min_angle = angle_diff

    return max_iou, min_angle.item()
        

def denormalize_img(img):
    img_vis = np.array(img.cpu())
    img_r = np.clip((img_vis[:, 0, :, :] * 0.229 + 0.485) * 255, 0, 255)
    img_g = np.clip((img_vis[:, 1, :, :] * 0.224 + 0.456) * 255, 0, 255)
    img_b = np.clip((img_vis[:, 2, :, :] * 0.225 + 0.406) * 255, 0, 255)
    
    img_bgr = np.concatenate((img_b, img_g, img_r), axis=0)
    img_bgr = np.moveaxis(img_bgr, 0, -1)
    img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)

    return img_bgr
