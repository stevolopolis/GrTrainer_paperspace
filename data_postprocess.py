"""
This file contains functions and code used to revert the 
preprocessing pipeline. This is mainly used to visualize the
dataset after running through DataPreProcessor to make sure
any transformations made does not alter the dataset content
in an unintended way.
"""

import torch
import random
import cv2
import os
import numpy as np

from torchvision import transforms

from grasp_utils import grasps_to_bboxes
from parameters import Params

params = Params()


def grasps_to_bboxes(grasps):
    """Converts grasp boxes to bounding boxes."""
    # convert grasp representation to bbox
    x = grasps[:,0] * 1024
    y = grasps[:,1] * 1024
    theta = torch.deg2rad(grasps[:,2] * 180 - 90)
    w = grasps[:,3] * 1024
    h = grasps[:,4] * 100
    
    x1 = x -w/2*torch.cos(theta) +h/2*torch.sin(theta)
    y1 = y -w/2*torch.sin(theta) -h/2*torch.cos(theta)
    x2 = x +w/2*torch.cos(theta) +h/2*torch.sin(theta)
    y2 = y +w/2*torch.sin(theta) -h/2*torch.cos(theta)
    x3 = x +w/2*torch.cos(theta) -h/2*torch.sin(theta)
    y3 = y +w/2*torch.sin(theta) +h/2*torch.cos(theta)
    x4 = x -w/2*torch.cos(theta) -h/2*torch.sin(theta)
    y4 = y -w/2*torch.sin(theta) +h/2*torch.cos(theta)
    bboxes = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), 1)
    return bboxes


def process(rgb, d):
        """
        Returns rgbd image with correct format for inputing to model:
            - Imagenet normalization
            - Concat depth channel to image
        """
        transformation_rgb = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        rgb = rgb / 255.0
        rgb = torch.moveaxis(rgb, -1, 0)
        rgb = transformation_rgb(rgb)
        if d is None:
            img = rgb
        elif params.NUM_CHANNEL == 3:
            # Input channels -- (gray, gray, depth)
            #rgb = transforms.Grayscale(num_output_channels=1)(rgb)
            #rgb = torch.cat((rgb, rgb), axis=0)
            # Input channels -- (red, green, depth)
            d = torch.unsqueeze(d, 2)
            d = d - torch.mean(d)
            d = torch.clip(d, -1, 1)
            d = torch.moveaxis(d, -1, 0)
            img = torch.cat((rgb[:2], d), axis=0)
        else:
            # Input channels -- (red, green, blue, depth)
            d = torch.unsqueeze(d, 2)
            d = d - torch.mean(d)
            d = torch.clip(d, -1, 1)
            d = torch.moveaxis(d, -1, 0)
            img = torch.cat((rgb, d), axis=0)

        img = torch.unsqueeze(img, 0)
        img = img.to(params.DEVICE)

        return img


class DataPostProcessor:
    def __init__(self):
        pass

    def map2grasp(self, map):
        grasp_candidates = set()
        for i in range(len(map)):
            for j in range(len(map[i])):
                if map[i][j][5] == 0:
                    continue
                grasp_x = (j + map[i][j][0]) / params.OUTPUT_SIZE
                grasp_y = (i + map[i][j][1]) / params.OUTPUT_SIZE
                grasp_theta = map[i][j][2]
                grasp_w = map[i][j][3] / params.OUTPUT_SIZE
                grasp_h = map[i][j][4]
                grasp = (grasp_x, grasp_y, grasp_theta, grasp_w, grasp_h)
                grasp_candidates.add(grasp)

        grasp_candidates = torch.tensor(list(grasp_candidates))
        return grasp_candidates

    def grasp2bbox(self, candidates):
        bboxes = grasps_to_bboxes(candidates)
        return bboxes

    def visualize_grasp(self, img, candidates):
        target_bboxes = grasps_to_bboxes(candidates)

        img_vis = np.array(img.cpu())
        img_r = np.clip((img_vis[:, 0, :, :] * 0.229 + 0.485) * 255, 0, 255)
        img_g = np.clip((img_vis[:, 1, :, :] * 0.224 + 0.456) * 255, 0, 255)
        img_b = np.clip((img_vis[:, 2, :, :] * 0.225 + 0.406) * 255, 0, 255)
        
        img_bgr = np.concatenate((img_b, img_g, img_r), axis=0)
        img_bgr = np.moveaxis(img_bgr, 0, -1)
        img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
        
        for bbox in target_bboxes:
            # Choose some 20% random bboxes to show:
            #if random.randint(0, 5) == 0:
            self.draw_bbox(img_bgr, bbox, (0, 255, 0))

        cv2.imshow('img', img_bgr)
        cv2.waitKey(0)

    def draw_bbox(self, img, bbox, color):
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


def load_grasp_label(file_path):
        """Returns a list of grasp labels from <file_path>."""
        grasp_list = []
        with open(file_path, 'r') as f:
            file = f.readlines()
            # dat format in each line: 'x;y;theta;w;h'
            for grasp in file:
                # remove '\n' from string
                grasp = grasp[:-1]
                label = grasp.split(';')
                label = noramlize_grasp(label)
                grasp_list.append(label)

        return grasp_list


def noramlize_grasp(label):
        """Returns normalize grasping labels."""
        norm_label = []
        for i, value in enumerate(label):
            if i == 4:
                # Height
                norm_label.append(float(value) / 100)
            elif i == 2:
                # Theta
                norm_label.append((float(value) + 90) / 180)
            elif i == 3:
                # Width
                norm_label.append(float(value) / 1024)
            else:
                # Coordinates
                norm_label.append(float(value) / 1024)

        return norm_label


if __name__ == '__main__':
    path = 'data/top_5_compressed/train'
    ori_path = 'data/top_5/train'
    processor = DataPostProcessor()
    for cls in os.listdir(path):
        for img_id in os.listdir(os.path.join(path, cls)):
            for img_id_with_var in os.listdir(os.path.join(path, cls, img_id)): 
                if not img_id_with_var.endswith('RGB.npy'):
                    continue
                
                var = img_id_with_var[:2]
                rgb_name = var + img_id + '_RGB.npy'
                d_name = var + img_id + '_perfect_depth.npy'
                map_name = var + img_id + '_grasps.npy'

                rgb = np.load(open(os.path.join(path, cls, img_id, rgb_name), 'rb'))
                d = np.load(open(os.path.join(path, cls, img_id, d_name), 'rb'))
                map = np.load(open(os.path.join(path, cls, img_id, map_name), 'rb'))

                true_grasps = load_grasp_label(os.path.join(ori_path, cls, img_id, var + img_id + '_grasps.txt'))
                true_grasps = torch.tensor(true_grasps)

                input_img = process(torch.tensor(rgb), torch.tensor(d))

                grasps = processor.map2grasp(map)
                processor.visualize_grasp(input_img, grasps)
                processor.visualize_grasp(input_img, true_grasps)
