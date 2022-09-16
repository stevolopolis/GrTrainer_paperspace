"""
This file contains the functions responsible for creating the
training and testing datasets in the <data> folder.

This file also contains a function that visualizes all grasp
candidates of an image.
"""

import glob
import os
import shutil
import cv2
import random
import numpy as np

from data_loader import DataLoader
from parameters import Params
from grasp_utils import grasps_to_bboxes

params = Params()

# Statistics of each class of data
"""{'Clothes_hanger': 5, 'walkman': 11, 'fish': 15, 'usb_drive': 24,
    'violin': 28, 'toy_car': 31, 'insect': 35, 'fork': 37, 'bed': 42,
    'Photo_Frame': 44, 'table': 59, 'laptop': 64, 'can': 65, 'cup': 68,
    'cell_phone': 72, 'sword': 79, 'knife': 84, 'sofa': 86, 'vase': 107,
    'stool': 114, 'computer_monitor': 142, 'gun': 143, 'toy_plane': 155,
    'guitar': 175, 'bottle': 178, 'pen+pencil': 190, 'plants': 204,
    'figurines': 219, 'Lamp': 267, 'Chair': 389}"""

DATA_PATH = 'data'

top_5 = ['Chair', 'Lamp', 'figurines', 'plants', 'pen+pencil']  # cls instances -- 190
top_10 = ['gun', 'computer_monitor', 'toy_plane', 'guitar',
          'bottle', 'pen+pencil', 'plants', 'figurines', 'Lamp', 'Chair']  # cls instances -- 143

def create_test(top_n_list, top_n_str, n_test_per_class):
    for cls in top_n_list:
        move_count = 0
        for img_path in glob.iglob('%s/%s/train/%s/*/*' % (DATA_PATH, top_n_str, cls)):
            if not img_path.endswith('RGB.png'):
                continue
            if move_count >= n_test_per_class:
                continue

            move_count += 1
            # E.g. '<img_idx>_<img_id>_<img_type>.png'
            img_name = img_path.split('\\')[-1]
            img_var = img_name.split('_')[0]
            img_id = img_name.split('_')[1]

            if cls not in os.listdir(os.path.join(DATA_PATH, top_n_str, 'test')):
                os.mkdir(os.path.join(DATA_PATH, top_n_str, 'test', cls))
            if img_id not in os.listdir(os.path.join(DATA_PATH, top_n_str, 'test', cls)):
                os.mkdir(os.path.join(DATA_PATH, top_n_str, 'test', cls, img_id))

            for file in glob.glob('%s/%s/train/%s/%s/%s_%s*' % (DATA_PATH, top_n_str, cls, img_id, img_var, img_id)):
                name = file.split('\\')[-1]
                shutil.move(file, '%s/%s/test/%s/%s/%s' % (DATA_PATH, top_n_str, cls, img_id, name))

def create_top_n(top_n_list, top_n_str, n_img_per_class):
    if top_n_str not in os.listdir(DATA_PATH):
        os.mkdir(os.path.join(DATA_PATH, top_n_str))
        os.mkdir(os.path.join(DATA_PATH, top_n_str, 'train'))
        os.mkdir(os.path.join(DATA_PATH, top_n_str, 'test'))
        
    for cls in top_n_list:
        n_img = 0
        if cls not in os.listdir(os.path.join(DATA_PATH, top_n_str, 'train')):
            os.mkdir(os.path.join(DATA_PATH, top_n_str, 'train', cls))
        if cls not in os.listdir(os.path.join(DATA_PATH, top_n_str, 'test')):
            os.mkdir(os.path.join(DATA_PATH, top_n_str, 'test', cls))
        img_list = []
        for img_path in glob.iglob('%s/*/%s/*/*' % (DATA_PATH, cls)):
            img_cls = img_path.split('\\')[-3]
            # E.g. '<img_idx>_<img_id>_<img_type>.png'
            img_name = img_path.split('\\')[-1]
            img_var = img_name.split('_')[0]
            img_id = img_name.split('_')[1]

            if img_var + '_' + img_id not in img_list:
                img_list.append(img_var + '_' + img_id)
                n_img += 1
            if n_img >= n_img_per_class:
                continue

            if img_id not in os.listdir(os.path.join(DATA_PATH, top_n_str, 'train', cls)):
                os.mkdir(os.path.join(DATA_PATH, top_n_str, 'train', cls, img_id))
            
            shutil.copyfile(img_path, os.path.join(DATA_PATH, top_n_str, 'train', cls, img_id, img_name))


def count():
    cls_list = []
    with open(os.path.join('data', 'cls.txt'), 'r') as f:
        file = f.readlines()
        for cls in file:
            # remove '\n' from string
            cls = cls[:-1]
            cls_list.append(cls)

    img_id_dict = {}
    for img_path in glob.iglob('%s/*/*/*/*' % 'data'):
        if not img_path.endswith('RGB.png'):
            continue
        
        img_cls = img_path.split('\\')[-3]
        # E.g. '<img_idx>_<img_id>_<img_type>.png'
        img_name = img_path.split('\\')[-1]
        img_var = img_name.split('_')[0]
        img_id = img_name.split('_')[1]
        img_id_with_var = img_var + '_' + img_id
        img_id_dict[img_id_with_var] = img_cls

    cls = list(img_id_dict.values())
    cls_dict = {}
    for i in range(30):
        cls_dict[cls_list[i]] = cls.count(cls_list[i])

    ordered_cls_dict = {k: v for k, v in sorted(cls_dict.items(), key=lambda item: item[1])}
    print(ordered_cls_dict)


def create_cls_txt(cls_list, file_path):
    with open(file_path, 'w') as f:
        for cls in cls_list:
            f.write(cls)
            f.write('\n')
    f.close()


def find_grasp_file():
    """
    Missing grasp files:
    Chair 1_4f4ce917619e3d8e3227163156e32e3c_grasps.txt
    Chair 3_4f4ce917619e3d8e3227163156e32e3c_grasps.txt
    Chair 0_5d60590d192c52553a23b8cb1a985a11_grasps.txt
    Chair 1_5d60590d192c52553a23b8cb1a985a11_grasps.txt
    Chair 2_5d60590d192c52553a23b8cb1a985a11_grasps.txt
    Chair 3_5d60590d192c52553a23b8cb1a985a11_grasps.txt
    Chair 4_5d60590d192c52553a23b8cb1a985a11_grasps.txt
    deleted from top_5/train alr
    """
    ls = glob.glob('%s/*/*/*.txt' % 'data/item-grasp')
    file_ls = [path.split('\\')[-1] for path in ls]
    print(len(file_ls))
    input()
    total = 0
    no_match = 0
    for img_path in glob.iglob('%s/*/*/*/*' % 'data/top_5'):
        if not img_path.endswith('RGB.png'):
            continue
        
        img_cls = img_path.split('\\')[-3]
        # E.g. '<img_idx>_<img_id>_grasps.txt'
        img_name = img_path.split('\\')[-1]
        img_var = img_name.split('_')[0]
        img_id = img_name.split('_')[1]
        img_grasp_name = img_var + '_' + img_id + '_grasps.txt'

        total += 1
        if img_grasp_name not in file_ls:
            print(img_cls, img_grasp_name)
            no_match += 1

    return no_match, total


def get_grasp_files():
    train_ls = glob.glob('%s/*/*/*RGB.png' % 'data/top_5/train')
    train_file_ls = [path.split('\\')[-1] for path in train_ls]
    test_ls = glob.glob('%s/*/*/*RGB.png' % 'data/top_5/test')
    test_file_ls = [path.split('\\')[-1] for path in test_ls]
    for img_path in glob.iglob('%s/*/*/*.txt' % 'data/item-grasp'):
        img_cls = img_path.split('\\')[-3]
        # E.g. '<img_idx>_<img_id>_grasps.txt'
        img_name = img_path.split('\\')[-1]
        img_var = img_name.split('_')[0]
        img_id = img_name.split('_')[1]
        img_rgb_name = img_var + '_' + img_id + '_RGB.png'

        if img_rgb_name in test_file_ls:
            shutil.copyfile(img_path, 'data/top_5/test/%s/%s/%s' % (img_cls, img_id, img_name))
        elif img_rgb_name in train_file_ls:
            shutil.copyfile(img_path, 'data/top_5/train/%s/%s/%s' % (img_cls, img_id, img_name))
        else:
            print(img_cls, img_name)


def test_data_loader():
    """Identical dataloader process as written in data_loader.py."""
    data_loader = DataLoader(params.TRAIN_PATH, 2, params.TRAIN_VAL_SPLIT)
    for img, label, candidates in data_loader.load_grasp():
        target_bbox = grasps_to_bboxes(label)
        target_bboxes = grasps_to_bboxes(candidates)

        img_vis = np.array(img.cpu())
        img_r = np.clip((img_vis[:, 0, :, :] * 0.229 + 0.485) * 255, 0, 255)
        img_g = np.clip((img_vis[:, 1, :, :] * 0.224 + 0.456) * 255, 0, 255)
        img_d = img_vis[:, 2, :, :][0]
        
        img_bgr = np.concatenate((img_g, img_g, img_r), axis=0)
        img_bgr = np.moveaxis(img_bgr, 0, -1)
        img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
        
        draw_bbox(img_bgr, target_bbox[0], (255, 0, 0))
        for bbox in target_bboxes:
            # Choose some random bboxes to show:
            if random.randint(0, 5) == 0:
                draw_bbox(img_bgr, bbox, (0, 255, 0))

        cv2.imshow('img', img_bgr)
        cv2.waitKey(0)


def draw_bbox(img, bbox, color):
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


if __name__ == '__main__':
    # Get dataset statistics
    #count()
    # Separate and create train/test dataset folders for CLS training
    #create_cls_txt(top_10, '%s/cls_top_10.txt' % DATA_PATH)
    #create_top_n(top_10, 'top_10', 143)
    #create_test(top_10, 'top_10', 15)
    # Add grasping .txt files to the train/test dataset folders
    #no_match, total = find_grasp_file()
    #print(no_match, total)
    #get_grasp_files()
    # Visualize Grasp training data to make sure it all makes sens
    test_data_loader()