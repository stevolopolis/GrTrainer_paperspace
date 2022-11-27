import captum.attr as C

import torch
import os
import cv2

from parameters import Params
import inference.models.alexnet as models
from data_loader_v2 import DataLoader
from captum_model_wrapper import ModelWrapper
from evaluation import denormalize_img

params = Params()

model_name = params.MODEL_NAME
weights_dir = params.MODEL_PATH
epoch = 150
weights_path = os.path.join(weights_dir, model_name, model_name + '_epoch%s.pth' % epoch)

# AlexNet with 1st, 2nd layer pretrained on Imagenet
model = models.AlexnetMap_v2().to(params.DEVICE)
model.load_state_dict(torch.load(weights_path))
model.eval()

data_loader = DataLoader('data/top_5_compressed_v2/train', 1, params.TRAIN_VAL_SPLIT, return_mask=True)
for i, (img, cls_map, label, img_mask) in enumerate(data_loader.load_cls()):
    captumModel = ModelWrapper(model, img_mask, label)
    ig = C.Lime(captumModel)

    print('Running IG...')
    attributions = ig.attribute(img, target=label, show_progress=True, n_samples=2000, perturbations_per_eval=1)

    print(attributions, attributions.shape)
    lime_input = img * attributions
    vis_input_rgb = lime_input[:, :3, :, :]
    vis_input_rgb = denormalize_img(vis_input_rgb)

    cv2.imshow('vis', vis_input_rgb)
    cv2.waitKey(0)

