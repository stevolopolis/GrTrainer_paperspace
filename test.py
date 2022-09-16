"""
This file tests the model's performance on the testing dataset.

For CLS, this script returns the testing accuracy.
For Grasp, this script returns the testing accuracy and visualizes the
grasp prediction.
"""

import torch
import os

from parameters import Params
from inference.models.grconvnet_cls import GrCLS
from inference.models.alexnet import AlexNet, myAlexNet
from inference.models.alexnet import PretrainedAlexnet, PretrainedAlexnetv2, PretrainedAlexnetv3, PretrainedAlexnetv4
from evaluation import get_test_acc, get_grasp_acc, visualize_grasp

params = Params()

model_name = params.MODEL_NAME
weights_dir = params.MODEL_PATH
for epoch in range(97, 98):
    weights_path = os.path.join(weights_dir, model_name, model_name + '_epoch%s.pth' % epoch)

    # Gr-convnet CLS model
    #model = GrCLS(n_cls=params.NUM_CLASS).to(params.DEVICE)
    # Raw AlexNet CLS model
    #model = AlexNet(n_cls=params.NUM_CLASS).to(params.DEVICE)
    # AlexNet with 1st, 2nd layer pretrained on Imagenet
    #model = PretrainedAlexnet(n_cls=params.NUM_CLASS).to(params.DEVICE)
    # AlexNet with 1st, 2nd layer pretrained on Imagenet
    model = PretrainedAlexnetv4(n_cls=params.NUM_CLASS).to(params.DEVICE)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    # Get test acc for CLS model
    #accuracy, loss = get_test_acc(model)
    # Get test acc for Grasp model
    #accuracy, loss = get_grasp_acc(model)
    #print('Epoch: %s' % epoch, accuracy, loss)
    
    # Visualize grasp predictions one by one
    visualize_grasp(model)
