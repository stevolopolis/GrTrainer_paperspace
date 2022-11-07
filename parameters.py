"""This file contains the Param class which contains all the 
parameters needed for data loading, data processing, model training,
and model processing.

This class is imported in all codes required for training. Any
changes to parameters could be done collectively here.
"""

import torch

class Params:
    """
    Parameters for training models.
    """
    def __init__(self):
        # Model name -- '<type>_<raw/pretrained>_<input>_<version>'
        # CLS name format
        self.MODEL_NAME = 'alexnetMap_cls_top5_v2.3'
        #self.MODEL_NAME = 'grConvMap_cls_top5_v1'
        # Grasp name format
        #self.MODEL_NAME = 'alexnetMap_grasp_top5_v2.1'

        # device: cpu / gpu
        self.DEVICE = torch.device('cpu') if torch.cuda.is_available() \
                                      else torch.device('cpu')
        # Training params
        self.NUM_CLASS = 5
        self.NUM_CHANNEL = 4
        self.OUTPUT_SIZE = 224  # 128 was used for training grCLS
        self.IMG_SIZE = (self.NUM_CHANNEL, self.OUTPUT_SIZE, self.OUTPUT_SIZE) 
        self.EPOCHS = 150
        self.LR = 5e-4
        self.BATCH_SIZE = 64
        self.TRAIN_VAL_SPLIT = 0.2

        # Paths
        self.DATA_PATH = 'data'
        self.TRAIN_PATH = 'data/top_5_compressed/train'
        self.TEST_PATH = 'data/top_5_compressed/test'
        self.LABEL_FILE = 'cls_top_5.txt'
        self.MODEL_PATH = 'trained-models'
        self.MODEL_LOG_PATH = 'trained-models/%s' % self.MODEL_NAME
        self.LOG_PATH = 'logs'
