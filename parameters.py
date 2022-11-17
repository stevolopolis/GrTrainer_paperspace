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
        #self.MODEL_NAME = 'alexnetMap_grasp_top5_v2.11.1'

        self.CLS_MODEL_NAME = 'alexnetMap_cls_top5_v2.3'
        self.GRASP_MODEL_NAME = 'alexnetMap_grasp_top5_v2.11.1'

        # device: cpu / gpu
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() \
                                      else torch.device('cpu')
        # Training params
        self.NUM_CLASS = 5
        self.NUM_CHANNEL = 4
        self.OUTPUT_SIZE = 224  # 128 was used for training grCLS
        self.IMG_SIZE = (self.NUM_CHANNEL, self.OUTPUT_SIZE, self.OUTPUT_SIZE) 
        self.EPOCHS = 150
        self.LR = 5e-4
        self.BATCH_SIZE = 256
        self.TRAIN_VAL_SPLIT = 0.0
        self.DISTILL_ALPHA = 1.0

        # Shapley params
        self.DATA_TRUNCATION = 0.5

        # Paths
        self.DATA_PATH = 'data'
        self.TRAIN_PATH = 'data/top_5_compressed/train'
        self.TRAIN_PATH_ALT = 'data/top_5_compressed_v2/train'
        self.TEST_PATH = 'data/top_5_compressed/test'
        self.TEST_PATH_ALT = 'data/top_5_compressed_v2/test'
        self.LABEL_FILE = 'cls_top_5.txt'

        self.MODEL_PATH = 'trained-models'
        self.CLS_MODEL_PATH = 'trained-models/%s/%s_epoch%s.pth' % (self.CLS_MODEL_NAME, self.CLS_MODEL_NAME, self.EPOCHS)
        self.GRASP_MODEL_PATH = 'trained-models/%s/%s_epoch%s.pth' % (self.GRASP_MODEL_NAME, self.GRASP_MODEL_NAME, 100)

        self.MODEL_LOG_PATH = 'trained-models/%s' % self.MODEL_NAME
        self.LOG_PATH = 'logs'
