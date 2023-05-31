import os
import datetime
import torch

from parameters import Params

params = Params()


class AddGaussianNoise(torch.nn.Module):
    """Gaussian noise augmentation fn used in DataLoader class."""
    def __init__(self, mean=0., std=1., device=params.DEVICE):
        self.std = std
        self.mean = mean
        self.device = device
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def log_writer(network_name, epoch, step, loss, train=True):
    """Writes training losses into a log.txt file."""
    filename = '%s_log.txt' % network_name
    # If log file does not exist, create a new .txt file and write training info
    if filename not in os.listdir(params.LOG_PATH):
        with open(os.path.join(params.LOG_PATH, filename), 'w') as f:
            initial_message = '%s Training Log at %s' % (network_name, datetime.datetime.now())
            f.write(initial_message + '\n')
    # If log file exists, write loss info to new line
    else:
        with open(os.path.join(params.LOG_PATH, filename), 'a') as f:
            if train:
                log_message = 'train-epoch-step: %s-%s -- Loss: %s' % (epoch, step, loss)
            else:
                log_message = 'val-epoch-step: %s-%s -- Loss: %s' % (epoch, step, loss)
            f.write('\n')
            f.write(log_message)


def epoch_logger(network_name, epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc):
    """Writes epoch loss and accuracy statistics to log fie."""
    filename = '%s_log.txt' % network_name
    with open(os.path.join(params.LOG_PATH, filename), 'a') as f:
        train_loss_mean = sum(train_loss) / len(train_loss)
        val_loss_mean = sum(val_loss) / len(val_loss)
        log_message = 'Epoch: %s -- Train Loss: %s -- Train Acc: %s -- Val Loss: %s -- Val Acc: %s\n \
                        Test Loss: %s -- Test Acc: %s' \
                        % (epoch,
                           round(train_loss_mean, 4), train_acc,
                           round(val_loss_mean, 4), val_acc,
                           test_loss, test_acc)
        f.write('\n')
        f.write(log_message)


def get_correct_preds(output, label):
    """Returns number of correct predictions out of number of instances."""
    _, prediction = torch.max(output.data, 1)
    _, ground_truth = torch.max(label.data, 1)
    total = label.size(0)
    correct = (prediction == ground_truth).sum().item()
    return correct, total


def get_correct_cls_preds_from_map(output, label):
    """
    Return number of correct predictions out of number of instances.
    Predictions are in the form of maps -- [batch_size, output_size, output_size, 6]
    """
    conf_map = torch.unsqueeze(output[:, 5, :, :], 1)
    conf_map = torch.cat((conf_map, conf_map, conf_map, conf_map, conf_map), 1)
    pred_map = output[:, :5, :, :]
    pred_val_map = conf_map * pred_map
    pred_val_sum = torch.sum(pred_val_map, (2, 3))
    pred = torch.max(pred_val_sum, 1)[1]  # indices
    correct = (pred == label).sum().item()
    """conf_map = torch.reshape(output[:, 5, :, :], (params.BATCH_SIZE, -1))
    max_conf_idx = torch.max(conf_map, 0)[1]
    pred_map = torch.reshape(output[:, :5, :, :], (params.BATCH_SIZE, 5, output.shape[2] * output.shape[3]))
    max_pred = pred_map
    pred = torch.max(pred_val_sum, 1)[1]  # indices
    correct = (pred == label).sum().item()"""

    return correct, len(pred)

def get_acc(correct, total):
    """Returns accuracy given number of correct predictions and total 
    number of predictions."""
    return round(100 * correct / total, 2)


def tensor_concat(tensor1, tensor2):
    """
    Concatenates two grasp candidate tensors.
    For example, this function allows tensor1 and tensor2 with the following shapes 
    to be concatenated:
        - tensor1.shape == (10, 10, 6)
        - tensor2.shape == (1, 6, 6)
    which becomes,
        - tensor_concat(tensor1, tensor2).shape == (11, 10, 6)
    Or
        - tensor1.shape == (3, 11, 4)
        - tensor2.shape == (5, 5, 4)
    which becomes,
        - tensor_concat(tensor1, tensor2).shape == (8, 11, 4)
    This function allows tensors of one unequal dimension to concatenate by 
    broadcasting the smaller tensor by copying it's last element multiple times
    to match the dimensions of the bigger tensor.
    """
    n_dim1 = tensor1.shape[1]
    n_dim2 = tensor2.shape[1]
    n_dim_diff = abs(n_dim1 - n_dim2)
    if n_dim1 < n_dim2:
        broadcasting_elem = torch.unsqueeze(tensor1[:, -1, :], dim=1).repeat(1, n_dim_diff, 1)
        broadcasted_tensor = torch.cat((tensor1, broadcasting_elem), dim=1)
        return torch.cat((broadcasted_tensor, tensor2), dim=0)
    elif n_dim1 > n_dim2:
        broadcasting_elem = torch.unsqueeze(tensor2[:, -1, :], dim=1).repeat(1, n_dim_diff, 1)
        broadcasted_tensor = torch.cat((tensor2, broadcasting_elem), dim=1)
        return torch.cat((tensor1, broadcasted_tensor), dim=0)
    else:
        return torch.cat((tensor1, tensor2), dim=0)

# Scratch code for visualizing image after augmentations.
# Feel free to copy other augmentation codes from DataLoader class
# To visualize the effects of each augmentation.
"""from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import torch.nn as nn
aug = AddGaussianNoise(0, .025)
img = Image.open('data/top_5/train/plants/2a7d62b731a04f5fa54b9afa882a89ed/0_2a7d62b731a04f5fa54b9afa882a89ed_RGB.png')
img = np.array(img)
img = img / 255
cv2.imshow('file', img)
cv2.waitKey(0)
img = torch.tensor(img)
img = torch.moveaxis(img, -1, 0)
random_transforms = transforms.RandomApply(nn.ModuleList([AddGaussianNoise(0, .1)]), p=0.25)
trans = transforms.Compose([
    transforms.RandomResizedCrop((params.OUTPUT_SIZE, params.OUTPUT_SIZE), scale=(.75, .85), ratio=(1, 1))
])
img = trans(img)
img = torch.moveaxis(img, 0, -1)
img = img.numpy()
print(img.shape)
img = np.clip(img, 0, 1)
print(np.max(img), np.min(img))
cv2.imshow('file', img)
cv2.waitKey(0)"""
