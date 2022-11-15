import torch
import torch.nn as nn

from torchvision.models import alexnet

from shapley import layer_wise_tmc_shapley as tmc
from shapley import layer_wise_tmab_shapley as tmab
from evaluation import get_cls_acc, get_grasp_acc
from inference.models import alexnet as models
from parameters import Params

params = Params()
SEED = None
TRUNCATION = 0.5


def cls_criterion(model, device=params.DEVICE):
    return get_cls_acc(model, include_depth=True, seed=None, dataset=params.TEST_PATH, truncation=TRUNCATION, device=device)[0]


def grasp_criterion(model):
    return get_grasp_acc(model)[0]


def get_model():
    model = models.AlexnetMap_v2().to(params.DEVICE)
    model.load_state_dict(torch.load(params.CLS_MODEL_PATH))
    model.eval()

    return model


def get_alexnet_model():
    model = alexnet(pretrained=True).to(params.DEVICE)
    model.eval()

    return model

def get_cuda_list():
    cuda_list = []
    num_of_gpus = torch.cuda.device_count()
    for i in range(num_of_gpus):
        cuda_device = 'cuda:%s' % i
        cuda_list.append(cuda_device)

    return cuda_list


if __name__ == "__main__":
    model = get_model()
    shapley_score = tmab(model=model,
                        layer="rgb_features.0.weight",
                        criterion=cls_criterion,
                        k=10,
                        vt=50)