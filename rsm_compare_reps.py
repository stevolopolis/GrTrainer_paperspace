import os

import torch
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import alexnet, resnet18

from tqdm import tqdm

from rsm_deepModelAnalysis import *

from parameters import Params
from data_loader_v2 import DataLoader
from inference.models import alexnet as models
from inference.models.alexnet_ductranvan import Net
from inference.models.grconvnet3 import GenerativeResnet
from shapley_analysis import get_shapley_top_k

params = Params()

SAMPLES_PER_CLS = 20


def get_comparison_models():
    cls_model = models.AlexnetMap_v3().to(params.DEVICE)
    cls_model.load_state_dict(torch.load(params.CLS_MODEL_PATH))
    cls_model.eval()

    grasp_model = models.AlexnetMap_v3().to(params.DEVICE)
    grasp_model.load_state_dict(torch.load(params.GRASP_MODEL_PATH))
    grasp_model.eval()

    #cls_baseline_model = alexnet(pretrained=True).to(params.DEVICE)
    cls_baseline_model = resnet18(pretrained=True).to(params.DEVICE)
    cls_baseline_model.eval()

    # For Redmond's alexnet
    """grasp_baseline_model = Net({"input_shape": (3,256,256),
                              "initial_filters": 16, 
                              "num_outputs": 5}).to(params.DEVICE)
    grasp_baseline_model.load_state_dict(torch.load('trained-models/weights.pt'))
    grasp_baseline_model.eval()"""

    # For Kumra's Gr-Convnet
    grasp_baseline_model = torch.load('trained-models/epoch_19_iou_0.98').to(params.DEVICE)
    grasp_baseline_model.eval()

    return cls_model, grasp_model, cls_baseline_model, grasp_baseline_model


def get_shapley(model_name):
    print('Obtaining shapley values for %s ...' % model_name)
    shapley = {}
    for layer in params.LAYERS:
        shapley[layer] = get_shapley_top_k(model_name, layer, params.TOP_K)

    return shapley


def format_rsm_list(rsm_list):
    new_rsm = {}
    for rsm in rsm_list:
        for key in rsm.keys():
            if key not in new_rsm.keys():
                new_rsm[key] = [rsm[key]]
            else:
                new_rsm[key].append(rsm[key])

    return new_rsm


def get_rsm_corr(rsm1, rsm2):   
    cls_r = np.empty((len(rsm1), len(rsm2)))

    pbar = tqdm(total=len(rsm1))
    for i, cls_kernel in enumerate(rsm1):
        for j, cls_baseline_kernel in enumerate(rsm2):
            cls_r[i , j] = compute_ssm(rsm1[cls_kernel], rsm2[cls_baseline_kernel])
        
        pbar.update(1)

    return cls_r


def combine_rsm(rsm1, rsm2, rsm3, rsm4):
    combined_rsm1 = np.concatenate((rsm1, rsm2), axis=0)
    combined_rsm2 = np.concatenate((rsm3, rsm4), axis=0)
    combined_rsm = np.concatenate((combined_rsm1, combined_rsm2), axis=1)

    return combined_rsm


def visualize_rsm(rsm, n1, n2, save_path, normalized=True):
    if normalized:
        heatmap = sb.heatmap(rsm, cmap='viridis', square=True, annot=False)
    else:
        heatmap = sb.heatmap(rsm, vmin=0.0, vmax=1.0, cmap='viridis', square=True, annot=False)

    heatmap.hlines([n1], *heatmap.get_xlim(), color='red', linewidth=1)
    heatmap.vlines([n2], *heatmap.get_ylim(), color='red', linewidth=1)
    
    plt.xlabel('Layers of reference models (Resnet-18 / GR-ConvNet)')
    plt.xticks([2.5, 5+4], ['CLS (ImageNet)', 'GRASP (Cornell)'])
    plt.ylabel('Layers of AlexNetMap (ours)')
    plt.yticks([n1/2, 3*n1/2], ['CLS', 'GRASP'])
    plt.tick_params(axis='x')
    plt.tick_params(axis='y', rotation=0)
    plt.savefig(save_path)
    plt.close()


def compare_reps(samples_per_cls=20, n_samples=10, model_name='alexnetMap'):
    print('Computing RSMs...')
    for i in range(n_samples):
        seed = random.randint(0, 1000)
        print("Seed: ", seed)

        cls_model, grasp_model, cls_baseline_model, grasp_baseline_model = get_comparison_models()
        cls_model_shapley = get_shapley(params.CLS_MODEL_NAME)
        grasp_model_shapley = get_shapley(params.GRASP_MODEL_NAME)

        cls_rsm, cls_activations, _ = get_RSM(cls_model, selected_kernels=cls_model_shapley, samples_per_cls=samples_per_cls, model_type='alexnetMap', seed=seed)
        grasp_rsm, grasp_activations, _ = get_RSM(grasp_model, selected_kernels=grasp_model_shapley, samples_per_cls=samples_per_cls, model_type='alexnetMap', seed=seed)
        cls_baseline_rsm, cls_baseline_activations, _ = get_RSM(cls_baseline_model, samples_per_cls=samples_per_cls, model_type='resnet', seed=seed)
        grasp_baseline_rsm, grasp_baseline_activations, _ = get_RSM(grasp_baseline_model, samples_per_cls=samples_per_cls, model_type='grconvnet_kumra', seed=seed)

        cls_cls_corr = get_rsm_corr(cls_rsm, cls_baseline_rsm)
        grasp_cls_corr = get_rsm_corr(grasp_rsm, cls_baseline_rsm)
        cls_grasp_corr = get_rsm_corr(cls_rsm, grasp_baseline_rsm)
        grasp_grasp_corr = get_rsm_corr(grasp_rsm, grasp_baseline_rsm)

        combined_corr = combine_rsm(cls_cls_corr, grasp_cls_corr, cls_grasp_corr, grasp_grasp_corr)

        # Create saving directories and save fig
        if 'inter_model_corr' not in os.listdir('vis'):
            os.makedirs('vis/inter_model_corr')
        if model_name not in os.listdir('vis/inter_model_corr'):
            os.makedirs('vis/inter_model_corr/%s' % model_name)
        if 'nInstance_'+str(samples_per_cls) not in os.listdir('vis/inter_model_corr/%s' % model_name):
            os.makedirs('vis/inter_model_corr/%s/nInstance_%s' % (model_name, samples_per_cls))

        save_path = 'vis/inter_model_corr/%s/nInstance_%s' % (model_name, samples_per_cls)
        save_name = '%s_nInstance_%s_sample_%s' % (model_name, samples_per_cls, i)
        visualize_rsm(combined_corr, 5, 5, '%s/%s' % (save_path, save_name), normalized=True)
        visualize_rsm(combined_corr, 5, 5, '%s/%s_raw' % (save_path, save_name), normalized=False)


def compare_top_k(samples_per_cls=20):
    cls_model_shapley = get_shapley(params.CLS_MODEL_NAME)
    grasp_model_shapley = get_shapley(params.GRASP_MODEL_NAME)

    cls_r = {layer: {0: []} for layer in params.LAYERS}
    grasp_r = {layer: {0: []} for layer in params.LAYERS}

    for i in range(params.TOP_K):
        cls_model, grasp_model, _, _ = get_comparison_models()

        selected_cls_model_shapley = {key: cls_model_shapley[key][i] for key in cls_model_shapley}
        selected_grasp_model_shapley = {key: grasp_model_shapley[key][i] for key in grasp_model_shapley}
        
        cls_rsm, cls_activations, _ = get_RSM(cls_model, selected_kernels=selected_cls_model_shapley, samples_per_cls=samples_per_cls, model_type='alexnetMap', seed=i)
        grasp_rsm, grasp_activations, _ = get_RSM(grasp_model, selected_kernels=selected_grasp_model_shapley, samples_per_cls=samples_per_cls, model_type='alexnetMap', seed=i)

        for layer in params.LAYERS:
            cls_r[layer].update({i: cls_rsm[layer]})
            grasp_r[layer].update({i: grasp_rsm[layer]})

    for layer in params.LAYERS:
        cls_cls_corr = get_rsm_corr(cls_r[layer], cls_r[layer])
        grasp_cls_corr = get_rsm_corr(grasp_r[layer], cls_r[layer])
        cls_grasp_corr = get_rsm_corr(cls_r[layer], grasp_r[layer])
        grasp_grasp_corr = get_rsm_corr(grasp_r[layer], grasp_r[layer])
        
        combined_corr = combine_rsm(cls_cls_corr, grasp_cls_corr, cls_grasp_corr, grasp_grasp_corr)

        save_path = 'vis/inter_model_corr/alexnetMap_shapley/top_k_per_layer_rsm/max_%s_%s_top_k_rsm' % (layer.split('.')[0], layer.split('.')[1])
        visualize_rsm(combined_corr, 5, 5, save_path)


if __name__ == '__main__':
    #compare_top_k(samples_per_cls=50)
    compare_reps(samples_per_cls=25, n_samples=3, model_name='alexnetMap_resnetImgn_grconvCorn')