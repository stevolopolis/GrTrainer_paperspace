import os

import torch
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import alexnet

from tqdm import tqdm

from rsm_deepModelAnalysis import *

from parameters import Params
from data_loader_v2 import DataLoader
from inference.models import alexnet as models
from inference.models.alexnet_ductranvan import Net

params = Params()

SAMPLES_PER_CLS = 20


def get_comparison_models():
    cls_model = models.AlexnetMap_v2().to(params.DEVICE)
    cls_model.load_state_dict(torch.load(params.CLS_MODEL_PATH))
    cls_model.eval()

    grasp_model = models.AlexnetMap_v2().to(params.DEVICE)
    grasp_model.load_state_dict(torch.load(params.GRASP_MODEL_PATH))
    grasp_model.eval()

    cls_baseline_model = alexnet(pretrained=True).to(params.DEVICE)
    cls_baseline_model.eval()

    grasp_baseline_model = Net({"input_shape": (3,256,256),
                              "initial_filters": 16, 
                              "num_outputs": 5}).to(params.DEVICE)
    grasp_baseline_model.load_state_dict(torch.load('trained-models/weights.pt'))
    grasp_baseline_model.eval()

    return cls_model, grasp_model, cls_baseline_model, grasp_baseline_model


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


def visualize_rsm(rsm, model_name, samples_per_cls, id):
    heatmap = sb.heatmap(rsm, cmap='viridis', vmin=0, vmax=1, square=True)

    if 'inter_model_corr' not in os.listdir('vis'):
        os.makedirs('vis/inter_model_corr')

    if model_name not in os.listdir('vis/inter_model_corr'):
        os.makedirs('vis/inter_model_corr/%s' % model_name)
    if 'nInstance_'+str(samples_per_cls) not in os.listdir('vis/inter_model_corr/%s' % model_name):
        os.makedirs('vis/inter_model_corr/%s/nInstance_%s' % (model_name, samples_per_cls))

    heatmap.hlines([4], *heatmap.get_xlim(), color='red', linewidth=1)
    heatmap.vlines([4], *heatmap.get_ylim(), color='red', linewidth=1)
    
    plt.title('RDM for Alexnet(imagenet / cornell)')
    plt.xlabel('Layers of Alexnet (Imagenet / Cornell)')
    plt.xticks([2, 6.5], ['CLS', 'GRASP'])
    plt.ylabel('Layers of Alexnet (Imagenet / Cornell)')
    plt.yticks([2, 6.5], ['CLS', 'GRASP'])
    plt.tick_params(axis='x')
    plt.tick_params(axis='y', rotation=0)
    plt.savefig('vis/inter_model_corr/refernece-rdm-imagenet-cornell')#%s/nInstance_%s/%s_nInstance_%s_sample_%s' % (model_name, samples_per_cls, model_name, samples_per_cls, id))
    plt.close()


def compare_reps(samples_per_cls=20, n_samples=10):
    print('Computing RSMs...')
    for i in range(n_samples):
        cls_model, grasp_model, cls_baseline_model, grasp_baseline_model = get_comparison_models()

        #cls_rsm, cls_activations, _ = get_RSM(cls_model, samples_per_cls=samples_per_cls, model_type='alexnetMap', seed=i)
        #grasp_rsm, grasp_activations, _ = get_RSM(grasp_model, samples_per_cls=samples_per_cls, model_type='alexnetMap', seed=i)
        cls_baseline_rsm, cls_baseline_activations, _ = get_RSM(cls_baseline_model, samples_per_cls=samples_per_cls, model_type='alexnet', seed=i)
        grasp_baseline_rsm, grasp_baseline_activations, _ = get_RSM(grasp_baseline_model, samples_per_cls=samples_per_cls, model_type='alexnet_ductran', seed=i)

        # Temporary useless layers / kernels to ignore
        #cls_rsm.pop('feat_15')
        #grasp_rsm.pop('feat_15')
        cls_baseline_rsm.pop('feat_0')

        cls_cls_corr = get_rsm_corr(cls_baseline_rsm, cls_baseline_rsm)
        grasp_cls_corr = get_rsm_corr(grasp_baseline_rsm, cls_baseline_rsm)
        cls_grasp_corr = get_rsm_corr(cls_baseline_rsm, grasp_baseline_rsm)
        grasp_grasp_corr = get_rsm_corr(grasp_baseline_rsm, grasp_baseline_rsm)

        combined_corr = combine_rsm(cls_cls_corr, grasp_cls_corr, cls_grasp_corr, grasp_grasp_corr)
        visualize_rsm(combined_corr, 'alexnetMap', samples_per_cls, i)


if __name__ == '__main__':
    for n in [50]:
        compare_reps(samples_per_cls=n, n_samples=5)