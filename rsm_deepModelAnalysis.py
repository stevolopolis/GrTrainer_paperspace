import random
import torch
import torch.nn as nn
from functools import partial
import collections

from rsm_generate_SSM import *

from parameters import Params
from data_loader_v2 import DataLoader
from inference.models import alexnet as models

params = Params()


def reorder_data(dataLoader, randomized_subset_n=None, seed=42):
    """
    Reorder dataLoader.img_id_list such that all classes are grouped together.
    Order of classes (with exact name spellings):
        - Chair
        - Lamp
        - figurines
        - plants
        - pen+pencil
    """
    chairs_id = []
    lamps_id = []
    figurines_id = []
    plants_id = []
    pens_id = []
    for id in dataLoader.img_id_list:
        cls = dataLoader.img_id_map[id]
        if cls == 'Chair':
            chairs_id.append(id)
        elif cls == 'Lamp':
            lamps_id.append(id)
        elif cls == 'figurines':
            figurines_id.append(id)
        elif cls == 'plants':
            plants_id.append(id)
        elif cls == 'pen+pencil':
            pens_id.append(id)

    new_list = chairs_id + lamps_id + figurines_id + plants_id + pens_id

    # Randomly subset <n> instances per class
    if randomized_subset_n is not None:
        new_chairs_id = []
        new_lamps_id = []
        new_figurines_id = []
        new_plants_id = []
        new_pens_id = []
        random.seed(seed)
        random_idx = random.sample(range(len(chairs_id)), randomized_subset_n)
        for idx in random_idx:
            new_chairs_id.append(chairs_id[idx])
            new_lamps_id.append(lamps_id[idx])
            new_figurines_id.append(figurines_id[idx])
            new_plants_id.append(plants_id[idx])
            new_pens_id.append(pens_id[idx])

        new_list = new_chairs_id + new_lamps_id + new_figurines_id + new_plants_id + new_pens_id

    dataLoader.img_id_list = new_list


def alexnetMap_register_hook(model, save_activation):
    """Register forward hook to all conv layers in alexnetMap model."""
    """for name, m in model.rgb_features.named_modules():
        if isinstance(m, nn.Conv2d):
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'rgbFeat_'+name))
    for name, m in model.d_features.named_modules():
        if isinstance(m, nn.Conv2d):
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'dFeat_'+name))"""
    for name, m in model.features.named_modules():
        if isinstance(m, nn.Conv2d):
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'feat_'+name))


def alexnet_register_hook(model, save_activation):
    """Register forward hook to all conv layers in alexnetMap model."""
    for name, m in model.features.named_modules():
        if isinstance(m, nn.Conv2d):
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'feat_'+name))


def alexnet_ductran_register_hook(model, save_activation):
    """Register forward hook to all conv layers in alexnetMap model."""
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, 'feat_'+name))


def foward_pass_dataset(model, dataLoader, model_type):
    """Forward pass through dataset to register activation maps."""
    if model_type == 'alexnetMap':
        # Forward pass through grasp dataset
        #for (img, _, _) in dataLoader.load_grasp():
        # Forward pass through cls dataset
        for (img, _, _) in dataLoader.load_cls():
            _ = model(img)
    elif model_type == 'alexnet' or model_type == 'alexnet_ductran':
        for (img, _, _) in dataLoader.load_cls(include_depth=False):
            _ = model(img)


def get_activations(model, dataLoader, model_type):
    """
    Code modified from 
    https://github.com/ShahabBakht/ventral-dorsal-model/blob/a959ac56650468894aa07a2e95eaf80250922791/RSM/deepModelsAnalysis.py
    """
    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)
    def save_activation(name, mod, inp, out):
        activations[name].append(out.cpu())

    # Save activation maps of all encoder conv layers in alexnetMap model
    if model_type == 'alexnetMap':
        alexnetMap_register_hook(model, save_activation)
    # Save activation maps of all encoder conv layers in alexnet (Imagenet)
    elif model_type == 'alexnet':
        alexnet_register_hook(model, save_activation)
    # Save activation maps of all encoder conv layers in alexnet (Cornell)
    elif model_type == 'alexnet_ductran':
        alexnet_ductran_register_hook(model, save_activation)

    # Forward pass through dataset to save activation maps
    foward_pass_dataset(model, dataLoader, model_type)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

    for name in activations.keys():
        activations[name] = activations[name].detach()
    
    return activations


def get_RSM(model, samples_per_cls=10, model_type='alexnetMap', seed=42):    
    print('Preparing dataset...')
    dataLoader = DataLoader(params.TRAIN_PATH_ALT, 1, params.TRAIN_VAL_SPLIT, seed=seed)
    reorder_data(dataLoader, randomized_subset_n=samples_per_cls, seed=seed)

    print('Obtaining activations...')
    activations = get_activations(model, dataLoader, model_type=model_type)
    activations_centered = center_activations(activations)
    
    print('Computing RSMs...')
    all_RSM = compute_similarity_matrices(activations_centered)
    
    return all_RSM, activations, model


"""if __name__ == '__main__':
    import os
    import seaborn as sb
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter
    
    SAMPLES_PER_CLS = 10

    # matplotlib tick parameters    
    formatter = FixedFormatter(['Chair', 'Lamp', 'figurines', 'plants', 'pen+pencil'])
    locator = FixedLocator(np.arange(SAMPLES_PER_CLS, SAMPLES_PER_CLS * 6, SAMPLES_PER_CLS))

    # Initialized pretrained alexnetMap model
    model = models.AlexnetMap_v2().to(params.DEVICE)
    model.load_state_dict(torch.load(params.MODEL_PATH))
    model.eval()

    rsm, activations, model = get_RSM(model, samples_per_cls=SAMPLES_PER_CLS)

    if params.MODEL_NAME not in os.listdir('vis/intralayer_corr'):
        os.makedirs('vis/intralayer_corr/%s' % params.MODEL_NAME)

    for layer in rsm:
        if layer in ('rgbFeat_0, dFeat_0', 'feat_15'): continue

        if 'layer_'+layer not in os.listdir('vis/intralayer_corr/%s' % params.MODEL_NAME):
            os.makedirs('vis/intralayer_corr/%s/layer_%s' % (params.MODEL_NAME, layer))

        #for kernel in rsm[layer]:
        heatmap = sb.heatmap(rsm[layer], cmap='viridis')
    
        heatmap.xaxis.set_major_formatter(formatter)
        heatmap.xaxis.set_major_locator(locator)
        heatmap.yaxis.set_major_formatter(formatter)
        heatmap.yaxis.set_major_locator(locator)
        plt.tick_params(axis='x', labelrotation=0)

        plt.title('Layer - %s' % (layer))
        plt.savefig('vis/intralayer_corr/%s/layer_%s' % (params.MODEL_NAME, layer))
        plt.close()"""
