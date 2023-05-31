import random
import torch
import collections
import cv2 

from rsm_generate_SSM import *

from parameters import Params
from data_loader_v2 import DataLoader
from inference.models import alexnet as models
from model_utils import *
from evaluation import denormalize_grasp, map2singlegrasp, denormalize_img

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


def foward_pass_dataset(model, dataLoader, model_type, get_output_summary=False):
    """Forward pass through dataset to register activation maps."""
    output_summary = None
    if model_type == 'alexnetMap' and get_output_summary:
        # Forward pass through grasp dataset
        for (img, _, _) in dataLoader.load_grasp():
            output = model(img)
            output = torch.moveaxis(output, 1, -1)
            # Denoramlize grasps
            denormalize_grasp(output)
            # Convert grasp map into single grasp prediction
            output_grasp = map2singlegrasp(output)
            if output_summary is None:
                output_summary = output_grasp
            else:
                output_summary = torch.cat((output_summary, output_grasp), dim=0)
        
        return output_summary
    elif model_type == 'alexnetMap' and not get_output_summary:
        # Forward pass through cls dataset
        for (img, _, _) in dataLoader.load_cls():
            _ = model(img)
    elif model_type in ('alexnet', 'alexnet_ductran', 'resnet'):
        for (img, _, _) in dataLoader.load_cls(include_depth=False):
            _ = model(img)
    elif model_type == 'grconvnet_kumra':
        for (img, _, _) in dataLoader.load_cls():
            _ = model(img)


def get_activations(model, dataLoader, model_type, selected_kernels=None, sorted_idx=None):
    """
    Code modified from 
    https://github.com/ShahabBakht/ventral-dorsal-model/blob/a959ac56650468894aa07a2e95eaf80250922791/RSM/deepModelsAnalysis.py
    """
    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)
    def save_activation(name, mod, inp, out):
        activations[name].append(out.cpu().detach())

    # Save activation maps of all encoder conv layers in alexnetMap model
    if model_type == 'alexnetMap':
        alexnetMap_register_hook(model, save_activation)
    # Save activation maps of all encoder conv layers in alexnet (Imagenet)
    elif model_type == 'alexnet':
        alexnet_register_hook(model, save_activation)
    # Save activation maps of all encoder conv layers in alexnet (Cornell)
    elif model_type == 'alexnet_ductran':
        alexnet_ductran_register_hook(model, save_activation)
    # Save activation maps of all encoder conv layers in grconvnet (Jacquard)
    elif model_type == 'grconvnet_kumra':
        grconvnet_kumra_register_hook(model, save_activation)
    # Save activation maps of all encoder conv layers in resnet (Imagenet)
    elif model_type == 'resnet':
        resnet_register_hook(model, save_activation)
    
    # Forward pass through dataset to save activation maps
    foward_pass_dataset(model, dataLoader, model_type)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

    # if there's not sorted indices, then use default order
    if sorted_idx is None:
        sorted_idx = [i for i in range(len(list(activations.values())[0]))]

    for name in activations.keys():
        if selected_kernels is not None:
            layer_activations = activations[name][sorted_idx]
            selected_layer_kernels = selected_kernels[name]
            layer_activations = layer_activations[:, selected_layer_kernels, :, :]
            activations[name] = get_activation_summary(layer_activations, method='simple-max')
        else:
            layer_activations = activations[name][sorted_idx]
            activations[name] = get_activation_summary(layer_activations, method='simple-max')
    
    return activations


def get_activation_summary(layer_activations, method='simple-mean'):
    if method == 'simple-mean':
        flattend_activations = torch.reshape(layer_activations, (layer_activations.shape[0], layer_activations.shape[1], -1))
        mean = torch.mean(flattend_activations, dim=2)
        return mean
    elif method == 'simple-max':
        flattend_activations = torch.reshape(layer_activations, (layer_activations.shape[0], layer_activations.shape[1], -1))
        max = torch.max(flattend_activations, dim=2)[0]
        return max
    elif method == 'mean-std':
        flattend_activations = torch.reshape(layer_activations, (layer_activations.shape[0], layer_activations.shape[1], -1))
        mean = torch.mean(flattend_activations, dim=2)
        std = torch.std(flattend_activations, dim=2)
        return torch.cat((torch.unsqueeze(mean, dim=2), torch.unsqueeze(std, dim=2)), dim=2)
    elif method == 'kernel-max':
        max = torch.nn.functional.adaptive_max_pool2d(layer_activations, (2, 2))
        return max
    elif method == 'aggregate-kernel-max':
        flattend_activations = torch.reshape(layer_activations, (layer_activations.shape[0], layer_activations.shape[1], -1))
        aggregate_max = torch.mean(flattend_activations, dim=2).unsqueeze(2)
        for output_size in [2, 8, 16]:
            max = torch.nn.functional.adaptive_max_pool2d(layer_activations, (output_size, output_size))
            flattened_max = torch.reshape(max, (max.shape[0], max.shape[1], -1))
            aggregate_max = torch.cat((aggregate_max, flattened_max), dim=2)
        return aggregate_max
    elif method == 'mean-max':
        flattend_activations = torch.reshape(layer_activations, (layer_activations.shape[0], layer_activations.shape[1], -1))
        mean = torch.mean(flattend_activations, dim=2)
        max = torch.max(flattend_activations, dim=2)[0]
        return torch.cat((torch.unsqueeze(mean, dim=2), torch.unsqueeze(max, dim=2)), dim=2)


def get_RSM(model, selected_kernels=None, samples_per_cls=10, model_type='alexnetMap', seed=42, sorted_idx=None, device=params.DEVICE):    
    print('Preparing dataset...')
    dataLoader = DataLoader(params.TRAIN_PATH, 1, 0.0, device=device, seed=seed)
    reorder_data(dataLoader, randomized_subset_n=samples_per_cls, seed=seed)

    print('Obtaining activations...')
    activations = get_activations(model, dataLoader, model_type, selected_kernels=selected_kernels, sorted_idx=sorted_idx)
    activations_centered = center_activations(activations)
    
    print('Computing RSMs...')
    all_RSM = compute_similarity_matrices(activations_centered)
    
    return all_RSM, activations, model


if __name__ == '__main__':
    import os
    import seaborn as sb
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from torchvision.models import alexnet

    from rsm_compare_reps import get_shapley
    
    SAMPLES_PER_CLS = 50
    model_path = params.GRASP_MODEL_PATH
    model_name = params.GRASP_MODEL_NAME
    #model_name = 'alexnet'
    model_name = 'grconvnetCorn'

    # matplotlib tick parameters    
    formatter = FixedFormatter(['Chair', 'Lamp', 'figurines', 'plants', 'pen+pencil'])
    #formatter = FixedFormatter([round(1*(1/6), 2), round(2*(1/6), 2), round(3*(1/6), 2), round(4*(1/6), 2), round(5*(1/6), 2)])
    locator = FixedLocator(np.arange(SAMPLES_PER_CLS, SAMPLES_PER_CLS * 6, SAMPLES_PER_CLS))


    if model_name not in os.listdir('vis/intralayer_corr'):
        os.makedirs('vis/intralayer_corr/%s' % model_name)

    # Gr-Convnet
    model = torch.load('trained-models/epoch_19_iou_0.98').to(params.DEVICE)
    model.eval()

    rsm, activations, model = get_RSM(model, selected_kernels=None, samples_per_cls=SAMPLES_PER_CLS, model_type='grconvnet_kumra', seed=42, sorted_idx=None)
    for layer in rsm:
        #for kernel in rsm[layer]:
        heatmap = sb.heatmap(rsm[layer], cmap='viridis')
    
        heatmap.xaxis.set_major_formatter(formatter)
        heatmap.xaxis.set_major_locator(locator)
        heatmap.yaxis.set_major_formatter(formatter)
        heatmap.yaxis.set_major_locator(locator)
        plt.tick_params(axis='x', labelrotation=0)

        plt.title('Layer - %s' % (layer))
        plt.savefig('vis/intralayer_corr/%s/max_layer_%s_%s.png' % (model_name, layer.split('_')[0], layer.split('_')[1]))
        plt.close()

    """
    # Imagenet alexnet
    model = alexnet(pretrained=True).to(params.DEVICE)
    model.eval()

    rsm, activations, model = get_RSM(model, selected_kernels=None, samples_per_cls=SAMPLES_PER_CLS, model_type='alexnet', seed=42, sorted_idx=None)
    for layer in rsm:
        #for kernel in rsm[layer]:
        heatmap = sb.heatmap(rsm[layer], cmap='viridis')
    
        heatmap.xaxis.set_major_formatter(formatter)
        heatmap.xaxis.set_major_locator(locator)
        heatmap.yaxis.set_major_formatter(formatter)
        heatmap.yaxis.set_major_locator(locator)
        plt.tick_params(axis='x', labelrotation=0)

        plt.title('Layer - %s' % (layer))
        plt.savefig('vis/intralayer_corr/%s/aggregate_kernelmax_layer_%s_%s.png' % (model_name, layer.split('_')[0], layer.split('_')[1]))
        plt.close()
    """

    model_shapley = get_shapley(model_name)

    """# Initialized pretrained alexnetMap model
    model = models.AlexnetMap_v3().to(params.DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    rsm, activations, model = get_RSM(model, selected_kernels=model_shapley, samples_per_cls=SAMPLES_PER_CLS, model_type='alexnetMap', seed=42)

    if model_name not in os.listdir('vis/intralayer_corr'):
        os.makedirs('vis/intralayer_corr/%s' % model_name)

    for layer in rsm:
        #for kernel in rsm[layer]:
        heatmap = sb.heatmap(rsm[layer], cmap='viridis')
    
        heatmap.xaxis.set_major_formatter(formatter)
        heatmap.xaxis.set_major_locator(locator)
        heatmap.yaxis.set_major_formatter(formatter)
        heatmap.yaxis.set_major_locator(locator)
        plt.tick_params(axis='x', labelrotation=0)

        plt.title('Layer - %s' % (layer))
        plt.savefig('vis/intralayer_corr/%s/top_k_rsm/max_layer_%s.png' % (model_name, "_".join(layer.split("."))))
        plt.close()"""

    
    """# Sort data by grasp parameters (e.g. angle, width, height)
    sorting_model = models.AlexnetMap_v3().to(params.DEVICE)
    sorting_model.load_state_dict(torch.load(params.GRASP_MODEL_PATH))
    sorting_model.eval()
    dataLoader = DataLoader(params.TRAIN_PATH, 1, 0.0, seed=42)
    reorder_data(dataLoader, randomized_subset_n=SAMPLES_PER_CLS, seed=42)
    sorted_idx = None
    #output_summary = foward_pass_dataset(model, dataLoader, 'alexnetMap', get_output_summary=True)
    #sorted_output, sorted_idx = torch.sort(output_summary[:, 3] + output_summary[:, 4])

    for i in range(params.TOP_K):
        # Initialized pretrained alexnetMap model
        model = models.AlexnetMap_v3().to(params.DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        selected_model_shapley = {key: model_shapley[key][i] for key in model_shapley}
        rsm, activations, model = get_RSM(model, selected_kernels=selected_model_shapley, samples_per_cls=SAMPLES_PER_CLS, model_type='alexnetMap', seed=42, sorted_idx=sorted_idx)

        if model_name not in os.listdir('vis/intralayer_corr'):
            os.makedirs('vis/intralayer_corr/%s' % model_name)

        for layer in rsm:
            #for kernel in rsm[layer]:
            heatmap = sb.heatmap(rsm[layer], cmap='viridis')
        
            heatmap.xaxis.set_major_formatter(formatter)
            heatmap.xaxis.set_major_locator(locator)
            heatmap.yaxis.set_major_formatter(formatter)
            heatmap.yaxis.set_major_locator(locator)
            plt.tick_params(axis='x', labelrotation=0)

            plt.title('Layer - %s; Kernel - %s; Rank - %s' % (layer, selected_model_shapley[layer], i))
            plt.savefig('vis/intralayer_corr/%s/k_individual_rsm/max_layer_%s_kernel_%s.png' % (model_name, '_'.join(layer.split('.')), selected_model_shapley[layer]))
            plt.close()
"""