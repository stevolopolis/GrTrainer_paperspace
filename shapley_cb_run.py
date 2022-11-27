import copy
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import inference.models.alexnet as models
from evaluation import get_cls_acc, get_grasp_acc
from parameters import Params

params = Params()


def remove_players(model: nn.Module, layer: str, removed_idx: list) -> nn.Module:
    """
    Silence the impact of weights within the <layer> of the <model>, except those in <weights>.
    Silenced weights are replaced by the mean weights of other functional weights
    """
    # Update new weights onto new model
    with torch.no_grad():
        for name, W in model.named_parameters():
            if name == layer+'.weight' or name == layer+'.bias':           
                # Calculate mean non-removed weight
                keeping_idx = [i for i in range(W.data.shape[0]) if i not in removed_idx]
                w_mean = torch.mean(W.data[keeping_idx], dim=0)
                W.data[removed_idx] = w_mean

    return model
        

def one_iteration(
    model, 
    layer,
    players,
    c,
    truncation,
    device='cuda',
    chosen_players=None,
    metric='accuracy', 
    task='cls'):
    '''One iteration of Neuron-Shapley algoirhtm.'''
    if chosen_players is None:
        chosen_players = np.arange(len(c.keys()))
    
    # Original performance of the model with all players present.
    init_val = get_acc(model, task=task, device=device)

    # A random ordering of players
    idxs = np.random.permutation(len(c.keys()))
    # -1 default value for players that have already converged
    marginals = -np.ones(len(c.keys()))
    marginals[chosen_players] = 0.

    truncation_counter = 0
    old_val = init_val

    pbar = tqdm(total=len(idxs))
    removing_players = []
    for idx in idxs:
        if idx in chosen_players:
            removing_players.append(players[c[idx]])
            partial_model = remove_players(model, layer, removing_players)     

            new_val = get_acc(partial_model, task=task, device=device)
            marginals[c[idx]] = old_val - new_val
            old_val = new_val
            
            if metric == 'accuracy' and new_val <= truncation:
                truncation_counter += 1
            else:
                truncation_counter = 0
            if truncation_counter > 5:
                break
        else:
            removing_players.append(players[c[idx]])
            partial_model = remove_players(model, layer, removing_players)      
            new_val = get_acc(partial_model, task=task, device=device)
            old_val = new_val

        pbar.set_postfix({'Acc': new_val})
        pbar.update(1)

    return idxs.reshape((1, -1)), marginals.reshape((1, -1))


def get_acc(model, task='cls', device='cuda:0'):
    if task == 'cls':
        return get_cls_acc(model, include_depth=True, seed=None, dataset=params.TEST_PATH, truncation=params.DATA_TRUNCATION, device=device)[0]
    elif task == 'grasp':
        return get_grasp_acc(model, include_depth=True, seed=None, dataset=params.TEST_PATH, truncation=params.DATA_TRUNCATION, device=device)[0]
    else:
        raise ValueError('Invalid task!')


class TensorID:
    """
    ID wrapper around a tensor
    """
    def __init__(self, layer: str, tensor: torch.Tensor, id_: int):
        self.layer = layer
        self.tensor = tensor
        self.id = id_

    def __str__(self):
        return f"<id:{self.id}, tensor:{self.tensor}>"

    def __repr__(self):
        return self.__str__()


def get_model(model_path, device=params.DEVICE):
    model = models.AlexnetMap_v3().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def get_weights(model, layer):
    for name, W in model.named_parameters():
        if name == layer+'.weight':
            weight = torch.split(W, 1, dim=0)
        elif name == layer+'.bias':
            bias = torch.split(W, 1, dim=0)

    assert weight is not None, f"Layer {layer} not found"
    assert bias is not None, f"Layer {layer} not found"

    weights = [TensorID(layer+'.weight', None, i) for i, _ in enumerate(weight)]
    biases = [TensorID(layer+'.bias', None, i) for i, _ in enumerate(bias)]

    return weights, biases


def get_players(directory, weights):
    ## Load the list of all players (filters) else save
    players = []
    for weight in weights:
        players.append(str(weight.id))
    np.sort(players)
    open(os.path.join(directory, 'players.txt'), 'w').write(','.join(players))

    return np.array(players).astype(np.int8)


def instantiate_tmab_logs(players, log_dir):
    ## Create placeholder for results in save ASAP to prevent having the 
    ## same expriment_number with other parallel cb_run.py scripts
    mem_tmc = np.zeros((0, len(players)))
    idxs_tmc = np.zeros((0, len(players))).astype(int)

    with h5py.File(log_dir, 'w') as foo:
        foo.create_dataset("mem_tmc", data=mem_tmc, compression='gzip')
        foo.create_dataset("idxs_tmc", data=idxs_tmc, compression='gzip')

    return mem_tmc, idxs_tmc


# Experiment parameters
SAVE_FREQ = 1
TASK = 'grasp'
LAYER = 'rgb_features.0'
METRIC = 'accuracy'
TRUNCATION_ACC = 50.
DEVICE = sys.argv[1]
DIR = 'shap'
if TASK == 'cls':
    MODEL_NAME = params.CLS_MODEL_NAME
    MODEL_PATH = params.CLS_MODEL_PATH
elif TASK == 'grasp':
    MODEL_NAME = params.GRASP_MODEL_NAME
    MODEL_PATH = params.GRASP_MODEL_PATH

PARALLEL_INSTANCE = sys.argv[2]

## CB directory
run_name = '%s_%s' % (MODEL_NAME, LAYER)
run_dir = os.path.join(DIR, run_name)
log_dir = os.path.join(run_dir, '%s_%s.h5' % (run_name, PARALLEL_INSTANCE))


if DIR not in os.listdir('.'):
    os.mkdir(DIR)
if run_name not in os.listdir(DIR):
    os.mkdir(run_dir)

## Load Model and get weights
model = get_model(MODEL_PATH, DEVICE)
weights, bias = get_weights(model, LAYER)
## Instantiate or load player list
players = get_players(run_dir, weights)
# Instantiate tmab logs
mem_tmc, idxs_tmc = instantiate_tmab_logs(players, log_dir)

## Running CB-Shapley
#c = {i: np.array([i]) for i in range(len(players))}
c = {i: i for i in range(len(players))}

counter = 0
while True:
    ## Load the list of players (filters) that are determined to be not confident enough
    ## by the cb_aggregate.py running in parallel to this script
    if 'chosen_players.txt' in os.listdir(run_dir):
        chosen_players = open(os.path.join(run_dir, 'chosen_players.txt')).read()
        chosen_players = np.array(chosen_players.split(',')).astype(int)
        if len(chosen_players) == 1:
            break
    else:
        chosen_players = None
        
    idxs, vals =  one_iteration(
        copy.deepcopy(model),
        LAYER, 
        players,
        c,
        TRUNCATION_ACC,
        device=DEVICE,
        chosen_players=chosen_players,
        metric=METRIC,
        task=TASK
    )

    mem_tmc = np.concatenate([mem_tmc, vals])
    idxs_tmc = np.concatenate([idxs_tmc, idxs])
    
    ## Save results every SAVE_FREQ iterations
    if counter % SAVE_FREQ == SAVE_FREQ - 1:
        with h5py.File(log_dir, 'w') as foo:
            foo.create_dataset("mem_tmc", data=mem_tmc, compression='gzip')
            foo.create_dataset("idxs_tmc", data=idxs_tmc, compression='gzip')
            
    counter += 1
