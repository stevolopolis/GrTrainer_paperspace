import numpy as np
import torch
import torch.nn as nn
import torchvision
import random
import copy
from tqdm import tqdm

class TensorID:
    """
    ID wrapper around a tensor
    """
    def __init__(self, tensor: torch.Tensor, id_: int):
        self.tensor = tensor
        self.id = id_

    def __str__(self):
        return f"<id:{self.id}, tensor:{self.tensor}>"

    def __repr__(self):
        return self.__str__()


def layer_silence(model: nn.Module, layer: str, weights: list) -> nn.Module:
    """
    Silence the impact of weights within the <layer> of the <model>, except those in <weights>.
    Silenced weights are replaced by the mean weights of other functional weights
    """
    # Update new weights onto new model
    model_new = copy.deepcopy(model)
    with torch.no_grad():
        for name, W in model_new.named_parameters():
            if name == layer:
                # Construct new weights
                n = W.data.shape[0]  # number of filters
                w_mean = torch.zeros_like(W.data[0])
                for w in weights:
                    w_mean = w_mean + w.tensor
                w_mean = w_mean / n
                w_new = w_mean.unsqueeze(0).repeat(n, *[1 for _ in w_mean.shape]).squeeze()
                #print(W.data.shape, w_new.shape)
                for w in weights:
                    w_new[w.id] = w.tensor
                # assign weights
                #print(W.data.shape, w_new.shape)
                W.data = w_new
    return model_new


def layer_wise_tmc_shapley(model: nn.Module, layer: str, criterion, iterations: int, vt: float, verbose=True)\
        -> torch.Tensor:
    """
    Truncated Monte Carlo method
    compute the shapley score for each conv filter of a layer
    Params:
        model: torch model
        layer: the layer name
        criterion: performance metric, returns a scalar
        k: k number of the top filters to look for
        vt: early truncation performance vt
        epsilon: failure tolerance. Default: 0
        delta: failure probability. Default: 0.05

    Return:
        shapley: torch tensor where each element corresponds to shapley score
    """
    # Initializations
    weight = None
    t = 0
    for name, W in model.named_parameters():
        if name == layer:
            weight = torch.split(W, 1, dim=0)
    assert weight is not None, f"Layer {layer} not found"
    weight = [TensorID(w, i) for i, w in enumerate(weight)]  # assign id to weight
    n = len(weight)
    shapley = torch.zeros(n)

    # TMC Iterations
    for _ in range(iterations):
        t += 1
        if verbose:
            print(f"Iteration {1}/{iterations}:")
        random.shuffle(weight)  # shuffle weights in place
        v = [criterion(model)]  # performance partition
        for j in tqdm(range(1, n+1)):
            if v[j-1] < vt:
                v.append(v[j-1])
            else:
                partial_model = layer_silence(model, layer, weight[j:])
                v.append(criterion(partial_model))
                print(v)
            # calculate mean shapley
            shapley[weight[j-1].id] = (v[j-1] - v[j] + shapley[weight[j-1].id] * (t-1))/t
    return shapley


def layer_wise_tmab_shapley(model: nn.Module, layer: str, criterion, k: float, vt: float, epsilon: float, delta: float)\
        -> torch.Tensor:
    """
    Truncated Multi Armed Bandit Method
    compute the shapley score for each conv filter of a layer
    Params:
        model: torch model
        layer: the layer name
        criterion: performance metric, returns a scalar
        k: k number of the top filters to look for
        vt: early truncation performance vt
        epsilon: failure tolerance. Default: 0
        delta: failure probability. Default: 0.05

    Return:
        shapley: torch tensor where each element corresponds to shapley score
    """
    # Initializations
    weight = None
    t = 0
    for name, W in model.named_parameters():
        if name == layer:
            weight = torch.split(W, 1, dim=0)
    assert weight is not None, f"Layer {layer} not found"
    weight = [TensorID(w, i) for i, w in enumerate(weight)]  # assign id to weight
    n = len(weight)
    shapley = torch.zeros(n)
    shapley_var = torch.zeros(n)
    que = range(n)

    # TMAB Iterations
    while que:
        t += 1
        random.shuffle(weight)  # shuffle weights in place
        v = [criterion(model)]  # performance partition
        for j in range(1, n+1):
            if j in que:
                if v[j-1] < vt:
                    v.append(v[j-1])
                else:
                    partial_model = layer_silence(model, layer, weight[j:])
                    v.append(criterion(partial_model))
                # calculate mean
                # calculate variance
                # empirical Bernstein confidence bound
        # reassign que
        pass

    return shapley
