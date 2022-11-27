import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from parameters import Params

# Experiment parameters
TYPES = ['cls', 'grasp']
LAYERS = ['rgb_features.0', 'features.0', 'features.4', 'features.7', 'features.10']
R = 100.
DELTA = 0.2

DIR = 'shap'

params = Params()


def get_players(directory):
    ## Load the list of all players (filters) else save
    if 'players.txt' in os.listdir(directory):
        players = open(os.path.join(directory, 'players.txt')).read().split(',')
        players = np.array(players)
    else:
        raise Exception("Players do not exist!")

    return players


def instatiate_chosen_players(directory, players):
    if 'chosen_players.txt' not in os.listdir(directory):
        open(os.path.join(directory, 'chosen_players.txt'), 'w').write(','.join(
            np.arange(len(players)).astype(str)))


def get_results_list(directory):
    results = []
    for file in os.listdir(directory):
        if file.endswith('.h5'):
            results.append(os.path.join(directory, file))

    return results


def get_result(h5file):
    with h5py.File(h5file, 'r') as foo:
        return foo['mem_tmc'][:]


def get_shapley_top_k(model_name, layer, k):
    ## CB directory
    run_name = '%s_%s' % (model_name, layer)
    run_dir = os.path.join(DIR, run_name)

    players = get_players(run_dir)
    instatiate_chosen_players(run_dir, players)    
    results = get_results_list(run_dir)

    squares, sums, counts = [np.zeros(len(players)) for _ in range(3)]

    for result in results:
        mem_tmc = get_result(result)
        sums += np.sum((mem_tmc != -1) * mem_tmc, 0)
        squares += np.sum((mem_tmc != -1) * (mem_tmc ** 2), 0)
        counts += np.sum(mem_tmc != -1, 0)

    # No. of iterations for each neuron
    counts = np.clip(counts, 1e-12, None)
    # Expected shapley values of each neuron
    vals = sums / (counts + 1e-12)

    sorted_vals_idx = np.argsort(vals)[-k:]

    return sorted_vals_idx


def get_variance_std(sums, vals, squares, counts):
    variances = R * np.ones_like(vals)
    variances[counts > 1] = squares[counts > 1]
    variances[counts > 1] -= (sums[counts > 1] ** 2) / counts[counts > 1]
    variances[counts > 1] /= (counts[counts > 1] - 1)

    stds = variances ** (1/2)

    return variances, stds


def get_cb_bounds(vals, variances, counts):
    # Empriical berstein conf bounds
    cbs = R * np.ones_like(vals)
    cbs[counts > 1] = np.sqrt(2 * variances[counts > 1] * np.log(2 / DELTA) / counts[counts > 1]) +\
    7/3 * R * np.log(2 / DELTA) / (counts[counts > 1] - 1)

    return cbs


def plot_shapley_dist(players, results, model_type, layer):
    """
    (Bar chart)
    (Shapley-values against Kernel idx)
    Plot shapley value bar chart with variance and conf bounds
    """
    # Create saving directory
    if 'shapley_dist' not in os.listdir('vis/shap'):
        os.mkdir(os.path.join('vis/shap', 'shapley_dist'))

    squares, sums, counts = [np.zeros(len(players)) for _ in range(3)]

    for result in results:
        mem_tmc = get_result(result)
        sums += np.sum((mem_tmc != -1) * mem_tmc, 0)
        squares += np.sum((mem_tmc != -1) * (mem_tmc ** 2), 0)
        counts += np.sum(mem_tmc != -1, 0)

    # No. of iterations for each neuron
    counts = np.clip(counts, 1e-12, None)
    # Expected shapley values of each neuron
    vals = sums / (counts + 1e-12)
    # Variance of shapley values of each neuron
    variances, stds = get_variance_std(sums, vals, squares, counts)
    # Empirical berstein confidence bounds for each neuron
    cbs = get_cb_bounds(vals, variances, counts)

    sorted_vals_idx = np.argsort(vals)[-params.TOP_K:]
    top_k_vals = np.zeros((len(vals)))
    top_k_vals[sorted_vals_idx] = 1
    colors = np.where(top_k_vals == 1, 'coral', 'turquoise')

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(vals)), vals, yerr=cbs, align='center', ecolor='lightgrey', color=colors)
    ax.set_xlabel('Kernel Index')
    ax.set_ylabel('Shapley Scores')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

    plt.savefig('vis/shap/shapley_dist/shapley_dist_%s_%s.png' % (model_type, '_'.join(layer.split('.'))))


def plot_shapley_conf_trend(players, results, model_type, layer):
    """
    (Line graph)
    (Variance/confBound against iteration)
    Plot variance / confidence bounds line graph against No. of iters
    """
    # Create saving directory
    if 'shapley_confidence_bounds' not in os.listdir('vis/shap'):
        os.mkdir(os.path.join('vis/shap', 'shapley_confidence_bounds'))

    squares, sums, counts = [np.zeros(len(players)) for _ in range(3)]

    iter = 0
    cbs_history = np.zeros((1, len(players)))
    for result in results:
        mem_tmc = get_result(result)
        for mem_tmc_instance in mem_tmc:
            sums += (mem_tmc_instance != -1) * mem_tmc_instance
            squares += (mem_tmc_instance != -1) * (mem_tmc_instance ** 2)
            counts += mem_tmc_instance != -1
            iter += 1
    
            # No. of iterations for each neuron
            counts = np.clip(counts, 1e-12, None)
            # Expected shapley values of each neuron
            vals = sums / (counts + 1e-12)
            # Variance of shapley values of each neuron
            variances, stds = get_variance_std(sums, vals, squares, counts)
            # Empirical berstein confidence bounds for each neuron
            cbs = get_cb_bounds(vals, variances, counts)
            cbs_history = np.concatenate((cbs_history, np.expand_dims(cbs, 0)), axis=0)

    # Remove initial cbs_history array (zero array)
    cbs_history = cbs_history[1:, :]

    sorted_vals_idx = np.argsort(vals)[-params.TOP_K:]
    top_k_vals = np.zeros((len(vals)))
    top_k_vals[sorted_vals_idx] = 1

    fix, ax = plt.subplots()
    iter_axis = np.arange((cbs_history.shape[0]))
    for kernel_idx in sorted_vals_idx:
        ax.plot(iter_axis, cbs_history[:, kernel_idx], label='Kernel %s' % kernel_idx)

    ax.plot(iter_axis, np.repeat(np.min(cbs_history), len(iter_axis)), '-.', linewidth=.5, color='red')
    
    ax.legend()
    plt.ylim([0, 25])
    plt.yticks(list(plt.yticks()[0]) + [np.min(cbs_history)])
    
    plt.savefig('vis/shap/shapley_confidence_bounds/shapley_confidence_bounds_%s_%s.png' % (model_type, '_'.join(layer.split('.'))))


if __name__ == '__main__':
    if DIR not in os.listdir('vis'):
        os.mkdir(os.path.join('vis', DIR))

    for model_type in TYPES:
        if model_type == 'cls':
            model_name = params.CLS_MODEL_NAME
        elif model_type == 'grasp':
            model_name = params.GRASP_MODEL_NAME

        for layer in LAYERS:
            ## CB directory
            run_name = '%s_%s' % (model_name, layer)
            run_dir = os.path.join(DIR, run_name)

            players = get_players(run_dir)
            instatiate_chosen_players(run_dir, players)    
            results = get_results_list(run_dir)

            plot_shapley_dist(players, results, model_type, layer)
            plot_shapley_conf_trend(players, results, model_type, layer)