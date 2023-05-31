import os
import sys
import numpy as np
import h5py

from parameters import Params

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


# Experiment parameters
SAVE_FREQ = 1
MODEL_NAME = params.CLS_MODEL_NAME
LAYER = 'features.10'
METRIC = 'accuracy'
TRUNCATION_ACC = 50.
DEVICE = sys.argv[1]
DIR = 'shap'

R = 100.
DELTA = 0.2
TOP_K = 20

## CB directory
run_name = '%s_%s' % (MODEL_NAME, LAYER)
run_dir = os.path.join(DIR, run_name)

while True:
    ## Start
    players = get_players(run_dir)
    instatiate_chosen_players(run_dir, players)    
    results = get_results_list(run_dir)

    squares, sums, counts = [np.zeros(len(players)) for _ in range(3)]
    max_vals, min_vals = -np.ones(len(players)), np.ones(len(players))

    for result in results:
        mem_tmc = get_result(result)
        sums += np.sum((mem_tmc != -1) * mem_tmc, 0)
        squares += np.sum((mem_tmc != -1) * (mem_tmc ** 2), 0)
        counts += np.sum(mem_tmc != -1, 0)
        
    counts = np.clip(counts, 1e-12, None)
    vals = sums / (counts + 1e-12)
    variances = R * np.ones_like(vals)
    variances[counts > 1] = squares[counts > 1]
    variances[counts > 1] -= (sums[counts > 1] ** 2) / counts[counts > 1]
    variances[counts > 1] /= (counts[counts > 1] - 1)

    if np.max(counts) == 0:
        os.remove(os.path.join(run_dir, result))

    # Empriical berstein conf bounds
    cbs = R * np.ones_like(vals)
    cbs[counts > 1] = np.sqrt(2 * variances[counts > 1] * np.log(2 / DELTA) / counts[counts > 1]) +\
    7/3 * R * np.log(2 / DELTA) / (counts[counts > 1] - 1)

    thresh = (vals)[np.argsort(vals)[-TOP_K - 1]]
    chosen_players = np.where(
        ((vals - cbs) < thresh) * ((vals + cbs) > thresh))[0]

    print(run_dir, np.mean(counts), len(chosen_players), np.mean(cbs))
    input()
    open(os.path.join(run_dir, 'chosen_players.txt'), 'w').write(
        ','.join(chosen_players.astype(str)))
    open(os.path.join(run_dir, 'variances.txt'), 'w').write(
        ','.join(variances.astype(str)))
    open(os.path.join(run_dir, 'vals.txt'), 'w').write(
        ','.join(vals.astype(str)))
    open(os.path.join(run_dir, 'counts.txt'), 'w').write(
        ','.join(counts.astype(str)))

    if len(chosen_players) == 1:
        break