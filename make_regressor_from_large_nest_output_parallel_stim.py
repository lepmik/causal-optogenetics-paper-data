import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import sys
import multiprocessing as mp

n_cores = 8

path = pathlib.Path(sys.argv[1])

with open(str(path / 'params.json'), 'r') as f:
    params = json.load(f)

    t_stop = params['status']['time']

conn = pd.read_feather(path / 'connections_0.feather')
sender_ids = conn.query('weight >= 0').source.sort_values().unique()
n_neurons = len(sender_ids)
min_sender_ids = min(sender_ids)

stim_data = np.load(path / 'stimulation_data_0.npz', allow_pickle=True)['data'][()]
stim_times = np.array(stim_data['times'])

X = np.lib.format.open_memmap(path / 'X.npy', dtype=int, shape=(len(stim_times), n_neurons), mode='w+')
Y = np.lib.format.open_memmap(path / 'Y.npy', dtype=int, shape=(len(stim_times), n_neurons), mode='w+')
Z = np.lib.format.open_memmap(path / 'Z.npy', dtype=int, shape=(len(stim_times), n_neurons), mode='w+')
Yb = np.lib.format.open_memmap(path / 'Yb.npy', dtype=int, shape=(len(stim_times), n_neurons), mode='w+')


def update_regressors(stims, spikes, z1=-2, z2=0, x1=1, x2=3, y1=3, y2=7, yb1=-4, yb2=0):
    senders = spikes.sender
    times = spikes.times

    for i, t in enumerate(stims):
        search = [
            t + x1, t + x2,
            t + y1, t + y2,
            t + z1, t + z2,
            t + yb1, t + yb2
        ]
        idx = np.searchsorted(times, search, side='right')
        X[i, senders[idx[0]: idx[1]] - min_sender_ids] = 1
        Y[i, senders[idx[2]: idx[3]] - min_sender_ids] = 1
        Z[i, senders[idx[4]: idx[5]] - min_sender_ids] = 1
        Yb[i, senders[idx[6]: idx[7]] - min_sender_ids] = 1


paths = [f for f in path.iterdir() if f.suffix == '.gdf' and f.stem.startswith('ex')]
for f in tqdm(paths):
    df = pd.read_csv(f, sep='\t', header=None)
    df = df.rename(columns={0:'sender', 1:'times'}).drop(columns=2)
    stim_times_split = np.array_split(stim_times, n_cores)
    processes = []
    for stims in stim_times_split:
        process = mp.Process(target=update_regressors, args=(stims, df))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


# save data to file
del X, Y, Z, Yb
