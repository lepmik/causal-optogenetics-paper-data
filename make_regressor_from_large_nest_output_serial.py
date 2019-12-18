import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import sys


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


def update_regressors(spikes, data, a, b):
    senders = spikes.sender
    times = spikes.times

    for i, t in enumerate(tqdm(stim_times)):
        idx = np.searchsorted(times, [t + a, t + b], side='right')
        data[i, senders[idx[0]: idx[1]] - min_sender_ids] = 1


def make_regressor(name, a, b):
    data = np.zeros((len(stim_times), n_neurons))

    paths = [f for f in path.iterdir() if f.suffix == '.gdf' and f.stem.startswith('ex')]
    for f in tqdm(paths):
        df = pd.read_csv(f, sep='\t', header=None)
        df = df.rename(columns={0:'sender', 1:'times'}).drop(columns=2)
        update_regressors(df, data, a, b)

    np.save(path / '{}.npy'.format(name), data)

make_regressor('X', 1, 3)
make_regressor('Y', 3, 7)
make_regressor('Z', -2, 0)
make_regressor('Yb', -4, 0)
