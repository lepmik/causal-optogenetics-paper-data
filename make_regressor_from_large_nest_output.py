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

conn = pd.read_feather('params_1_psc_connections.feather')
sender_ids = conn.query('weight >= 0').source.sort_values().unique()

stim_data = np.load(path / 'stimulation_data_0.npz', allow_pickle=True)['data'][()]
stim_times = np.array(stim_data['times'])

N = len(sender_ids)

def make_regressors(stim_times, spikes, z1=-2, z2=0, x1=1, x2=3, y1=3, y2=7, yb1=-4, yb2=0):
    senders = spikes.sender
    times = spikes.times
    min_sender_ids = min(sender_ids)
    z = np.zeros((len(stim_times), N))
    x = np.zeros((len(stim_times), N))
    y = np.zeros((len(stim_times), N))
    yb = np.zeros((len(stim_times), N))
    for i, t in enumerate(tqdm(stim_times)):
        idx = np.searchsorted(times, [t + z1, t + z2], side='right')
        id_spiked_before = senders[idx[0]: idx[1]] - min_sender_ids

        z[i, id_spiked_before] = 1

        idx = np.searchsorted(times, [t + x1, t + x2], side='right')
        id_spiked = senders[idx[0]: idx[1]] - min_sender_ids
        x[i, id_spiked] = 1

        idx = np.searchsorted(times, [t + y1, t + y2], side='right')
        id_spiked = senders[idx[0]: idx[1]] - min_sender_ids
        y[i, id_spiked] = 1

        idx = np.searchsorted(times, [t + yb1, t + yb2], side='right')
        id_spiked_before = senders[idx[0]: idx[1]] - min_sender_ids

        yb[i, id_spiked_before] = 1

    return x, y, z, yb

X = np.zeros((len(stim_times), N))
Y = np.zeros((len(stim_times), N))
Z = np.zeros((len(stim_times), N))
Yb = np.zeros((len(stim_times), N))

for f in path.iterdir():
    if not f.suffix == '.gdf':
        continue
    if not f.stem.startswith('ex'):
        continue
    df = pd.read_csv(f, sep='\t', header=None).rename(columns={0:'sender',1:'times'}).drop(columns=2)

    x, y, z, yb = make_regressors(stim_times, df)
    X += x
    Y += y
    Z += z
    Yb += yb

np.save(path / 'X.npy', X)
np.save(path / 'Y.npy', Y)
np.save(path / 'Z.npy', Z)
np.save(path / 'Yb.npy', Yb)
