import numpy as np
import pandas as pd
from simulator_psc import Simulator
import sys
import imp
import os.path as op
import nest
from tqdm import tqdm
from pandarallel import pandarallel
import warnings
warnings.filterwarnings("ignore")

# Initialization
pandarallel.initialize(progress_bar=True)


def make_regressors(stim_data, sender_ids, spikes, z1=-2, z2=0, x1=1, x2=3, y1=3, y2=7, yb1=-4, yb2=0):

    n_neurons = len(sender_ids)
    min_sender_ids = min(sender_ids)
    stim_times = np.array(stim_data['times'])

    X = np.zeros((len(stim_times), n_neurons))
    Y = np.zeros((len(stim_times), n_neurons))
    Z = np.zeros((len(stim_times), n_neurons))
    Yb = np.zeros((len(stim_times), n_neurons))

    senders = spikes.senders
    times = spikes.times

    for i, t in enumerate(stim_times):
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

    return X, Y, Z, Yb


def compute_conditional_means(row, X, Y, Z, Yb, min_sender_ids):

    z = Z[:, int(row.source) - min_sender_ids]
    x = X[:, int(row.source) - min_sender_ids]
    y = Y[:, int(row.target) - min_sender_ids]
    yb = Yb[:, int(row.target) - min_sender_ids]

    y_ref = y[z==1].mean()
    yb_ref = yb[z==1].mean()

    y_base = y[x==0].mean()
    yb_base = yb[x==0].mean()

    y_respons = y[x==1].mean()
    yb_respons = yb[x==1].mean()

    return y_ref, yb_ref, y_base, yb_base, y_respons, yb_respons


if __name__ == '__main__':
    if len(sys.argv) == 5:
        data_path, param_module, connfile, n_runs = sys.argv[1:]
    else:
        raise IOError('Usage: "python simulator.py data_path parameters [connfile]')

    jobname = param_module.replace('.py', '')
    currdir = op.dirname(op.abspath(__file__))
    f, p, d = imp.find_module(jobname, [currdir])
    parameters = imp.load_module(jobname, f, p, d).parameters

    labels = 'y_ref, yb_ref, y_base, yb_base, y_respons, yb_respons'
    stim_amps = None
    for n in tqdm(range(int(n_runs))):
        seed = np.random.randint(1000, 9999)
        sim = Simulator(
            parameters,
            data_path=data_path,
            jobname=jobname,
            verbose=False,
            save_to_file=False
        )
        sim.simulate(
            state=False,
            progress_bar=True,
            connfile=connfile,
            stim_amps=stim_amps
        )
        if stim_amps is None:
            stim_amps = sim.stim_amps
        spikes = sim.get_spikes('ex')
        sim.dump_params_to_json(suffix=n)

        sender_ids_ex = sim.connections.query('weight >= 0').source.sort_values().unique()

        X, Y, Z, Yb = make_regressors(
            stim_data=sim.stim_data,
            sender_ids=sender_ids_ex,
            spikes=spikes
        )

        stim_nodes = list(stim_amps.keys())
        connections = sim.connections.query('weight >= 0')
        min_sender_ids = min(connections.source)
        connections = connections.query(
            'target not in @stim_nodes and target in @sender_ids_ex')
        connections.loc[:,'source_stimulated'] = connections.source.isin(
            stim_nodes)
        connections.loc[:,'target_stimulated'] = connections.target.isin(
            stim_nodes)

        connections['stim_amp_source'] = connections.parallel_apply(
            lambda x: stim_amps.get(x.source, 0), axis=1)

        connections.parallel_apply(
            compute_conditional_means, axis=1, X=X, Y=Y, Z=Z, Yb=Yb,
            min_sender_ids=min_sender_ids
        )

        connections.reset_index(drop=True).to_feather(
            sim.data_path / 'conditional_means_{}.feather'.format(n))
