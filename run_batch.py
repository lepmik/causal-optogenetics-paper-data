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


# def make_regressors(
#     stim_data, sender_ids, spikes,
#     z1=-2, z2=0, x1=1, x2=3, y1=3, y2=7, yb1=-4, yb2=0):
#
#     n_neurons = len(sender_ids)
#     min_sender_ids = min(sender_ids)
#     stim_times = np.array(stim_data['times'])
#
#     X = np.zeros((len(stim_times), n_neurons))
#     Y = np.zeros((len(stim_times), n_neurons))
#     Z = np.zeros((len(stim_times), n_neurons))
#     Yb = np.zeros((len(stim_times), n_neurons))
#
#     senders = spikes.senders
#     times = spikes.times
#
#     for i, t in enumerate(stim_times):
#         search = [
#             t + x1, t + x2,
#             t + y1, t + y2,
#             t + z1, t + z2,
#             t + yb1, t + yb2
#         ]
#         idx = np.searchsorted(times, search, side='right')
#         X[i, senders[idx[0]: idx[1]] - 1] = 1
#         Y[i, senders[idx[2]: idx[3]] - 1] = 1
#         Z[i, senders[idx[4]: idx[5]] - 1] = 1
#         Yb[i, senders[idx[6]: idx[7]] - 1] = 1
#
#     return X, Y, Z, Yb


def compute_response(row, spikes, a, b):
    senders = spikes.senders
    times = spikes.times

    idx = np.searchsorted(times, [a, b], side='right')
    row.loc[senders[idx[0]: idx[1]]] = 1
    return row


def compute_conditional_means(row, X, Y, Z, Yb, min_sender_ids):

    z = Z.loc[:, int(row.source)]
    x = X.loc[:, int(row.source)]
    y = Y.loc[:, int(row.target)]
    yb = Yb.loc[:, int(row.target)]

    y_ref = y[z==1].mean()
    yb_ref = yb[z==1].mean()

    y_base = y[x==0].mean()
    yb_base = yb[x==0].mean()

    y_respons = y[x==1].mean()
    yb_respons = yb[x==1].mean()

    return pd.Series({
        'y_ref': y_ref,
        'yb_ref': yb_ref,
        'y_base': y_base,
        'yb_base': yb_base,
        'y_respons': y_respons,
        'yb_respons': yb_respons
    })


if __name__ == '__main__':
    if len(sys.argv) == 5:
        data_path, param_module, connfile, n_runs = sys.argv[1:]
    else:
        raise IOError('Usage: "python simulator.py data_path parameters [connfile]')

    jobname = param_module.replace('.py', '')
    currdir = op.dirname(op.abspath(__file__))
    f, p, d = imp.find_module(jobname, [currdir])
    parameters = imp.load_module(jobname, f, p, d).parameters
    np.random.seed(parameters['msd'])
    labels = 'y_ref, yb_ref, y_base, yb_base, y_respons, yb_respons'
    stim_amps = None
    for n in tqdm(range(int(n_runs))):
        seed = np.random.randint(1000, 9999)
        sim = Simulator(
            parameters,
            data_path=data_path,
            jobname=jobname,
            verbose=False,
            save_to_file=False,
            msd=seed
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
        assert min(sender_ids_ex) == 1

        X = pd.DataFrame(
            np.zeros((len(stim_times), n_neurons)),
            index=stim_data['times'], columns=sender_ids_ex
        )
        Y, Z, Yb = X.copy(), X.copy(), X.copy()

        X = X.parallel_apply(
            compute_response, spikes=spikes, a=1, b=3, axis=1, result_type='expand')
        Y = Y.parallel_apply(
            compute_response, spikes=spikes, a=3, b=7, axis=1, result_type='expand')
        Z = Z.parallel_apply(
            compute_response, spikes=spikes, a=-2, b=0, axis=1, result_type='expand')
        Yb = Yb.parallel_apply(
            compute_response, spikes=spikes, a=-4, b=0, axis=1, result_type='expand')

        stim_nodes = list(stim_amps.keys())
        results = sim.connections.query('weight >= 0')
        min_sender_ids = min(results.source)
        results = results.query(
            'target not in @stim_nodes and target in @sender_ids_ex')
        results.loc[:,'source_stimulated'] = results.source.isin(
            stim_nodes)
        results.loc[:,'target_stimulated'] = results.target.isin(
            stim_nodes)

        results['stim_amp_source'] = results.parallel_apply(
            lambda x: stim_amps.get(x.source, 0), axis=1)

        results = results.join(results.parallel_apply(
            compute_conditional_means, axis=1,
            X=X, Y=Y, Z=Z, Yb=Yb,
            min_sender_ids=min_sender_ids,
            result_type='expand'
        ))

        results.reset_index(drop=True).to_feather(
            sim.data_path / 'conditional_means_{}.feather'.format(n))
