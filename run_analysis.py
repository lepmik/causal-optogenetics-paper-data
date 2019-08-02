import sys
from causal_optoconnectics.core import causal_connectivity, hit_rate
from causal_optoconnectics.cch import transfer_probability
import pandas as pd
import numpy as np
import os.path as op
from tools import csv_append_dict
from tools_analysis import (corrcoef, coef_var)
import quantities as pq
from tqdm import tqdm

data_path = 'results'
trials = [30000]
# trials = [5000, 10000, 15000, 20000, 25000, 30000]
N = 100

iv_params = {
    'x_mu': 2e-3,
    'x_sigma': 1e-3,
    'y_mu': 7e-3,
    'y_sigma': 3e-3,
    'n_bases': 20,
    'bin_size': 1e-3,
    'offset': 1e-3
}

trans_prob_params = {
    'y_mu': 4e-3,
    'y_sigma': 3e-3,
    'bin_size': 1e-3,
    'limit': 15e-3,
    'hollow_fraction': .6,
    'width': 60
}

binsize_corr = 5e-3
source_pop = 'ex'
target_pop = 'ex'

if len(sys.argv) != 2:
    raise IOError('Usage: "python analyse.py parameters"')
param_module = sys.argv[1]
jobname = param_module.replace('.py', '')

print('Loading data')
data = np.load(op.join(data_path, jobname + '.npz'), allow_pickle=True)['data'][()]
print('Organizing data')
conn = data['connections']

spiketrains = {}
for pop in ['ex', 'in']:
    senders = data['spiketrains'][pop]['senders']
    times = data['spiketrains'][pop]['times'] / 1000 # ms -> s
    spk = {sender: {'pop': pop, 'times': times[sender==senders]}
           for sender in np.unique(senders)}
    spiketrains.update(spk)

stim_amps = data['stim_amps'][source_pop]
data['epoch']['times'] /= 1000 # ms -> s

sources = np.array([
    n for n in data['stim_nodes'][source_pop]
    if n in data['spiketrains'][source_pop]['senders']]) # if had spikes
targets = np.array([
    n for n in data['nodes'][target_pop]
    if (n not in sources and n in data['spiketrains'][target_pop]['senders'])])

idx_s = np.random.randint(0, len(sources), int(N / 2))
idx_t = np.random.randint(0, len(targets), int(N / 2))

sources, targets = sources[idx_s], targets[idx_t]

assert len(sources) + len(targets) == N
print('Starting main loop')
pbar = tqdm(total=int(N / 2)**2 * len(trials))
for N_trials in trials:
    stim_times = data['epoch']['times'][:N_trials + 1]
    period = np.min(np.diff(stim_times))
    t_stop = stim_times[-1] + period
    for source in sources:
        for target in targets: #NOTE different latency in inhibitory neurons
            if target == source:
                continue
            pbar.update(1)
            source_t = spiketrains[source]['times']
            target_t = spiketrains[target]['times']
            spike_trains = [source_t[source_t <= t_stop],
                            target_t[target_t <= t_stop]]
            iv = causal_connectivity(
                *spike_trains, stim_times, **iv_params)

            stim_amp = float(stim_amps[source])
            # cc, cv and stuff
            w = conn[(conn.source==source) & (conn.target==target)].weight
            n_syn = len(w)
            weight = w.sum()
            cc = corrcoef(
                spike_trains, t_stop, binsize=binsize_corr)[0, 1]
            source_cv, target_cv = coef_var(spike_trains)
            # trans_prob
            trans_prob, ppeak, pfast, ptime, cmax = transfer_probability(
                *spike_trains, **trans_prob_params)

            _hit_rate = hit_rate(
                source_t, stim_times, iv_params['x_mu'], iv_params['x_sigma'])
            r = {
                'rate_1': len(source_t) / t_stop,
                'rate_2': len(target_t) / t_stop,
                'hit_rate': _hit_rate,
                'ppeak': ppeak,
                'pfast': pfast,
                'ptime': ptime,
                'cmax': cmax,
                'cch': trans_prob,
                'iv': iv,
                'weight': weight,
                'n_syn': n_syn,
                'stim_amp': stim_amp,
                'source': source,
                'source_pop': spiketrains[source]['pop'],
                'source_cv': float(source_cv),
                'target_cv': float(target_cv),
                'cc': cc,
                'target': target,
                'target_pop': spiketrains[target]['pop']
            }
            csv_append_dict(op.join(data_path, jobname + '_analyse_{}'.format(N_trials)), r)
pbar.close()
