import sys
from method import IV
import pandas as pd
import numpy as np
import os.path as op
from tools import csv_append_dict
from tools_analysis import (corrcoef, coef_var, transfer_probability)
import quantities as pq
from tqdm import tqdm

data_path = 'results'
# trials = [30000]
trials = [5000, 10000, 15000, 20000, 25000, 30000]
N = 100
winsize_iv = 4
# latency_iv = 4

winsize_cch = 3
latency_cch = 3
binsize_corr = 5.
min_stim = 1.
source_pop = 'ex'
target_pop = 'ex'

if len(sys.argv) != 2:
    raise IOError('Usage: "python analyse.py parameters"')
param_module = sys.argv[1]
jobname = param_module.replace('.py', '')

data = np.load(op.join(data_path, jobname + '.npz'))['data'][()]

latency_iv = data['params']['delay'] + data['params']['tau_syn_ex']
conn = data['connections']

spiketrains = {}
for pop in ['ex', 'in']:
    senders = data['spiketrains'][pop]['senders']
    times = data['spiketrains'][pop]['times']
    spk = {sender: {'pop': pop, 'times': times[sender==senders]}
           for sender in np.unique(senders)}
    spiketrains.update(spk)
stim_amps = data['stim_amps'][source_pop]
sources = np.array(data['stim_nodes'][source_pop])
targets = np.array([n for n in data['nodes'][target_pop] if n not in sources])
idx_s = np.random.randint(0, len(sources), int(N / 2))
idx_t = np.random.randint(0, len(targets), int(N / 2))
sources, targets = sources[idx_s], targets[idx_t]
assert len(sources) + len(targets) == N
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
            iv = IV(*spike_trains, stim_times,
                     winsize=winsize_iv, latency=latency_iv)
            try:
                lr = iv.logreg
                logreg = float(lr.coef_)
                logreg_intercept = float(lr.intercept_)
            except ValueError:
                logreg = np.nan
                logreg_intercept = np.nan

            stim_amp = float(stim_amps[stim_amps.node==source].amp.values)
            # cc, cv and stuff
            w = conn[(conn.source==source) & (conn.target==target)].weight
            n_syn = len(w)
            weight = w.sum()
            t_stop = data['params']['status']['time']
            cc = corrcoef(spike_trains, t_stop,
                          binsize=binsize_corr)[0, 1]
            source_cv, target_cv = coef_var(spike_trains)
            # trans_prob
            trans_prob, ppeak, pfast, ptime, cmax = transfer_probability(
                *spike_trains, binsize=1, limit=15, hollow_fraction=.6,
                width=10, latency=latency_cch, winsize=winsize_cch)
            r = {
                'rate_1': len(spike_trains[0]) / t_stop,
                'rate_2': len(spike_trains[1]) / t_stop,
                'hit_rate': iv.hit_rate,
                'ppeak': ppeak,
                'pfast': pfast,
                'ptime': ptime,
                'cmax': cmax,
                'cch': trans_prob,
                'iv_wald': iv.wald,
                # 'iv_cch': iv.trans_prob,
                # 'iv_pcausal': iv.prob['pcausal'],
                # 'iv_pfast': iv.prob['pfast'],
                # 'iv_ppeak': iv.prob['ppeak'],
                # 'iv_ptime': iv.prob['ptime'],
                'logreg': logreg,
                'logreg_intercept': logreg_intercept,
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