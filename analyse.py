import sys
from method import IV
import pandas as pd
import numpy as np
import os.path as op
from tools import csv_append_dict, corrcoef, coef_var
import quantities as pq
from exana.statistics.tools import ccg_significance
from tqdm import tqdm

data_path = 'results'
N = 10000
winsize = 4
binsize_corr = 5.
min_stim = 1.

if len(sys.argv) != 2:
    raise IOError('Usage: "python run.py parameters"')
param_module = sys.argv[1]
jobname = param_module.replace('.py', '')

dataa = np.load(op.join(data_path, jobname + '.npz'))['data'][()]

latency = dataa['params']['delay'] + dataa['params']['tau_syn_ex']

conn = dataa['connections']
spiketrains = {s['sender']: {'pop': pop, 'times': s['times']}
               for pop in ['in', 'ex']
               for s in dataa['spiketrains'][pop]}
sources = [s for s in dataa['stim_nodes']['ex'] if s in spiketrains.keys()]
# targets = dataa['nodes']['ex'] + dataa['nodes']['in']
targets = dataa['nodes']['ex']
targets = [s for s in targets if s in spiketrains.keys()]
pbar = tqdm(total=len(sources) * (len(targets) - len(sources)))
for source in sources:
    for target in targets: #NOTE different latency in inhibitory neurons
        if target in sources:
            continue
        pbar.update(1)
        stim_times = dataa['epoch']['times'][:N + 1]
        period = np.min(np.diff(stim_times))
        t_stop = stim_times[-1] + period
        source_t = spiketrains[source]['times']
        target_t = spiketrains[target]['times']
        spike_trains = [source_t[source_t <= t_stop],
                        target_t[target_t <= t_stop]]
        iv = IV(*spike_trains, stim_times,
                 winsize=winsize, latency=latency)
        stim_amp = dataa['stim_amps'][spiketrains[source]['pop']]
        try:
            lr = iv.logreg
            logreg = float(lr.coef_)
            logreg_intercept = float(lr.intercept_)
        except ValueError:
            logreg = np.nan
            logreg_intercept = np.nan
        try:
            lr = iv.logreg_ns
            logreg_ns = float(lr.coef_)
            logreg_ns_intercept = float(lr.intercept_)
        except ValueError:
            logreg_ns = np.nan
            logreg_ns_intercept = np.nan
        # cc, cv and stuff
        w = conn[(conn.source==source) & (conn.target==target)].weight
        n_syn = len(w)
        weight = w.sum()
        t_stop = dataa['params']['status']['time']
        cc = corrcoef(spike_trains, t_stop,
                      binsize=binsize_corr)[0, 1]
        source_cv, target_cv = coef_var(spike_trains)
        # trans_prob
        try:
            pcausal, pfast, bins, cch, cch_s = ccg_significance(
                *spike_trains, limit=15, binsize=1,
                hollow_fraction=.6, width=10)
            assert len(pcausal) == len(cch)
            mask = (bins >= latency) & (bins <= latency + winsize)
            cmax = np.max(cch[mask])
            idx, = np.where(cch==cmax)
            if len(idx) > 1:
                bin_idx, = np.where(mask)
                idxmask = [i in bin_idx for i in idx]
                assert sum(idxmask) > 0
                idx = idx[idxmask]
                if len(idx) > 1:
                    idx = idx[0]
            pfast = float(pfast[idx])
            pcausal = float(pcausal[idx])
            ptime = float(bins[idx])
            trans_prob = sum(cch[mask] - cch_s[mask]) / len(spike_trains[0])
        except:
            cmax, pfast, pcausal, ptime, trans_prob = [np.nan] * 5
        r = {
            'pcausal': pcausal,
            'pfast': pfast,
            'ptime': ptime,
            'cmax': cmax,
            'trans_prob': trans_prob,
            'wald': iv.wald,
            'wald_ns': iv.wald_ns,
            'logreg': logreg,
            'logreg_ns': logreg_ns,
            'logreg_intercept': logreg_intercept,
            'logreg_ns_intercept': logreg_ns_intercept,
            'weight': weight,
            'n_syn': n_syn,
            'stim_amp': float(stim_amp[stim_amp.node==source].amp),
            'source': source,
            'source_pop': spiketrains[source]['pop'],
            'source_cv': float(source_cv),
            'target_cv': float(target_cv),
            'cc': cc,
            'target': target,
            'target_pop': spiketrains[target]['pop']
        }
        csv_append_dict(op.join(data_path, jobname + '_analyse'), r)
pbar.close()
