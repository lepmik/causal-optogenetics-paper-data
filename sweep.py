import pandas as pd
import numpy as np
import quantities as pq
import os.path as op
from tools import corrcoef, csv_append_dict
from params_AI import parameters
from tqdm import tqdm
from simulator import Simulator


parameters.update({
    'stim_N': 9,
    'stim_period': 500,
    'stim_amp_ex': 0.0,
    'stim_amp_in': 0.0,
    'stim_dist': None,
})

simtime_s = (parameters['stim_N'] + 1) * parameters['stim_period'] / 1000
rate_th = 1
spike_n_th = rate_th * simtime_s
binsize_corr = 5.
N_corr = 100


def sweep(a, b, fname, amin, amax, astep, bmin, bmax, bstep, **kwargs):
    print('Sweeping ' + fname)
    arange = np.arange(amin, amax, astep)
    brange = np.arange(bmin, bmax, bstep)
    pbar = tqdm(total=len(arange) * len(brange))
    for _a in arange:
        for _b in brange:
            pbar.update(1)
            par = {a: _a, b: _b}
            if kwargs:
                par.update(kwargs)
            sim = Simulator(parameters, **par)
            sim.simulate(save=False, raster=False)
            p = sim.data['params']
            t_stop = sim.data['params']['status']['time']
            spiketrains = [s['times'] for s in sim.data['spiketrains']['ex']
                           if len(s['times']) > spike_n_th]
            if len(spiketrains) > 1:
                cv = np.array([np.diff(t).std()/np.diff(t).mean()
                               for t in spiketrains])
                ccg = corrcoef(spiketrains[:N_corr], t_stop, binsize_corr)
                m = np.triu(ccg, k=1)
                p.update({
                    'CC_mean': np.nanmean(m),
                    'CC_min': np.nanmin(m),
                    'CC_max': np.nanmax(m),
                    'CV_mean': cv.mean(),
                    'CV_min': cv.min(),
                    'CV_max': cv.max()})
                csv_append_dict(op.join('results', fname), p)
    pbar.close()


# sweep('eta', 'g', 'sweep_eta_g_tau_syn40',
#       tau_syn_in=4., tau_syn_ex=4.,
#       amin=0.1, amax=4.1, astep=.1,
#       bmin=.1, bmax=10.1, bstep=.1)

sweep('eta', 'g', 'sweep_eta_g_tau_syn10',
      tau_syn_in=1., tau_syn_ex=1.,
      amin=0.1, amax=4.1, astep=.1,
      bmin=.1, bmax=10.1, bstep=.1)

# sweep('eta', 'g', 'sweep_eta_g_tau_syn_in01_tau_syn_ex40',
#       tau_syn_in=.1, tau_syn_ex=4.,
#       amin=0.1, amax=4.1, astep=.1,
#       bmin=.1, bmax=15.1, bstep=.1)

sweep('tau_syn_in', 'tau_syn_ex', 'sweep_tau_syn_in_tau_syn_ex',
      amin=0.1, amax=4.1, astep=.1,
      bmin=0.1, bmax=4.1, bstep=.1)

sweep('gauss_std', 'gauss_mean', 'sweep_gauss_m_s',
       amin=.01, amax=2.02, astep=.02,
       bmin=.1, bmax=4.1, bstep=.1)

sweep('p_var', 'g', 'sweep_pvar_g',
      amin=.01, amax=2.02, astep=.02,
      bmin=.1, bmax=10.1, bstep=.1)

sweep('eps', 'g', 'sweep_eps_g',
      amin=0.1, amax=1.1, astep=.1,
      bmin=.1, bmax=10.1, bstep=.1)
