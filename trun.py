import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from simulator import Simulator
import pandas as pd
import numpy as np
import quantities as pq
import neo
from tools import corr_coef
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

par = {
    'msd'             : 1234, # Master seed
    'num_threads'     : 4,
    'N_neurons'       : 1250,
    'N_ex'            : 1000,
    'N_in'            : 250,
    'N_rec'           : None,
    'res'             : 0.1, # Temporal resolution for simulation Delta t in ms
    'delay'           : 1.5, # Synaptic delay in ms
    'eta'             : 1., # external poisson rate in Hz
    # Neuron parameters
    't_ref'           : 2.0, # Duration of refractory period in ms
    'V_m'             : 0.0, # Membrane potential, initial condition in mV
    'E_L'             : 0.0, # Leak reversal potential in mV
    'V_reset'         : 0.0, # Reset potential of the membrane in mV
    'tau_m'           : 20.0, # Membrane timeconstant in ms
    'C_m'             : 1.0, # Capacity of the membrane in pF
    'V_th'            : 20.0, # Spike threshold in mV
    'tau_syn_ex'      : 4., # Time constants of the excitatory synaptic exponential function in ms
    'tau_syn_in'      : 4., # Time constants of the inhibitory synaptic exponential function in ms
    # Connection parameters
    'J'               : .1,
    'g'               : 4,
    'eps'            : 0.1, # connection prob
    'J_high'         : 6., # max connection strength (low is 0)
    'p_var'          : .5, # percentage variation of mean in lognormal dist
    # Stimulation parameters
    'stim_nodes_ex'   : tuple(np.arange(1,102,1)),
    'stim_nodes_in'   : (),
    'stim_dist'       : 'poisson',
    'stim_amp_ex'     : 0.0, # pA
    'stim_amp_in'     : 0.0, # pA
    'stim_period'     : 30.0, # ms
    'stim_max_period' : 100,
    'stim_duration'   : 2.0, # ms
    'stim_N'          : 30,
}

sim = Simulator(par, fname='method_network')
sim.simulate(save=False, raster=True)
dataa = sim.data
print('rate')
print(dataa['params']['rate_ex'], dataa['params']['rate_in'])

N = 100
t_stop = dataa['params']['status']['time']
spiketrains_ex = [neo.SpikeTrain(times=sptr['times'] * pq.ms,
                              t_start=0*pq.ms, t_stop=t_stop*pq.ms)
               for sptr in dataa['spiketrains']['ex'][:N]]
spiketrains_in = [neo.SpikeTrain(times=sptr['times'] * pq.ms,
                              t_start=0*pq.ms, t_stop=t_stop*pq.ms)
               for sptr in dataa['spiketrains']['in'][:N]]
spiketrains = spiketrains_ex + spiketrains_in

ccg = corr_coef(spiketrains_ex)
m = np.abs(np.triu(ccg, k=1))
print('CC')
print(np.nanmean(m), np.nanmin(m), np.nanmax(m))
print('CV')
cv = np.array([np.diff(t).std()/np.diff(t).mean() for t in spiketrains_in if len(t) > 10])
print(cv.mean(), cv.min(), cv.max())


def get_trials(sender, pop='ex'):
    period = np.min(np.diff(dataa['epoch']['times']))
    try:
        n = [s['sender'] for s in dataa['spiketrains'][pop]].index(sender)
    except ValueError as e:
        e.args = (e.args[0] + ', unable to retreive {} in population "{}"'.format(sender, pop),)
        raise
    spike_train = dataa['spiketrains'][pop][n]['times']
    trials = [spike_train[(spike_train > t) & (spike_train  <= t + period)] - t
              for t in dataa['epoch']['times']]
    ids = [np.ones(len(t)) * idx for idx, t in enumerate(trials)]

    times = [t for trial in trials for t in trial]
    trial_num = [i for ii in ids for i in ii]
    return times, trial_num, n

# period = np.min(np.diff(dataa['epoch']['times']))
# binsize = 1
# bins = np.arange(0, period + binsize, binsize)
# B = 1
# pop = 'ex'
# conn = dataa['connections']
# target = conn.loc[(conn.source==B) &
#                   (~conn.target.isin(dataa['stim_nodes'][pop])) &
#                   (conn.target.isin(dataa['nodes'][pop]))]
# C = target.loc[target.weight==target.weight.max()].target.iloc[0]
#
# fig = plt.figure()
# gs = GridSpec(2, 1, hspace=0.05)
# ax2 = fig.add_subplot(gs[1, 0])
# ax1 = fig.add_subplot(gs[0, 0], sharex=ax2)
#
# times, trial_num, B_idx = get_trials(B)
#
# ax2.scatter(times, trial_num, color='k', s=1)
# ax1.hist(times, bins=bins, width=binsize);
# plt.setp(ax1.get_xticklabels(), visible=False)
#
# # plot C
# fig = plt.figure()
# gs = GridSpec(2, 1, hspace=0.5)
# ax2 = fig.add_subplot(gs[1, 0])
# ax1 = fig.add_subplot(gs[0, 0], sharex=ax2)
#
# times, trial_num, C_idx = get_trials(C, pop)
#
# ax2.scatter(times, trial_num, color='k', s=1)
# ax1.hist(times, bins=bins, width=binsize)
# plt.setp(ax1.get_xticklabels(), visible=False)

plt.show()
