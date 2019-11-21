import brian2
# brian2.prefs.codegen.target = 'cython'
# brian2.prefs.codegen.target = 'numpy'
import numpy as np
from params_brian2_three_neurons import parameters as p
import os
import sys
from brian2 import ms, pA, nS, second
from tqdm import tqdm

eqs = '''dV_m/dt = (g_L*(E_L-V_m)+Ie+Ii+I+Ix)/(C_m) : volt
         Ie = ge*(E_ex-V_m) : amp
         Ii = gi*(E_in-V_m) : amp
         dge/dt = -ge/(tau_syn_ex) : siemens
         dgi/dt = -gi/(tau_syn_in) : siemens
         Ix = sizes*0.5*(1+sin(2*pi*rates*t)) : amp
         rates : Hz
         sizes : amp
         I : amp'''


def branched_triple(progress_bar=None, **kwargs):
    if kwargs:
        p.update(kwargs)
    brian2.seed(p['msd'])
    data_path = 'three_neuron_brian'
    os.makedirs(data_path, exist_ok=True)

    nodes = brian2.NeuronGroup(
        p['N_neurons'],
        model=eqs,
        threshold='V_m > V_th',
        reset='V_m = V_reset',
        refractory=p['t_ref'],
        namespace=p,
        method='euler',
        dt=p['res']
    )
    nodes_stim = nodes[:2]

    nodes.sizes = p['s_sin']
    nodes.rates= p['r_sin']

    syn_ex = brian2.Synapses(
        nodes,
        nodes,
        model='w:siemens',
        on_pre='ge+=w',
        delay=p['syn_delay']
    )

    syn_ex.connect(i=0, j=2)
    syn_ex.w = p['J_02']

    poissInp = brian2.PoissonInput(
        nodes, 'ge',
        N=p['N_p'],
        rate=p['rate_p'],
        weight=p['J_ex']
    )

    # 1 for times without stim
    spk_mon1 = brian2.SpikeMonitor(nodes)

    # 2 for stimulation periods
    spk_mon2 = brian2.SpikeMonitor(nodes)
    spk_mon2_t = []
    spk_mon2_i = []
    spk_mon2.active = False

    stim_times = []
    next_stop = []

    # find last index +1 of stim supgroup
    idx_stop_pl_1 = nodes_stim.stop
    # make sure that stim subgroups starts with 0
    assert nodes_stim.start == 0


    @brian2.network_operation(when='end')
    def stop_for_stim(t):
        if  t / ms > next_stop[0]:
            # and theres a new spike
            brian2.stop()

    if progress_bar is not None:
        pbar = progress_bar(total=int(p['runtime'] / p['res']))
    # run init time without stimulation
    stop_for_stim.active = False
    brian2.run(p['init_simtime'])
    # sys.stdout.write('\r'+str(brian2.defaultclock.t / ms))
    t2 = (brian2.defaultclock.t - p['res']) / ms
    next_stop.append(t2 + np.random.uniform(p['t_dist_min'] / ms, p['t_dist_max'] / ms))
    # now with stimulation
    stop_for_stim.active = True

    brian2.run(p['runtime'] - brian2.defaultclock.t)

    while brian2.defaultclock.t < p['runtime']:
        # stimulation only after init_simtime
        stop_for_stim.active = False
        # get timepoint of branching, shift by the delay of 0.1 ms
        if progress_bar is not None:
            pbar.update(int((brian2.defaultclock.t / ms - t2) / (p['res'] / ms)))
        t2 = (brian2.defaultclock.t - p['res']) / ms
        next_stop[0] = t2 + np.random.uniform(p['t_dist_min'] / ms, p['t_dist_max'] / ms)
        stim_times.append(t2)
        # store network state before stimulation
        brian2.store()
        # We'll need an explicit seed here, otherwise we'll continue with different
        # random numbers after the restore
        # use_seed = brian2.randint(brian2.iinfo(np.int32).max)
        # brian2.seed(use_seed)
        # change spike monitors
        spk_mon1.active = False
        spk_mon2.active = True
        # stimulate
        nodes_stim.I = p['stim_amp_ex']
        brian2.run(p['stim_duration'])
        # turn stimuli off, but keep on simulation
        t_left = p['simtime_stim'] - p['stim_duration']
        nodes_stim.I = 0. * pA
        brian2.run(t_left)
        # store data of intermittent run
        spk_mon2_t.append(np.array(spk_mon2.t / ms))
        spk_mon2_i.append(np.array(spk_mon2.i).astype(int))
        # restore previous network state and continue with simulation
        stop_for_stim.active = True
        brian2.restore()
        brian2.seed(p['msd'])
        spk_mon1.active = True
        spk_mon2.active = False
        # sys.stdout.write('\r' + str(brian2.defaultclock.t / ms))
        brian2.run(p['runtime'] - brian2.defaultclock.t)
    # sys.stdout.write('\r'+str(brian2.defaultclock.t/ms))
    if progress_bar is not None:
        pbar.update(int((brian2.defaultclock.t / ms - t2) / (p['res'] / ms)))
        pbar.close()
    return np.array(stim_times), np.array(spk_mon1.i).astype(int), np.array(spk_mon1.t / ms), spk_mon2_i, spk_mon2_t


if __name__ == '__main__':
    from collections import defaultdict
    from joblib import Parallel, delayed
    J_ACs = np.arange(0, 15.2, .2)
    simulations_J_AC = defaultdict(list)
    for a, result in zip(J_ACs, Parallel(n_jobs=8)(delayed(branched_triple)(
        runtime=100 * second, J_02=a * nS) for a in tqdm(J_ACs))):
        stim_times, spk_base_i, spk_base_t, spk_branch_i, spk_branch_t = result

        simulations_J_AC['stim_times'].append(stim_times)
        simulations_J_AC['spk_base_i'].append(spk_base_i)
        simulations_J_AC['spk_base_t'].append(spk_base_t)
        simulations_J_AC['spk_branch_i'].append(spk_branch_i)
        simulations_J_AC['spk_branch_t'].append(spk_branch_t)
        simulations_J_AC['J_AC'].append(a)

    np.savez('simulations_J_AC_osc_brian2.npz', data=simulations_J_AC)
