import nest
import pandas as pd
import numpy as np

import copy

par = {
    'msd'            : 1234, # Master seed
    'num_threads'    : 4,
    'N_neurons'      : 3,
    'res'            : 0.1, # Temporal resolution for simulation Delta t in ms
    'delay'          : 1.5, # Synaptic delay in ms
    'rate_p'         : 1000., # external poisson rate in Hz
    'J_p'            : .3,
    'gauss_mean'     : .1,
    'gauss_std'      : 1.1,
    # Neuron parameters
    't_ref'          : 2.0, # Duration of refractory period in ms
    'V_m'            : 0.0, # Membrane potential, initial condition in mV
    'E_L'            : 0.0, # Leak reversal potential in mV
    'V_reset'        : 0.0, # Reset potential of the membrane in mV
    'tau_m'          : 20.0, # Membrane timeconstant in ms
    'C_m'            : 1.0, # Capacity of the membrane in pF
    'V_th'           : 20.0, # Spike threshold in mV
    'tau_syn_ex'     : 1., # Time constants of the excitatory synaptic exponential function in ms
    'tau_syn_in'     : 1., # Time constants of the inhibitory synaptic exponential function in ms
    # Connection parameters
    'J_AB'           : 0., # mV
    'J_AC'           : 0,
    'J_BC'           : 0.,
    'J_BA'           : 0,
    'J_CA'           : 0,
    'J_CB'           : 0,
    'J_DC'           : 2.,
    'C'              : 1, # indegree
    # Stimulation parameters
    'stim_amp_A'      : 10.0, # pA
    'stim_amp_B'      : 10.0, # pA
    'stim_period'     : 30.0, # ms
    'stim_max_period' : 60.0, # only applies to poisson
    'stim_duration'   : 2.0, # ms
    'stim_N'          : 30000,
    'stim_dist'       : 'poisson'
}


def poisson_clipped(N, period, low, high):
    poisson = []
    while len(poisson) < N:
        p = - np.log(1 - np.random.uniform(0, 1)) * period
        if p >= low and p <= high:
            poisson.append(p)
    stim_times = [poisson[0]]
    for idx, isi in enumerate(poisson[1:]):
        stim_times.append(stim_times[idx] + isi)
    return np.array(stim_times).round()


def simulate(par, **kwargs):
    par = copy.deepcopy(par)
    if kwargs:
        assert all(k in par for k in kwargs.keys())
        par.update(kwargs)
    if par['stim_dist'] is None:
        stim_times = np.linspace(par['stim_period'],
                                 par['stim_N'] * par['stim_period'],
                                 par['stim_N'])
    elif par['stim_dist'] == 'poisson':
        stim_times = poisson_clipped(
                N=par['stim_N'], period=par['stim_period'],
                low=par['stim_period'], high=par['stim_max_period'])
    print('simulating ', stim_times[-1])
    print('stimulating ', len(stim_times))
    # Set kernel
    nest.ResetKernel()
    nest.SetKernelStatus({"local_num_threads": par['num_threads']})
    N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
    pyrngs = [np.random.RandomState(s) for s in range(par['msd'], par['msd'] + N_vp)]
    nest.SetKernelStatus({'grng_seed' : par['msd'] + N_vp})
    nest.SetKernelStatus({'rng_seeds' : range(par['msd'] + N_vp + 1, par['msd'] + 2 * N_vp + 1)})
    nest.SetStatus([0], [{"resolution": par['res']}])

    # Make the nodes
    nodes = nest.Create('iaf_psc_alpha', par['N_neurons'])
    keys = ['t_ref', 'V_m', 'E_L', 'V_reset', 'tau_m', 'C_m', 'V_th', 'tau_syn_ex', 'tau_syn_in']
    nest.SetStatus(nodes, [{k: par[k] for k in keys}])
    
    spike_generator = nest.Create('spike_generator', params = {'spike_times': [10.0, 20.0, 50.0]})

    nodes = nodes + spike_generator
    
    # Connect nodes
    nn = {
        'A': tuple([nodes[0]]),
        'B': tuple([nodes[1]]),
        'C': tuple([nodes[2]]),
        'D': tuple([nodes[3]])
    }
    
    for key in ['J_AB', 'J_AC', 'J_BA', 'J_BC', 'J_CA', 'J_CB', 'J_DC']:
        j = par.get(key)
        if j != 0 and j is not None:
            print('connecting ', key, j)
            s, r = nn[key[-2]], nn[key[-1]]
            conn_dict = {'rule': 'fixed_indegree', 'indegree': par['C']}
            nest.Connect(s, r, conn_dict,
                         {"weight": j, "delay": par['delay']})

    # Set background drive
    background = nest.Create("poisson_generator", 1,
                              params={"rate": par['rate_p']})
    nest.Connect(background, nodes,
                 {'rule': 'fixed_indegree', 'indegree': 1},
                 {"weight": par['J_p'], "delay": par['res']})

    # Set channel noise
    channelnoise = nest.Create("noise_generator", 1,
                              params={"mean": par['gauss_mean'],
                                      'std': par['gauss_std']})
    nest.Connect(channelnoise, nodes)

    # Connect spike detector
    spks = nest.Create("spike_detector", 1,
                         params=[{"label": "Exc", "to_file": False}])
    # connect using all_to_all: all recorded excitatory neurons to one detector
    nest.Connect(nodes, spks)

    # Simulate one period without stimulation
    nest.Simulate(stim_times[0])

    # Set dc stimulation
    stims = []
    for n, a in enumerate([par['stim_amp_A'], par['stim_amp_B']]):
        stim = nest.Create(
            "dc_generator",
            params={'amplitude': a,
                    'start': 0.,
                    'stop': par['stim_duration']})
        nest.Connect(stim, tuple([nodes[n]]))
        stims.append(stim)

    # Run multiple trials
    for s in np.diff(stim_times):
        for stim in stims:
            nest.SetStatus(stim, {'origin': nest.GetKernelStatus()['time']})
        nest.Simulate(s)
    nest.Simulate(np.min(np.diff(stim_times)))

    # Organize data
    conns = nest.GetConnections(source=nodes, target=nodes)
    df = pd.DataFrame(nest.GetStatus(spks, 'events')[0])
    names = ['A', 'B', 'C']
    spiketrains = [{'times': np.array(attr['times']),
                    'sender': sndr,
                    'name': names[sndr - 1]}
                   for sndr, attr in df.groupby('senders')]
    data = {
        'spiketrains': spiketrains,
        'epoch': {'times': stim_times,
                  'durations': [par['stim_duration']] * len(stim_times)},
        'connections': list(nest.GetStatus(conns, ('weight', 'source', 'target'))),
        'status': nest.GetKernelStatus(),
        'params': par
    }
    return data

if __name__ == '__main__':
    dataa = simulate(par, stim_N=30000)
    np.savez('triple', data=dataa)
