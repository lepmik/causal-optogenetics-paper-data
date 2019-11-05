import nest
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

nest.set_verbosity(100)

parameters = {
    'msd'            : 1234, # Master seed
    'num_threads'    : 4,
    'N_neurons'      : 3,
    'res'            : 0.1, # Temporal resolution for simulation Delta t in ms
    'delay'          : 1.5, # Synaptic delay in ms
    'record_state'   : False,
    'verbose'        : False,
    # Drive
    'poisson_rate'   : 1000., # external poisson rate in Hz
    'poisson_J'      : .3,
    'gauss_mean'     : .1,
    'gauss_std'      : 1.1,
    'ac_amp'         : 1.,
    'ac_offset'      : -.5,
    'ac_freq'        : 10.,
    # Neuron parameters
    't_ref'          : 4.0, # Duration of refractory period in ms
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
    'J_AC'           : 2.,
    'J_BC'           : 0,
    'J_BA'           : 0,
    'J_CA'           : 0,
    'J_CB'           : 0,
    'C'              : 1, # indegree
    # Stimulation parameters
    'stim_amp_A'      : 6.0, # pA
    'stim_amp_B'      : 6.0, # pA
    'stim_rate'       : 30.0, # Hz
    'stim_duration'   : 2.0, # ms
    'stop_time'       : 500, # s
    'stim_isi_min'    : 30e-3,
}


def prune(a, ref):
    b = np.concatenate(([False], np.diff(a) < ref))
    c = np.concatenate(([False], np.diff(b.astype(int)) > 0))
    d = a[~c]
    if any(np.diff(a) < ref):
        d = prune(d, ref)
    return d


def generate_stim_times(stim_rate, stop_time, stim_isi_min):
    stim_times = np.sort(np.random.uniform(
        0, stop_time, int(stim_rate * stop_time)))
    return prune(stim_times, stim_isi_min)
    return stim_times


def simulate(par=None, pbar=None, **kwargs):
    if pbar is None:
         pbar = tqdm

    def vprint(*args):
        if kwargs.get('verbose', False):
            print(*args)
    if par is not None:
        par = copy.deepcopy(par)
    else:
        par = parameters
    if kwargs:
        assert all(k in par for k in kwargs.keys())
        par.update(kwargs)
    stim_times = generate_stim_times(
        par['stim_rate'], par['stop_time'], par['stim_isi_min']) * 1000
    stim_times = stim_times.round(1) # round to simulation resolution

    vprint('simulating ', stim_times[-1])
    vprint('stimulating ', len(stim_times))

    # Set kernel
    nest.ResetKernel()
    nest.SetKernelStatus({"local_num_threads": par['num_threads'], "print_time": False})
    N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
    pyrngs = [np.random.RandomState(s) for s in range(par['msd'], par['msd'] + N_vp)]
    nest.SetKernelStatus({'grng_seed' : par['msd'] + N_vp})
    nest.SetKernelStatus({'rng_seeds' : range(par['msd'] + N_vp + 1, par['msd'] + 2 * N_vp + 1)})
    nest.SetStatus([0], [{"resolution": par['res']}])

    # Make the nodes
    nodes = nest.Create('iaf_psc_alpha', par['N_neurons'])
    keys = [
        't_ref', 'V_m', 'E_L', 'V_reset', 'tau_m', 'C_m',
        'V_th', 'tau_syn_ex', 'tau_syn_in'
    ]
    nest.SetStatus(nodes, [{k: par[k] for k in keys}])

    # Connect nodes
    nn = {
        'A': tuple([nodes[0]]),
        'B': tuple([nodes[1]]),
        'C': tuple([nodes[2]])
    }

    for key in ['J_AB', 'J_AC', 'J_BA', 'J_BC', 'J_CA', 'J_CB']:
        j = par.get(key)
        if j != 0 and j is not None:
            vprint('connecting ', key, j)
            s, r = nn[key[-2]], nn[key[-1]]
            nest.Connect(
                s, r,
                {
                    'rule': 'fixed_indegree', 'indegree': par['C']
                },
                {
                    "weight": j, "delay": par['delay']
                }
            )

    # Set background drive
    background = nest.Create(
        "poisson_generator", 1, params={
            "rate": par['poisson_rate']
        }
    )
    nest.Connect(
        background, nodes,
        {
            'rule': 'fixed_indegree', 'indegree': 1},
        {
            "weight": par['poisson_J'], "delay": par['res']
        }
    )

    # Set confounding drive
    ac = nest.Create(
        "ac_generator", 1, params={
            "offset": par['ac_offset'],
            'amplitude': par['ac_amp'],
            'frequency': par['ac_freq']
        }
    )
    nest.Connect(ac, nodes)

    # Set channel noise
    channelnoise = nest.Create(
        "noise_generator", 1, params={
            "mean": par['gauss_mean'],
            'std': par['gauss_std']
        }
    )
    nest.Connect(channelnoise, nodes)

    # Connect spike detector
    spks = nest.Create(
        "spike_detector", 1, params=[{
            "label": "spikes", "to_file": False
        }]
    )
    # connect using all_to_all: all recorded excitatory neurons to one detector
    nest.Connect(nodes, spks)

    if par.get('record_state'):
        vm = nest.Create("multimeter",
            params = {"interval": par['res'],
                      "record_from": ['V_m'],
                      "withgid": True, "to_file": False,
                      "label": 'state', "withtime": True})
        nest.Connect(vm, nodes)

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
    for s in pbar(np.diff(stim_times)):
        for stim in stims:
            nest.SetStatus(stim, {'origin': nest.GetKernelStatus()['time']})
        nest.Simulate(s)
    nest.Simulate(np.min(np.diff(stim_times)))

    # Organize data
    conns = nest.GetConnections(source=nodes, target=nodes)
    conn_include = ('weight', 'source', 'target')
    conns = list(nest.GetStatus(conns, conn_include))
    conns = pd.DataFrame(conns, columns=conn_include)

    data = {
        'spiketrains': pd.DataFrame(nest.GetStatus(spks, 'events')[0]),
        'names': ['A', 'B', 'C'],
        'stim_times': stim_times,
        'connections':conns,
        'status': nest.GetKernelStatus(),
        'params': par
    }
    if par.get('record_state'):
        data.update({
            'state': pd.DataFrame(nest.GetStatus(vm, 'events')[0])
        })
    return data

if __name__ == '__main__':
    dataa = simulate(par, stop_time=4000)
    np.savez('results/triple_4000.npz', data=dataa)
