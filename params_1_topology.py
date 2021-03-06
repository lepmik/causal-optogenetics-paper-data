import numpy as np

parameters = {
    'setup': [
        'set_kernel',
        'set_nodes',
        'set_connections_topology',
        # 'create_nodes',
        # 'set_connections_from_file',
        'set_background_ex',
        'set_spike_rec',
        # 'set_state_rec',
        'simulate_trials_branch'
    ],
    'msd'             : 50311, # Master seed
    'num_threads'     : 20,
    'N_neurons'       : 4000,
    'N_ex'            : 3200,
    'N_in'            : 800,
    'N_rec_spike_ex'  : None,
    'N_rec_spike_in'  : None,
    'N_rec_state_ex'  : None,
    'N_rec_state_in'  : None,
    'res'             : 0.1, # Temporal resolution for simulation Delta t in ms
    'delay'           : 1.5, # Synaptic delay in ms
    # Neuron parameters
    't_ref'           : 4.0, # Duration of refractory period in ms
    'V_m'             : 0.0, # Membrane potential, initial condition in mV
    'E_L'             : 0.0, # Leak reversal potential in mV
    'V_reset'         : 0.0, # Reset potential of the membrane in mV
    'tau_m'           : 20.0, # Membrane timeconstant in ms
    'C_m'             : 1.0, # Capacity of the membrane in pF
    'V_th'            : 20.0, # Spike threshold in mV
    'tau_syn_ex'      : 1., # Time constants of the excitatory synaptic exponential function in ms
    'tau_syn_in'      : 1., # Time constants of the inhibitory synaptic exponential function in ms
    # Connection parameters
    'J_ex'            : 1., # mV
    'g'               : 5.,
    'J_p'             : .2, # mV
    'rate_p'          : 1500., # connection prob
    # Topology
    'position'       : 'grid',
    'topology_dim'   : 1,
    'extent'         : 2 * np.pi,
    'mask_ex_in'     : {'doughnut': {
                            'inner_radius': .5 * np.pi,
                            'outer_radius': 1. * np.pi},
                        "anchor": [0., 0.2]
                        },
    'mask_in_ex'     : {'circular': {'radius': .1 * np.pi}},
    # Stimulation parameters
    'init_simtime'    : 1000., # ms
    'stim_N_ex'       : 1000,
    'stim_N_in'       : 0,
    'stim_isi_min'    : 4.0, # ms
    'post_stimtime'   : 15.0, # ms
    'stim_duration'   : 2.0, # ms
    'stim_trials'     : 100000, #s
    'stim_amp_ex'     : 0.0, # pA
    'stim_amp_in'     : 0.0, # pA
    # Optogenetics
    'I0': 10, # light intensity leaving fibre mW/mm2
    'r': 100e-3, # 100 um radius of fiber
    'n': 1.36, # refraction index of gray matter
    'NA': 0.37, # Numerical Aperture of fiber
    'S': 10.3, # mm^-1 scattering index for rat, mouse = 11.2
    'N_pos': 100,
    'depth': .7,
    'Imax': 642, # max current pA
    'K': 0.84, # half-maximal light sensitivity of the ChR2 mW/mm2
    'n_hill': 0.76, # Hill coefficient
}
