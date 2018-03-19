from quantities import s, ms, mV, pF, Hz, nS, pA
import numpy as np

parameters = {
    'msd'             : 1234, # Master seed
    'num_threads'     : 4,
    'N_neurons'       : 1250,
    'N_ex'            : 1000,
    'N_in'            : 250,
    'N_rec_spike_ex'  : None,
    'N_rec_spike_in'  : None,
    'N_rec_state_ex'  : None,
    'N_rec_state_in'  : None,
    'res'             : 0.1, # Temporal resolution for simulation Delta t in ms
    'delay'           : 1.5, # Synaptic delay in ms
    'eta'             : 1.0, # external poisson rate in Hz
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
    'J'               : .5, # mV
    'g'               : 9.4,
    'eps'            : 0.1, # connection prob
    'J_high'         : 15., # max connection strength (low is 0)
    'p_var'          : .5, # percentage variation of mean in lognormal dist
    # Stimulation parameters
    'stim_nodes_ex'   : tuple(np.arange(1,602,1)),
    'stim_nodes_in'   : (),
    'stim_dist'       : None,
    'stim_amp_ex'     : 5.0, # pA
    'stim_amp_in'     : 0.0, # pA
    'stim_period'     : 30.0, # ms
    'stim_max_period' : 100, # only applies to poisson
    'stim_duration'   : 2.0, # ms
    'stim_N'          : 30000,
    # channelnoise
    'gauss_mean'      : 0.,
    'gauss_std'       : 0.
}
