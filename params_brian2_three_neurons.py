from brian2 import ms, Hz, mV, nS, pF, pA

parameters = {
    'msd'             : 1234, # Master seed
    'N_neurons'       : 3,
    'N_ex'            : 3,
    'N_in'            : 0,
    'res'             : 0.1 * ms, # Temporal resolution for simulation Delta t in ms
    'syn_delay'       : 1.5 * ms, # Synaptic delay in ms
    # Neuron parameters
    't_ref'           : 4.0 * ms, # Duration of refractory period in ms
    'V_m_init'        : -70.0 * mV, # Membrane potential, initial condition in mV
    'E_L'             : -70.0 * mV, # Leak reversal potential in mV
    'V_reset'         : -70.0 * mV, # Reset potential of the membrane in mV
    'g_L'             : 16.7 * nS, # Membrane leak
    'C_m'             : 250.0 * pF, # Capacity of the membrane in pF
    'V_th'            : -50.0 * mV, # Spike threshold in mV
    'tau_syn_ex'      : 0.326 * ms, # Time constants of the excitatory synaptic exponential function in ms
    'tau_syn_in'      : 0.326 * ms, # Time constants of the inhibitory synaptic exponential function in ms
    'E_ex'            : 0. * mV, #Excitatory reversal potential in mV.
    'E_in'            : -80. * mV, #Inhibitory reversal potential in mV.
    # Background parameters
    's_sin'          : 100 * pA, # size of sinusoidal fluctuations
    'r_sin'          : 10 * Hz, # rate of sinusoidal fluctuations
    # Connection parameters
    'rate_p'          : 1. * Hz, # external poisson rate
    'N_p'             : 29000, # number of poisson inputs to each neuron
    'J_ex'            : .68 * nS,
    'J_02'            : 10 * nS,
    # Stimulation parameters
    'stim_N_ex'       : 1000,
    'stim_amp_ex'     : 700.0 * pA, # pA
    'stim_amp_in'     : 0.0, # pA
    'stim_duration'   : 2.0 * ms, # ms
    'simtime_stim'    : 15.0 * ms, # ms
    'init_simtime'    : 30. *  ms, #ms
    'runtime'         : 1000 * ms, #ms
    't_dist_min'      : 5.*ms, # minimal distance between stimulations
    't_dist_max'      : 10.*ms, # maximal distance between stimulations
}
