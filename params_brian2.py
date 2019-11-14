from brian2 import ms, Hz, mV, nS, pF, pA

parameters = {
    'msd'             : 1234, # Master seed
    'num_threads'     : 20,
    'N_neurons'       : 4000,
    'N_ex'            : 3200,
    'N_in'            : 800,
    'res'             : 0.1*ms, # Temporal resolution for simulation Delta t in ms
    'syn_delay'       : 1.5*ms, # Synaptic delay in ms
    'n_save_spikes'   : 100, # number of branches before spikes are saved
    # Neuron parameters
    't_ref'           : 4.0*ms, # Duration of refractory period in ms
    'V_m_init'        : -70.0*mV, # Membrane potential, initial condition in mV
    'E_L'             : -70.0*mV, # Leak reversal potential in mV
    'V_reset'         : -70.0*mV, # Reset potential of the membrane in mV
    'g_L'             : 16.7*nS, # Membrane leak
    'C_m'             : 250.0*pF, # Capacity of the membrane in pF
    'V_th'            : -50.0*mV, # Spike threshold in mV
    'tau_syn_ex'      : 0.326*ms, # Time constants of the excitatory synaptic exponential function in ms
    'tau_syn_in'      : 0.326*ms, # Time constants of the inhibitory synaptic exponential function in ms
    'E_ex'            : 0.*mV, #Excitatory reversal potential in mV.
    'E_in'            : -80.*mV, #Inhibitory reversal potential in mV.
    # Connection parameters
    'rate_p'          : 1. * Hz, # external poisson rate
    'N_p'             : 27500, # number of poisson inputs to each neuron
    's_sin'          : 100 * pA, # size of sinusoidal fluctuations
    'r_sin'          : 10 * Hz, # rate of sinusoidal fluctuations    
    'eps_p'           : 0.1, # poisson rate = rate_p * C_p, C_p = eps_p * N_neurons
    'J_ex'            : .68 *nS, #
    'g'               : 2.7, #
    'eps'            : 0.1, # connection prob
    'J_high'         : 5.*nS, # max connection strength (0.05 ~ 5 mV)
    'J_low'          : 0.0*nS,
    'p_var'          : 0.1, # percentage variation of mean in lognormal dist
    # Stimulation parameters
    'stim_N_ex'       : 1000,    
    'stim_amp_ex'     : 900.0 * pA, # pA
    'stim_amp_in'     : 0.0, # pA
    't_pre_stim'      : 2.0*ms, # time before a stimulation
    't_stim'   : 2.0*ms, # ms
    't_after_stim' : 4.0*ms, # ms    
    'init_simtime'    : 500.* ms, #ms
    'runtime'         : 1000000.*ms, #ms
    't_dist_min'      : 4.*ms, # minimal distance between stimulations
    't_dist_max'      : 10.*ms, # maximal distance between stimulations
}
