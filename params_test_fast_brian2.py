from brian2 import ms, second, Hz, mV, pA, nS, pF

parameters = {
    'msd'             : 1234, # Master seed
    'num_threads'     : 8,
    'N_neurons'       : 4000,
    'N_ex'            : 3200,
    'N_in'            : 800,
    'N_rec_spike_ex'  : None,
    'N_rec_spike_in'  : None,
    'N_rec_state_ex'  : None,
    'N_rec_state_in'  : None,
    'res'             : 0.1*ms, # Temporal resolution for simulation Delta t in ms
    'syn_delay'       : 1.5*ms, # Synaptic delay in ms
    # Neuron parameters
    't_ref'           : 4.0*ms, # Duration of refractory period in ms
    'V_m'             : -70.0*mV, # Membrane potential, initial condition in mV
    'E_L'             : -70.0*mV, # Leak reversal potential in mV
    'V_reset'         : -70.0*mV, # Reset potential of the membrane in mV
    'g_L'             : 16.7, # Membrane leak
    'C_m'             : 250.0*pF, # Capacity of the membrane in pF
    'V_th'            : -50.0*mV, # Spike threshold in mV
    'tau_syn_ex'      : 0.326*ms, # Time constants of the excitatory synaptic exponential function in ms
    'tau_syn_in'      : 0.326*ms, # Time constants of the inhibitory synaptic exponential function in ms
    'E_ex'            : 0.*mV, #Excitatory reversal potential in mV.
    'E_in'            : -80.*mV, #Inhibitory reversal potential in mV.
    # Connection parameters
    'rate_p'          : 25.5, # external poisson rate
    # 'eps_p'           : 0.25, # poisson rate = rate_p * C_p, C_p = eps_p * N_neurons
    'eps_p'           : 0.1, # poisson rate = rate_p * C_p, C_p = eps_p * N_neurons
    'J_ex'            : .68, #
    'g'               : 2.7, #
    'eps'            : 0.1, # connection prob
    'J_high'         : 5., # max connection strength (0.05 ~ 5 mV)
    'J_low'          : 0.0,
    'p_var'          : 0.1, # percentage variation of mean in lognormal dist
    # Stimulation parameters
    # 'stim_N_ex'       : 7000,
    'stim_N_ex'       : 2000,
    'stim_N_in'       : 0,
    #'stim_dist'       : 'poisson',
    'stim_amp_ex'     : 900.0, # pA
    'stim_amp_in'     : 0.0, # pA
    #'stim_rate'       : 30.0,
    #'stim_isi_min'    : 200.0, # ms
    'stim_duration'   : 2.0, # ms
    #'stop_time'       : 16000, #s
    'n_trials'        : 4,
    'init_simtime'    : 600., #ms
    'simtime'         : 20, #ms
    'delay'           : 20, #ms
    # Optogenetics
    'I0': 10, # light intensity leaving fibre mW/mm2 DONT ADJUST, USE stim_amp_ex
    'r': 100e-3, # 100 um radius of fiber
    'n': 1.36, # refraction index of gray matter
    'NA': 0.37, # Numerical Aperture of fiber
    'S': 10.3, # mm^-1 scattering index for rat, mouse = 11.2
    'N_pos': 200,
    'depth': .7,
    'Imax': 642, # max current pA
    'K': 0.84, # half-maximal light sensitivity of the ChR2 mW/mm2
    'n_hill': 0.76, # Hill coefficient
}
