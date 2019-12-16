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
    # Neuron parameters excitatory
    't_ref_ex'           : 2.0, # Duration of refractory period in ms
    'V_m_ex'             : 0.0, # Membrane potential, initial condition in mV
    'E_L_ex'             : 0.0, # Leak reversal potential in mV
    'V_reset_ex'         : 0.0, # Reset potential of the membrane in mV
    'tau_m_ex'           : 20.0, # Membrane timeconstant in ms
    'C_m_ex'             : 1.0, # Capacity of the membrane in pF
    'V_th_ex'            : 20.0, # Spike threshold in mV
    'tau_syn_ex_ex'      : 1., # Time constants of the excitatory synaptic exponential function in ms
    'tau_syn_in_ex'      : 1., # Time constants of the inhibitory synaptic exponential function in ms
    'E_ex_ex': 0, #Excitatory reversal potential in mV.
    'E_in_ex': -70, #Inhibitory reversal potential in mV.
    # Neuron parameters inhibitory
    't_ref_in'           : 2.0, # Duration of refractory period in ms
    'V_m_in'             : 0.0, # Membrane potential, initial condition in mV
    'E_L_in'             : 0.0, # Leak reversal potential in mV
    'V_reset_in'         : 0.0, # Reset potential of the membrane in mV
    'tau_m_in'           : 20.0, # Membrane timeconstant in ms
    'C_m_in'             : 1.0, # Capacity of the membrane in pF
    'V_th_in'            : 20.0, # Spike threshold in mV
    'tau_syn_ex_in'      : 1., # Time constants of the excitatory synaptic exponential function in ms
    'tau_syn_in_in'      : 1., # Time constants of the inhibitory synaptic exponential function in ms
    'E_ex_in': 0, #Excitatory reversal potential in mV.
    'E_in_in': -70, #Inhibitory reversal potential in mV.
    # Connection parameters
    'eta'             : .9, # external poisson rate in Hz
    'J'               : .2, # mV
    'g'               : 9.9,
    'eps'            : 0.1, # connection prob
    'J_high'         : 2.0, # max connection strength
    'J_low'          : 0.0,
    'p_var'          : .5, # percentage variation of mean in lognormal dist
    # Stimulation parameters
    'stim_N_ex'       : 800,
    'stim_N_in'       : 0,
    'stim_dist'       : 'poisson',
    'stim_amp_ex'     : 8.0, # pA
    'stim_amp_in'     : 0.0, # pA
    'stim_isi_min'    : 100.0, # ms
    'stim_duration'   : 2.0, # ms
    'stop_time'       : 2000, # s
    # Optogenetics
    'I0': 10, # light intensity leaving fibre mW/mm2
    'r': 100e-3, # 100 um radius of fiber
    'n': 1.36, # refraction index of gray matter
    'NA': 0.37, # Numerical Aperture of fiber
    'S': 10.3, # mm^-1 scattering index for rat, mouse = 11.2
    'N_pos': 10,
    'depth': .7,
    'Imax': 642, # max current pA
    'K': 0.84, # half-maximal light sensitivity of the ChR2 mW/mm2
    'n_hill': 0.76, # Hill coefficient
}