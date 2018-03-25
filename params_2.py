from quantities import s, ms, mV, pF, Hz, nS, pA
import numpy as np

parameters = {
    'msd'             : 1221, # Master seed
    'num_threads'     : 4,
    'N_neurons'       : 1250,
    'N_ex'            : 1000,
    'N_in'            : 250,
    'N_rec_spike_ex'  : None,
    'N_rec_spike_in'  : None,
    'N_rec_state_ex'  : None,
    'N_rec_state_in'  : None,
    'verbose'         : False,
    'res'             : 0.1, # Temporal resolution for simulation Delta t in ms
    'delay'           : 1.5, # Synaptic delay in ms
    # Neuron parameters
    't_ref'           : 2.0, # Duration of refractory period in ms
    'V_m'             : 0.0, # Membrane potential, initial condition in mV
    'E_L'             : 0.0, # Leak reversal potential in mV
    'V_reset'         : 0.0, # Reset potential of the membrane in mV
    'tau_m'           : 20.0, # Membrane timeconstant in ms
    'C_m'             : 1.0, # Capacity of the membrane in pF
    'V_th'            : 20.0, # Spike threshold in mV
    'tau_syn_ex'      : 1., # Time constants of the excitatory synaptic exponential function in ms
    'tau_syn_in'      : 1., # Time constants of the inhibitory synaptic exponential function in ms
    # Connection parameters
    'eta'             : .9, # external poisson rate in Hz
    'J'               : .2, # mV
    'g'               : 4.4,
    'eps'            : 0.1, # connection prob
    'J_high'         : 2.05, # max connection strength
    'J_low'          : 0.05,
    'p_var'          : .5, # percentage variation of mean in lognormal dist
    # Stimulation parameters
    'stim_N_ex'       : 950,
    'stim_N_in'       : 0,
    'stim_dist'       : 'poisson',
    'stim_amp_ex'     : 10.0, # pA
    'stim_amp_in'     : 0.0, # pA
    'stim_period'     : 100.0, # ms
    'stim_max_period' : 150, # only applies to poisson
    'stim_duration'   : 2.0, # ms
    'stim_N'          : 30000,
    # Optogenetics
    'r': 100e-3, # 100 um radius of fiber
    'n': 1.36, # refraction index of gray matter
    'NA': 0.37, # Numerical Aperture of fiber
    'S': 10.3, # mm^-1 scattering index for rat, mouse = 11.2
    'density': 10e4, # N/mm³
}
