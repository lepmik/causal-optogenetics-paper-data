import quantities as pq
import numpy as np


E_L = - 60 * pq.mV

N = 20000
p = .5
N_ex = np.ceil(N*p).astype(int)
N_in = np.ceil(N*(1-p)).astype(int)
assert N_ex + N_in == N

# References:
# tau AMPA https://www.sciencedirect.com/science/article/pii/S0006349599769900
# tau GABA https://www.physiology.org/doi/full/10.1152/jn.1999.81.4.1531
# tau GABA also https://neuronaldynamics.epfl.ch/online/Ch3.S1.html
# neurons  fig 7 in https://www.ncbi.nlm.nih.gov/pubmed/?term=tewari+%5Bauthor%5D+perinueuronal+netw

parameters = {
    'setup'          : [
        'set_kernel',
        'set_nodes',
        'connect_topology',
        'set_background',
        # 'set_spatial_input'
        # 'set_oscillatory_input',
        ],
    'neuron'         : 'aeif_cond_beta_multisynapse',
    'record_from'    : ["V_m", 'w'],
    'simtime'        : 20000 * pq.ms, # Simulation time
    'msd'            : 1234, # Master seed
    'num_threads'    : 4,
    'res'            : 0.1 * pq.ms, # Temporal resolution for simulation Delta t
    # background noise
    'p_rate_ex'      : 850 * pq.Hz, # Cortical Poisson input rate to excitatory neurons
    'p_start_ex'     : None,
    'p_stop_ex'      : None,
    'p_J_ex'         : .1,
    'p_rate_in'      : 0 * pq.Hz,
    'p_stop_in'      : None,
    'p_J_in'         : 0,
    # numbers
    'N_neurons'      : N, # Total number of neurons
    'N_ex'           : N_ex,
    'N_in'           : N_in,
    'N_rec_spikes_ex': N_ex,
    'N_rec_spikes_in': N_in,
    # state rec
    'N_rec_state_ex' : N_ex,
    'N_rec_state_in' : N_in,
    # Excitatory params
    "a_ex"           : 0.,  # nS
    "b_ex"           : 0., # post spike response
    "tau_w_ex"       : 1., # Adaptation time constant
    'C_m_ex'         : 70 * pq.pF, # Membrane capacitance excitatory neurons
    'g_L_ex'         : 4 * pq.nS, # embrane conductance
    'E_L_ex'         : E_L, # Reversal potential excitatory neurons
    'V_th_ex'        : -50.0 * pq.mV, # Spike threshold excitatory neurons
    'Delta_T_ex'     : 2.5 * pq.ms, # Slope factor in
    'V_reset_ex'     : E_L, # Reset potential excitatory neurons
    "V_peak_ex"      : -40.0 * pq.mV,
    't_ref_ex'       : 1 * pq.ms, # Refractory period excitatory neurons
    'V_m_ex'         : E_L, # Initial condition
    'w_ex'           : 0.0 * pq.pA, # Initial condition
    # Inhibitory params
    "a_in"           : 0.0,  # nS
    "b_in"           : 0.0, # post spike response
    "tau_w_in"       : 1., # Adaptation time constant, gives no contributions as long w0 = a = b = 0
    'C_m_in'         : 30 * pq.pF, # Membrane capacitance excitatory neurons
    'g_L_in'         : 4.5 * pq.nS, # Membrane conductance
    'E_L_in'         : E_L, # Reversal potential excitatory neurons
    'V_th_in'        : -45.0 * pq.mV, # Spike threshold excitatory neurons
    'Delta_T_in'     : 2.5 * pq.ms, # Slope factor in
    'V_reset_in'     : E_L, # Reset potential excitatory neurons
    "V_peak_in"      : -40.0 * pq.mV,
    't_ref_in'       : 1 * pq.ms, # Refractory period excitatory neurons
    'V_m_in'         : E_L, # Initial condition
    'w_in'           : 0.0 * pq.pA, # Initial condition
    # Synaptic params
    'position'       : 'grid',
    'topology_dim'   : 2,
    'delay'          : 0.1 * pq.ms, # Synaptic delay
    # inhibitory synapses on excitatory neurons (AMPA, GABA_A)
    'E_rev_ex'       : [0., -75.],
    'tau_rise_ex'    : [1., 1.],
    'tau_decay_ex'   : [5., 5.],
    # excitatory synapses on inhibitory neurons (AMPA, NMDA, GABA_A)
    'E_rev_in'       : [0., 0., -75.],
    'tau_rise_in'    : [1., 10., 1.],
    'tau_decay_in'   : [5., 100., 5.],
    # connectivity
    'extent'         : 2 * np.pi,
    'mask_ex_in'     : {'doughnut': {'inner_radius': .2 * np.pi,
                                     'outer_radius': .4 * np.pi}},
    'mask_in_ex'     : {'circular': {'radius': .15 * np.pi}},
    # Excitatory connectivity params
    'J_ex_in'        : [0.02, 0.007], # (AMPA, NMDA)
    # Inhibitory connectivity params
    'J_in_ex'        : [0, 0.01], # (AMPA, GABA_A)
}
