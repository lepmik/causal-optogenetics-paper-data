from quantities import s, ms, mV, pF, Hz, nS, pA
import numpy as np

g = 5.0  # ratio inhibitory weight/excitatory weight
eta = 2.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability

N_neurons = 1000
P_ex = 0.8

N_ex = int(np.ceil(N_neurons * P_ex))
N_in = int(np.ceil(N_neurons * (1 - P_ex)))
C_ex = int(epsilon * N_ex)  # number of excitatory synapses per neuron
C_in = int(epsilon * N_in)  # number of inhibitory synapses per neuron
C_tot = int(C_in + C_ex)      # total number of synapses per neuron

tau_m = 20.0  # time constant of membrane potential in ms
theta = 20.0  # membrane threshold potential in mV
J = 0.1   # postsynaptic amplitude in mV
nr_ports = 100  # number of receptor types
tau_syn = [0.1 + 0.01 * i for i in range(nr_ports)]
C_m = 1.0
t_ref = 2.0
J_ex = J       # amplitude of excitatory postsynaptic current
J_in = - g * J_ex  # amplitude of inhibitory postsynaptic current

nu_th = theta / (J * C_ex * tau_m)
nu_ex = eta * nu_th
p_rate = 1000.0 * nu_ex * C_ex

parameters = {
    'neuron'         : 'iaf_psc_exp_multisynapse',
    'record_from'    : ["V_m"],
    'msd'            : 1234, # Master seed
    'J_bg'           : J_ex, # Cortical input efficacy to neurons
    'p_rate'         : p_rate * Hz, # Cortical Poisson input rate to neurons
    'N_neurons'      : N_neurons, # Total number of neurons
    'N_rec'          : N_neurons, # Number of neurons to record from
    'simtime'        : np.nan, # Simulation time
    'res'            : 0.1 * ms, # Temporal resolution for simulation Delta t
    'delay'          : 1.5 * ms, # Synaptic delay
    'nr_ports'       : nr_ports,
    'P_ex'           : P_ex,
    'num_trials'     : 10,
    'trial_start'    : 100 * ms,
    'trial_duration' : 5 * ms,
    'dc_amp'         : 10 * pA,
    'N_stim'         : 100,
    # Excitatory parameters
    't_ref_ex'       : t_ref, # Duration of refractory period.
    'V_m_ex'         : 0 * mV, # Membrane potential, Initial condition
    'E_L_ex'         : 0 * mV, # Leak reversal potential.
    'V_reset_ex'     : 0 * mV, # Reset potential of the membrane.
    'C_m_ex'         : C_m, # Capacity of the membrane
    'V_th_ex'        : theta, # Spike threshold.
    'g_L_ex'         : 1 * nS, # Leak conductance;
    'tau_syn_ex'     : tau_syn, # Time constants of the excitatory synaptic exponential function.
    # Excitatory connectivity params
    'J_ex'           : J_ex, # Synaptic efficacy excitatory
    'C_ex'           : C_ex, # Number of incomming synapses pr neuron to excitatory
    # Inhibitory parameters
    't_ref_in'       : t_ref, # Duration of refractory period.
    'V_m_in'         : 0. * mV, # Membrane potential, Initial condition
    'E_L_in'         : 0. * mV, # Leak reversal potential.
    'V_reset_in'     : 0. * mV, # Reset potential of the membrane.
    'C_m_in'         : C_m, # Capacity of the membrane
    'V_th_in'        : theta, # Spike threshold.
    'g_L_in'         : 1 * nS, # Leak conductance;
    'tau_syn_in'     : tau_syn, # Time constants of the inhibitory synaptic exponential function.
    # Inhibitory connectivity params
    'J_in'           : J_in, # Synaptic efficacy inhibitory
    'C_in'           : C_in, # Number of incomming synapses pr neuron to inhibitory
}
