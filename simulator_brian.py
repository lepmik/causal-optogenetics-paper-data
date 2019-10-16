import numpy as np
import pathlib
import pandas as pd
import os
import json
import time
import copy

import brian2 as br2
from brian2 import ms, second, Hz, mV, pA, nS, pF

try:
    from tqdm import tqdm
    PBAR = True
except ImportError:
    PBAR = False


def hasattrs(o, *args):
    if any(hasattr(o, a) for a in args):
        return True
    else:
        return False


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 0:
                return float(obj)
            else:
                return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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


class Simulator:
    def __init__(self, parameters, verbose=False, **kwargs):
        parameters = copy.deepcopy(parameters)
        if kwargs:
            parameters.update(kwargs)

        if 'data_path' not in parameters:
            parameters['data_path'] = os.getcwd()
        self.data_path = pathlib.Path(parameters['data_path'])
        parameters['verbose'] = verbose
        self.p = parameters
        self.net = br2.Network()

    def vprint(self, *val):
        if self.p['verbose']:
            print('', *val)
            
    def set_neurons(self):
        # define and create neurons
        keys = [
              'E_L', #Leak reversal potential in mV.
              'C_m', #Capacity of the membrane in pF
              't_ref', #Duration of refractory period in ms.
              'V_th', #Spike threshold in mV.
              'V_reset', #Reset potential of the membrane in mV.
              'E_ex', #Excitatory reversal potential in mV.
              'E_in', #Inhibitory reversal potential in mV.
              'g_L', #Leak conductance in nS;
              'tau_syn_ex', #Rise time of the excitatory synaptic alpha function in ms.
              'tau_syn_in', #Rise time of the inhibitory synaptic alpha function in ms.
        ]
        p_nodes = {k: self.p[k] for k in keys}

        eqs = '''dV_m/dt = (g_L*(E_L-V_m)+Ie+Ii+I)/(C_m) : volt
                 Ie = ge*(E_ex-V_m) : amp
                 Ii = gi*(E_in-V_m) : amp
                 dge/dt = -ge/(tau_syn_ex) : siemens
                 dgi/dt = -gi/(tau_syn_in) : siemens
                 I : amp '''
        nodes = br2.NeuronGroup(
            self.p['N_neurons'],
            model=eqs,
            threshold='V_m > V_th',
            reset='V_m = V_reset',
            refractory=p_nodes['t_ref'],
            namespace=p_nodes,
            method='euler',
        )

        self.p['C_ex'] = int(self.p['eps'] * self.p['N_ex'])
        self.p['C_in'] = int(self.p['eps'] * self.p['N_in'])
        n = self.p['tau_syn_in'] * abs(self.p['E_L'] - self.p['E_in'])
        d = self.p['J_ex'] * self.p['tau_syn_ex'] * abs(
            self.p['E_L'] - self.p['E_ex'])
        self.p['J_in'] = - self.p['g'] * d / n

        self.nodes = nodes
        self.nodes_ex = nodes[:self.p['N_ex']]
        self.nodes_in = nodes[self.p['N_ex']:]

        self.net.add(nodes)
        
    def set_connections(self):
        # define synapses
        syn_ex_ex = br2.Synapses(
            self.nodes_ex,
            self.nodes_ex,
            model='w:siemens',
            on_pre='ge+=w',
            delay=self.p['syn_delay'])
        
        syn_ex_in = br2.Synapses(
            self.nodes_ex,
            self.nodes_in,
            model='w:siemens',
            on_pre='ge+=w',
            delay=self.p['syn_delay'])

        syn_in_ex = br2.Synapses(
            self.nodes_in,
            self.nodes_ex,
            model='w:siemens',
            on_pre='gi+=w',
            delay=self.p['syn_delay'])
        syn_in_in = br2.Synapses(
            self.nodes_in,
            self.nodes_in,
            model='w:siemens',
            on_pre='gi+=w',
            delay=self.p['syn_delay'])
        
        def mu(mean):
            return np.log(mean / np.sqrt(1 + self.p['p_var'] / mean**2))

        def sigma(mean):
            return np.sqrt(np.log(1 + self.p['p_var'] / mean**2))

        self.vprint(
            'Connecting excitatory neurons J = ', self.p['J_ex'], 'C = ',
            self.p['C_ex'],
            'and inhibitory neurons J = ', self.p['J_in'], 'C = ',
            self.p['C_in'])

        # find sources for all excitatory targets
        for j in tqdm(range(self.p['N_ex'])):
            # connect excitatory neurons
            syn_ex_ex.connect(
                i=np.random.choice(
                    range(0, self.p['N_ex']),
                    size=self.p['C_ex'],
                    replace=False),
                j=j,
            )
            # connect inhibitory neurons
            syn_in_ex.connect(
                i=np.random.choice(
                    range(0, self.p['N_in']),
                    size=self.p['C_in'],
                    replace=False),
                j=j,
            )

        # find sources for all inhibitory targets
        for j in tqdm(range(self.p['N_in'])):
            syn_ex_in.connect(
                i=np.random.choice(
                    range(0, self.p['N_ex']),
                    size=self.p['C_ex'],
                    replace=False),
                j=j,
            )
            # connect inhibitory neurons
            syn_in_in.connect(
                i=np.random.choice(
                    range(0, self.p['N_in']),
                    size=self.p['C_in'],
                    replace=False),
                j=j,
            )

        # assigning synaptic weights
        syn_ex_in.w = self.p['J_ex']
        syn_in_ex.w = self.p['J_in']
        syn_in_in.w = self.p['J_in']
        
        self.vprint('Set lognormal weight distribution')
        # define log normal weight distribution
        # get variable without units
        J_ex_val = self.p['J_ex']/nS
        J_ex_j = np.clip(
            np.random.lognormal(
                mean=mu(J_ex_val),
                sigma=sigma(J_ex_val),
                size=len(syn_ex_ex.w)),
            self.p['J_low'],
            self.p['J_high'])
        # reattach unit for conductance
        syn_ex_ex.w = J_ex_j * nS

        syn = [syn_ex_ex, syn_ex_in, syn_in_ex, syn_in_in]
        self.syn = np.array(syn).reshape(2, 2)
        self.net.add(*syn)
        
    def set_background(self):
        self.p['C_p'] = int(self.p['eps_p'] * self.p['N_neurons'])
        self.vprint('Connecting background rate = ',
                    self.p['rate_p'], 'C = ', self.p['C_p'])
        poissInp = br2.PoissonInput(
            self.nodes, 'ge',
            N=self.p['N_ex'],
            rate=self.p['rate_p'] * self.p['C_p'],
            weight=self.p['J_ex'])
        self.poissInp = poissInp
        self.net.add(poissInp)

    def set_spike_rec(self):
        spk_mon = br2.SpikeMonitor(self.nodes)
        self.spk_mon = spk_mon
        self.net.add(spk_mon)

    def set_event_monitor(self):
        evnt_mon_ex = br2.EventMonitor(
            self.nodes_ex,
            event='spike',
            variables=['V_m', 'ge', 'gi'])
        self.evnt_mon_ex = evnt_mon_ex
        self.net.add(evnt_mon_ex)

    def simulate(self, state=False,
                 progress_bar=False):
        self.vprint('Setting neurons')
        self.set_neurons()
        self.vprint('Setting background')
        self.set_background()
        self.vprint('Setting connections')
        self.set_connections()
        self.vprint('Setting spike recording')
        self.set_spike_rec()
        self.vprint('Run simulation')
        self.net.run(
            100*br2.ms,
            report='std::cout << (int)(completed*100.) << "% completed" << std::endl;')
)


if __name__ == '__main__':
    import imp
    import os
    import os.path as op
    
    data_path = '/home/jovyan/work/instrumentalVariable/test_data_sim/'
    param_module = 'params_test_fast_br2.py'
    os.makedirs(data_path, exist_ok=True)
    jobname = param_module.replace('.py', '')
    currdir = op.dirname(op.abspath(__file__))
    f, p, d = imp.find_module(jobname, [currdir])
    parameters = imp.load_module(jobname, f, p, d).parameters
    
    br2.BrianLogger.log_level_info()
    sim = Simulator(
        parameters,
        data_path=data_path,
        jobname=jobname,
        verbose=True)
    sim.vprint('Setting neurons')
    sim.set_neurons()
    sim.vprint('Setting background')
    sim.set_background()
    sim.vprint('Setting connections')
    sim.set_connections()
    sim.vprint('Setting spike recording')
    sim.set_spike_rec()
    sim.set_state_rec()

    br2.run(100*ms)
