import numpy as np
import pathlib
import pandas as pd
import os
import json
import time
import copy
import brian2 as br2
import pdb
import pickle as pkl

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


def create_connection_matrix(p, seed=None):
    """ create_connection_matrix creates a numpy array that
    holds synaptic weights.

    Params
    ---------
    p: dict, Parameter dictionary

    Returns
    ---------
    m: np.ndarray, connection matrix
    """

    if seed:
        np.random.seed(seed)

    def mu(mean):
        return np.log(mean / np.sqrt(1 + p['p_var'] / mean**2))

    def sigma(mean):
        return np.sqrt(np.log(1 + p['p_var'] / mean**2))

    print('Finding excitatory and inhibitory projections')
    print(
        'J_ex = ', p['J_ex'], 'C_ex = ', p['C_ex'],
        'J_in = ', p['J_in'], 'C_in = ', p['C_in'])

    # weights without units 
    J_ex = p['J_ex']/br2.nS
    J_in = p['J_in']/br2.nS
    J_high = p['J_high']/br2.nS
    J_low = p['J_low']/br2.nS

    # initialize connection matrix
    n_nrns = p['N_ex']+p['N_in']
    m = np.zeros((n_nrns, n_nrns))
    
    # find sources for all targets

    for j in tqdm(range(n_nrns)):
        # connections from excitatory neurons

        # avoid ex autapses
        range_ex = np.arange(0, p['N_ex'])        
        range_ex = range_ex[range_ex != j]

        i = np.random.choice(
            range_ex,
            size=p['C_ex'],
            replace=False)

        # lognormal distribution of synaptic weights for EE syns
        if j < p['N_ex']:
            J_ex_j = np.clip(
                np.random.lognormal(
                    mean=mu(J_ex),
                    sigma=sigma(J_ex),
                    size=p['C_ex']),
                J_low,
                J_high)
            m[j, i] = J_ex_j
        else:
            m[j, i] = J_ex
        
        # connections from inhibitory neurons
        # avoid in autapses        
        range_in = np.arange(p['N_ex'], n_nrns)
        range_in = range_in[range_in != j]

        i=np.random.choice(
            range_in,
            size=p['C_in'],
            replace=False)
        m[j, i] = J_in

    # add nano siemens again
    m = m * br2.nS

    return m

def where_repeated_by_val(m):
    """
    where_repeated_by_val takes a matrix,
    applies np.where and repeats the result by the 
    corresponding value  m
    """
    assert np.any(m>0)
    id0, id1 = np.where(m)
    rep = m[m>0].flatten()
    id0 = np.repeat(id0, rep)
    id1 = np.repeat(id1, rep)
    return id0, id1
    
def generate_poisson_spike_trains(
        n_targets, n_inputs,
        rate, start, stop, dt,
        seed=None,
        approximate_by_normal=True):
    """
    generate_poisson_spike_trains generates poisson inputs for each of its targets
    
    Params
    --------
    n_targets : int - number of poisson trains generated
    n_inputs : int - number of inputs per target
    rate : float - firing rate of each input
    start : br2.second - start time
    stop : br2.second - stop time
    dt : br2.second - time resolution
    seed : int - numpy seed
    """
    # check for correct units
    assert rate.dimensions == br2.Hz.dimensions
    assert start.dimensions == stop.dimensions
    assert dt.dimensions == stop.dimensions

    rndm = np.random.RandomState(seed)
    p = rate*dt

    times = np.arange(start, stop, dt)
    n_timesteps = len(times)
    
    use_normal = approximate_by_normal and (n_inputs*p > 5) and n_inputs*(1-p) > 5
    if use_normal:
        loc = n_inputs*p
        scale = np.sqrt(n_inputs*p*(1-p))
        m = rndm.normal(loc, scale, size=(n_targets, n_timesteps))
    else:
        m = rndm.binomial(n_inputs, p, size=(n_targets, n_timesteps))
        
    # convert to idcs, times
    idcs, tbins = where_repeated_by_val(m)
    ts = times[tbins]
    ts = ts*br2.units.get_unit(start.dimensions)
    return idcs, ts
    

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
        br2.defaultclock.dt = self.p['res']

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
                 I : amp
                 record_me : boolean'''
        nodes = br2.NeuronGroup(
            self.p['N_neurons'],
            model=eqs,
            threshold='V_m > V_th',
            reset='V_m = V_reset',
            refractory=p_nodes['t_ref'],
            namespace=p_nodes,
            method='euler',
            # event whenever a spike happens in group
            events={'record_me': 'record_me'}
        )

#        nodes.V_m = np.random.uniform(
#            self.p['V_reset'], self.p['V_th'], size=self.p['N_neurons'])*br2.mV

        self.p['C_ex'] = int(self.p['eps'] * self.p['N_ex'])
        self.p['C_in'] = int(self.p['eps'] * self.p['N_in'])
        n = self.p['tau_syn_in'] * abs(self.p['E_L'] - self.p['E_in'])
        d = self.p['J_ex'] * self.p['tau_syn_ex'] * abs(
            self.p['E_L'] - self.p['E_ex'])
        self.p['J_in'] = self.p['g'] * d / n

        nodes.thresholder['record_me'].when = 'after_synapses'
        nodes.run_on_event('record_me', 'record_me = False')

        self.nodes = nodes
        self.nodes_ex = nodes[:self.p['N_ex']]
        self.nodes_in = nodes[self.p['N_ex']:]

        self.net.add(nodes)

    def set_connections_from_matrix(self, m):
        syn_ex = br2.Synapses(
            self.nodes_ex,
            self.nodes,
            model='w:siemens',
            on_pre='ge+=w',
            delay=self.p['syn_delay'])
        
        syn_in = br2.Synapses(
            self.nodes_in,
            self.nodes,
            model='w:siemens',
            on_pre='gi+=w',
            delay=self.p['syn_delay'])

        N_ex = self.p['N_ex']
        j, i = np.where(m)
        j_ex, i_ex = j[i<N_ex], i[i<N_ex]
        j_in, i_in = j[i>=N_ex], i[i>=N_ex]        

        syn_ex.connect(i=i_ex, j=j_ex)
        syn_ex.w = m[j_ex, i_ex]

        syn_in.connect(i=i_in-N_ex, j=j_in)
        syn_in.w = m[j_in, i_in]

        self.syn_ex = syn_ex
        self.syn_in = syn_in

        self.net.add(syn_ex)
        self.net.add(syn_in)        

    def set_network_event_monitor(self):
        event_mon = br2.EventMonitor(
            self.nodes, 'record_me',
            variables=['V_m', 'ge', 'gi'],
            when='before_resets')
        # for finding any spike in the network
        propagate_spikes = br2.Synapses(self.nodes, self.nodes, on_pre='record_me = True')
        propagate_spikes.connect()

        self.net.add(propagate_spikes)
        self.net.add(event_mon)

        self.event_mon = event_mon

    def del_network_event_monitor(self):
        self.net.remove(self.event_mon)
        del self.event_mon        
        
    def set_background(self):
        self.vprint('Connecting background rate = ',
                    self.p['rate_p'], 'N = ', self.p['N_p'])
        poissInp = br2.PoissonInput(
            self.nodes, 'ge',
            N=self.p['N_p'],
            rate=self.p['rate_p'],
            weight=self.p['J_ex'])
        self.poissInp = poissInp
        self.net.add(poissInp)

    def set_spike_rec(self):
        spk_mon = br2.SpikeMonitor(self.nodes)
        self.spk_mon = spk_mon
        self.net.add(spk_mon)

    def del_spike_rec(self):
        self.net.remove(self.spk_mon)
        del self.spk_mon        

    def set_event_monitor(self):
        evnt_mon_ex = br2.EventMonitor(
            self.nodes_ex,
            event='spike',
            variables=['V_m', 'ge', 'gi'])
        self.evnt_mon_ex = evnt_mon_ex
        self.net.add(evnt_mon_ex)

    def set_state_monitor(self):
        state_mon = br2.StateMonitor(
            self.nodes_ex[:10],
            variables=['V_m', 'ge', 'gi'],
            record=True)
        self.state_mon = state_mon
        self.net.add(state_mon)        

    def set_stimulation(self):
        n_stim = self.p['stim_N_ex']
        nodes_ex_stim = self.nodes_ex[:n_stim]
        self.nodes_ex_stim = nodes_ex_stim

    def simulate_stimulate_and_restore_network_state(self):
        br2.seed(self.p['msd'])
        
        self.vprint('Setting neurons')
        self.set_neurons()

        self.vprint('Setting stimulation')        
        self.set_stimulation()
        
        self.vprint('Setting background')
        self.set_background()
        
        self.vprint('Setting connections')
        m = create_connection_matrix(self.p, self.p['msd'])
        self.set_connections_from_matrix(m)

        self.vprint('Store connections')
        with open(str(self.data_path)+'/'+'m.pkl', 'wb') as f:
            pkl.dump(m, f)

        self.vprint('Set spike monitors')
        # 1 for times without stim
        spk_mon1 = br2.SpikeMonitor(self.nodes)
        self.spk_mon1 = spk_mon1
        self.net.add(spk_mon1)

        # 2 for stimulation periods
        spk_mon2 = br2.SpikeMonitor(self.nodes)
        self.spk_mon2 = spk_mon2
        self.net.add(spk_mon2)
        spk_mon2_t = []
        spk_mon2_i = []        
        spk_mon2.active = False
        
        self.vprint('Simulate')
        runtime = self.p['simtime']        
        pbar = tqdm(total=runtime/br2.ms)

        def stop_for_spikes():
            pdb.set_trace()
            if len(self.nodes_ex_stim.spikes):
                self.net.stop()
        net_op = br2.NetworkOperation(stop_for_spikes)
        self.net.add(net_op)

        init_simtime = self.p['init_simtime']
        while br2.defaultclock.t < runtime:
            # stimulation only after init_simtime 
            if br2.defaultclock.t > init_simtime:
                # store network state before stimulation
                self.net.store()
                # We'll need an explicit seed here, otherwise we'll continue with different
                # random numbers after the restore
                use_seed = br2.randint(br2.iinfo(np.int32).max)
                br2.seed(use_seed)
                # change spike monitors
                spk_mon1.active = False
                spk_mon2.active = True
                # stimulate
                self.nodes_ex_stim.I = self.p['stim_amp_ex']
                self.net.run(self.p['stim_duration'])
                # turn stimuli off, but keep on simulation
                t_left = self.p['simtime_stim'] - self.p['stim_duration']
                self.nodes_ex_stim.I = 0.*br2.pA
                self.net.run(t_left)            
                # go back to initial point and continue run (without stimulation)
                # Note that this would reset monitor 2 as well, so we store its data elsewhere
                spk_mon2_t.extend(spk_mon2.t/ms)
                spk_mon2_i.extend(spk_mon2.i)            
                self.net.restore()
                br2.seed(use_seed)
                spk_mon1.active = True
                spk_mon2.active = False
                
            pbar.update(br2.defaultclock.t/br2.ms)
            self.net.run(runtime - br2.defaultclock.t)

        self.vprint('Store spikes')
        
        t1, i1 = np.array(self.spk_mon1.t), np.array(self.spk_mon1.i)
        with open(str(self.data_path) + '/' + 'spks1.csv', 'w') as f:
                pd.DataFrame({'t': t1, 'id': i1.astype(int)}
                ).to_csv(f, header=False, index=False)
        t2, i2 = np.array(spk_mon2_t), np.array(spk_mon2_i)
        with open(str(self.data_path) + '/' + 'spks2.csv', 'w') as f:
                pd.DataFrame({'t': t2, 'id': i2.astype(int)}
                ).to_csv(f, header=False, index=False)
        self.spk_mon2_t = spk_mon2_t
        self.spk_mon2_i = spk_mon2_i        


    def simulate_and_store_network_state_per_spike(
            self, simtime, simtime_stepsize, state=False,
            progress_bar=False,
            seed=1234):
        from brian2.core.functions import timestep        
        self.vprint('Setting background')
        self.set_background()
        self.vprint('Setting connections')
        m = create_connection_matrix(self.p, seed)
        self.set_connections_from_matrix(m)
        self.vprint('Store connections')
        with open(str(self.data_path)+'/'+'m.pkl', 'wb') as f:
            pkl.dump(m, f)

        # create file to store spikes times and ids
        with open(
                str(self.data_path) + '/' +
                'spks.csv',
                'w') as f:
            pass

        self.vprint('Run simulation and store network state')
        for t_start in tqdm(
                np.arange(0, simtime, simtime_stepsize),
                desc='Store states'):
            # generate monitors for this segment
            self.set_spike_rec()
            self.set_network_event_monitor()
            # run segment
            self.net.run(simtime_stepsize)
            # get spikes and gids
            t_spks_i, gids_i = np.array(self.spk_mon.t), np.array(self.spk_mon.i)
            with open(
                    str(self.data_path) + '/' +
                    'spks.csv',
                    'a') as f:
                pd.DataFrame(
                    {'t': t_spks_i, 'id': gids_i.astype(int)}
                ).to_csv(f, header=False, index=False, mode='a')

            # get network states for each spike
            t_evnts = self.event_mon.t
            V_m = self.event_mon.V_m
            ge, gi = self.event_mon.ge, self.event_mon.gi
            for t in np.unique(t_evnts):
                t_bool = t_evnts == t
                V_m_i = V_m[t_bool]
                ge_i, gi_i = ge[t_bool], gi[t_bool]
                
                t_int = timestep(t, self.p['res'])
                
                with open(
                        str(self.data_path) + '/' +
                        'networkstate_' +
                        str(t_int) + '.pkl',
                        'wb') as f:
                    pkl.dump(
                        {'V_m': V_m_i,
                         'ge': ge_i,
                         'gi': gi_i},
                        f)
            self.del_network_event_monitor()
            self.del_spike_rec()

    def set_spikegeneratorgroup(self):
        spk_gen = br2.SpikeGeneratorGroup(self.p['N_neurons'], [], []*br2.ms)
        self.spk_gen = spk_gen
        self.net.add(spk_gen)
        syn_poiss = br2.Synapses(
            self.spk_gen,
            self.nodes,
            model='w:siemens',
            on_pre='ge+=w',
            delay=0*br2.ms)
        syn_poiss.connect(j='i')

    def run_from_network_state(state_time, simtime, seed):
        # load network state
        t_int = timestep(state_time, self.p['res'])
        with open(
                str(self.data_path) + '/' +
                'networkstate_' +
                str(t_int) + '.pkl',
                'rb') as f:
            pkl_obj = pkl.load(f)
        V_m = plk_obj['V_m']
        ge = plk_obj['ge']
        gi = plk_obj['gi']

        # create fixed poisson input
        poiss_t, poiss_ids = generate_poisson_spike_trains(
            self.p['N_neurons'],
            self.p['N_p'],
            self.p['rate_p'],
            0*br2.ms, simtime, self.p['res'],
            seed=seed)

        # run first simulation
        
        # input
        t = self.net.t
        self.spk_gen.set_spikes(t + poiss_ids, poiss_t)
        
        # set variables
        self.nodes.V_m = V_m
        self.nodes.ge = ge
        self.nodes.gi = gi
        
        self.net.run(simtime)

        # run second simulation
        
        # input
        t = self.net.t
        self.spk_gen.set_spikes(t + poiss_ids, poiss_t)
        
        # set variables
        self.nodes.V_m = V_m
        self.nodes.ge = ge
        self.nodes.gi = gi
        
        self.net.run(simtime)

        spks_t, spks_i = self.spk_rec.t, self.spk_rec.i

        bool0 = spks_t <= t+simtime
        bool1 = ~bool0
        spks_t0 = spks_t[bool0]
        spks_i0 = spks_i[bool0]
        spks_t1 = spks_t[bool1]
        spks_i1 = spks_i[bool1]

        assert np.array_equal(spks_t0, spks_t1)
        
    def simulate(
            self, simtime, state=False,
            progress_bar=False,
            seed=1234):
        self.vprint('Setting neurons')
        
        self.set_neurons()
        self.vprint('Setting background')
        self.set_background()
        self.vprint('Setting connections')
        m = create_connection_matrix(self.p)
        self.set_connections_from_matrix(m)
        self.vprint('Setting spike recording')
        self.set_spike_rec()
        self.vprint('Setting network event recorder')
        self.set_network_event_monitor()
        self.vprint('Run simulation')
        self.net.run(simtime)


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
    sim.set_backgrnound()
    sim.vprint('Setting connections')
    sim.set_connections()
    sim.vprint('Setting spike recording')
    sim.set_spike_rec()
    sim.set_state_rec()

    br2.run(100*ms)


    # def set_connections(self):
    #     # define synapses
    #     syn_ex = br2.Synapses(
    #         self.nodes_ex,
    #         self.nodes,
    #         model='w:siemens',
    #         on_pre='ge+=w',
    #         delay=self.p['syn_delay'])
        
    #     syn_ex_in = br2.Synapses(
    #         self.nodes_ex,
    #         self.nodes_in,
    #         model='w:siemens',
    #         on_pre='ge+=w',
    #         delay=self.p['syn_delay'])

    #     syn_in_ex = br2.Synapses(
    #         self.nodes_in,
    #         self.nodes_ex,
    #         model='w:siemens',
    #         on_pre='gi+=w',
    #         delay=self.p['syn_delay'])
    #     syn_in_in = br2.Synapses(
    #         self.nodes_in,
    #         self.nodes_in,
    #         model='w:siemens',
    #         on_pre='gi+=w',
    #         delay=self.p['syn_delay'])
        
    #     def mu(mean):
    #         return np.log(mean / np.sqrt(1 + self.p['p_var'] / mean**2))

    #     def sigma(mean):
    #         return np.sqrt(np.log(1 + self.p['p_var'] / mean**2))

    #     self.vprint('Connecting excitatory neurons J = ', self.p['J_ex'], 'C = ', self.p['C_ex'])

    #     # find sources for all excitatory targets
    #     for j in tqdm(range(self.p['N_ex'])):
    #         # connect excitatory neurons
    #         syn_ex_ex.connect(
    #             i=np.random.choice(
    #                 range(0, self.p['N_ex']),
    #                 size=self.p['C_ex'],
    #                 replace=False),
    #             j=j,
    #         )
    #         # connect inhibitory neurons
    #         syn_in_ex.connect(
    #             i=np.random.choice(
    #                 range(0, self.p['N_in']),
    #                 size=self.p['C_in'],
    #                 replace=False),
    #             j=j,
    #         )

    #     self.vprint('Connecting inhibitory neurons J = ', self.p['J_in'], 'C = ', self.p['C_in'])
    #     # find sources for all inhibitory targets
    #     for j in tqdm(range(self.p['N_in'])):
    #         syn_ex_in.connect(
    #             i=np.random.choice(
    #                 range(0, self.p['N_ex']),
    #                 size=self.p['C_ex'],
    #                 replace=False),
    #            j=j,
    #         )
    #         # connect inhibitory neurons
    #         syn_in_in.connect(
    #             i=np.random.choice(
    #                 range(0, self.p['N_in']),
    #                 size=self.p['C_in'],
    #                 replace=False),
    #             j=j,
    #         )

    #     # assigning synaptic weights
    #     syn_ex_in.w = self.p['J_ex']
    #     syn_in_ex.w = self.p['J_in']
    #     syn_in_in.w = self.p['J_in']
        
    #     self.vprint('Set lognormal weight distribution')
    #     # define log normal weight distribution
    #     # get variable without units
    #     J_ex_val = self.p['J_ex']/br2.nS
    #     J_ex_j = np.clip(
    #         np.random.lognormal(
    #             mean=mu(J_ex_val),
    #             sigma=sigma(J_ex_val),
    #             size=len(syn_ex_ex.w)),
    #         self.p['J_low'],
    #         self.p['J_high'])
    #     # reattach unit for conductance
    #     syn_ex_ex.w = J_ex_j * br2.nS

    #     syn = [syn_ex_ex, syn_ex_in, syn_in_ex, syn_in_in]
    #     self.syn = syn
    #     self.net.add(*syn)
        
    
