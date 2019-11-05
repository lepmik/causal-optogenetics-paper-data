import numpy as np
from causal_optoconnectics.core import hit_rate
import nest
import pandas as pd
import quantities as pq
import os
import shutil
import matplotlib.pyplot as plt
from pprint import pprint
import json
import time
import copy
try:
    from tqdm import tqdm
    PBAR = True
except ImportError:
    PBAR = False
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False


nest.set_verbosity(100)


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
    def __init__(self, parameters, mpi=False, verbose=False, **kwargs):
        self.mpi = mpi
        if self.mpi:
            assert HAS_MPI
            self.comm = MPI.COMM_WORLD
        parameters = copy.deepcopy(parameters)
        if kwargs:
            parameters.update(kwargs)

        if 'data_path' not in parameters:
            parameters['data_path'] = os.getcwd()
        if 'fname' not in parameters:
            parameters['fname'] = 'simulation_output'
        else:
            if len(os.path.splitext(parameters['fname'])[-1]) != 0:
                raise ValueError('fname should not have extention, it is' +
                                 ' properly added when needed')
        parameters['verbose'] = verbose
        self.p = parameters

    def vprint(self, *val):
        if self.p['verbose']:
            print(*val)

    def set_kernel(self):
        if hasattr(self, '_data'):
            del(self._data)
        num_threads = 1 if self.mpi else self.p['num_threads']
        nest.ResetKernel()
        nest.SetKernelStatus({"overwrite_files": True,
                              "data_path": self.p['data_path'],
                              "data_prefix": "",
                              "print_time": False,
                              "local_num_threads": num_threads})
        msd = self.p['msd']
        # numpy seed
        np.random.seed(msd)
        # nest seed
        N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
        pyrngs = [np.random.RandomState(s) for s in range(msd, msd + N_vp)]
        nest.SetKernelStatus({'grng_seed' : msd + N_vp})
        nest.SetKernelStatus(
            {'rng_seeds' : range(msd + N_vp + 1, msd + 2 * N_vp + 1)})
        # set dt
        nest.SetStatus([0], [{"resolution": self.p['res']}])

    def set_neurons(self):
        self.nodes = nest.Create('iaf_cond_alpha', self.p['N_neurons'])
        keys = [
              'V_m', #Membrane potential in mV
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
        self.nodes_ex = self.nodes[:self.p['N_ex']]
        self.nodes_in = self.nodes[self.p['N_ex']:]
        nest.SetStatus(self.nodes_ex, [{k: self.p[k+'_ex'] for k in keys}])
        nest.SetStatus(self.nodes_in, [{k: self.p[k+'_in'] for k in keys}])
        self.p['C_ex'] = int(self.p['eps'] * self.p['N_ex'])
        self.p['C_in'] = int(self.p['eps'] * self.p['N_in'])
        n = self.p['tau_syn_in_ex'] * abs(self.p['E_L_ex'] - self.p['E_in_ex'])
        d = self.p['J_ex'] * self.p['tau_syn_ex_ex'] * abs(self.p['E_L_ex'] - self.p['E_ex_ex'])
        self.p['J_in'] = - self.p['g'] * d / n

    def set_connections(self):

        def mu(mean):
            return np.log(mean / np.sqrt(1 + self.p['p_var'] / mean**2))

        def sigma(mean):
            return np.sqrt(np.log(1 + self.p['p_var'] / mean**2))

        self.vprint(
            'Connecting excitatory neurons J = ', self.p['J_ex'], 'C = ',
            self.p['C_ex'])
        #------------------------------- LOGNORMAL -----------------------------
        nest.Connect(
            self.nodes_ex, self.nodes,
            {'rule': 'fixed_indegree', 'indegree': self.p['C_ex']},
            {"weight": {'distribution': 'lognormal_clipped',
                        'mu': mu(self.p['J_ex']),
                        'sigma': sigma(self.p['J_ex']),
                        'low': self.p['J_low'], 'high': self.p['J_high']},
            "delay": self.p['delay']})

        #------------------------------- NORMAL -----------------------------
        # nest.Connect(
        #     self.nodes_ex, self.nodes,
        #     {'rule': 'fixed_indegree', 'indegree': self.p['C_ex']},
        #     {"weight": {'distribution': 'normal',
        #                 'mu': self.p['J_ex'],
        #                 'sigma': self.p['p_var']}})

        #------------------------------- FIXED -----------------------------
        # nest.Connect(
        #     self.nodes_ex, self.nodes,
        #     {'rule': 'fixed_indegree', 'indegree': self.p['C_ex']},
        #     {"weight": self.p['J_ex'],
        #     "delay": self.p['delay']})

        self.vprint(
            'Connecting inhibitory neurons J = ', self.p['J_in'], 'C = ',
            self.p['C_in'])
        #------------------------------- FIXED -----------------------------
        nest.Connect(
            self.nodes_in, self.nodes,
            {'rule': 'fixed_indegree', 'indegree': self.p['C_in']},
            {"weight": self.p['J_in'],
            "delay": self.p['delay']})

        #------------------------------- LOGNORMAL -----------------------------
        # nest.Connect(
        #     self.nodes_in, self.nodes,
        #     {'rule': 'fixed_indegree', 'indegree': self.p['C_in']},
        #     {"weight": {'distribution': 'lognormal_clipped',
        #                 'mu': mu(self.p['J_in']),
        #                 'sigma': sigma(self.p['J_in']),
        #                 'low': self.p['J_low'], 'high': self.p['J_high']},
        #     "delay": self.p['delay']})
        # connect(self.nodes_in, self.p['J_in'], self.p['C_in'],
        #         self.p['J_high'], self.p['J_low'])
        # # switch inhibitory neurons
        # conns = nest.GetConnections(source=self.nodes_in,
        #                             synapse_model='static_synapse')
        # conn_vals = nest.GetStatus(conns, 'weight')
        # nest.SetStatus(conns, [{'weight': -1.*val} for val in conn_vals])

    def set_background(self):
        self.p['C_p'] = int(self.p['eps_p'] * self.p['N_neurons'])
        self.vprint('Connecting background rate = ', self.p['rate_p'], 'C = ',
              self.p['C_p'])
        background = nest.Create(
            "poisson_generator", 1,
             params={"rate": self.p['rate_p'] * self.p['C_p']})
        nest.Connect(
            background, self.nodes,
            {'rule': 'all_to_all'},
            {"weight": self.p['J_ex'],  "delay": self.p['delay']})

    def set_channelnoise(self):
        channelnoise = nest.Create("noise_generator", 1,
                              params={"mean": self.p['gauss_mean'],
                                      'std': self.p['gauss_std']})
        nest.Connect(channelnoise, self.nodes)

    def set_stimulation(self):
        if self.p['stim_dist'] is None:
            self.stim_times = np.linspace(
                self.p['stim_rate'],
                self.p['stop_time'],
                self.p['stop_time'] / self.p['stim_rate'])
        elif self.p['stim_dist'] == 'poisson':
            stim_times = generate_stim_times(
                self.p['stim_rate'],
                self.p['stop_time'],
                self.p['stim_isi_min']*1e-3) * 1000
            self.stim_times = stim_times.round(1) # round to simulation resolution

    def set_spike_rec(self):
        spks = nest.Create("spike_detector", 2,
                         params=[{"label": "ex", "to_file": False},
                                 {"label": "in", "to_file": False}])
        # connect using all_to_all: all recorded excitatory neurons to one detector
        self.spikes_ex, self.spikes_in = spks[:1], spks[1:]
        N_rec_ex = self.p.get('N_rec_spike_ex') or self.p['N_ex']
        N_rec_in = self.p.get('N_rec_spike_in') or self.p['N_in']
        nest.Connect(self.nodes_ex[:N_rec_ex], self.spikes_ex)
        nest.Connect(self.nodes_in[:N_rec_in], self.spikes_in)

    def set_state_rec(self):
        sampling_period = self.p.get('sampling_period') or self.p['res']
        N_rec_ex = self.p.get('N_rec_state_ex') or self.p['N_ex']
        N_rec_in = self.p.get('N_rec_state_in') or self.p['N_in']
        def connect(nodes, label, N_rec):
            vm = nest.Create("multimeter",
                params = {"interval": sampling_period,
                          "record_from": ['V_m'],
                          "withgid": True, "to_file": False,
                          "label": label, "withtime": True})
            nest.Connect(vm, nodes[:N_rec])
            return vm
        self.vm_ex = connect(self.nodes_ex, "exc_v", N_rec_ex)
        self.vm_in = connect(self.nodes_in, "inh_v", N_rec_in)

    def simulate_trials(self, progress_bar=False):
        if progress_bar:
            if not callable(progress_bar):
                progress_bar = tqdm
        else:
            progress_bar = lambda x: x

        def intensity(z):
            rho = self.p['r'] * np.sqrt((self.p['n'] / self.p['NA'])**2 - 1)
            return rho**2 / ((self.p['S'] * z + 1) * (z + rho)**2)

        def affected_neurons(z):
            theta = np.arcsin(self.p['NA'] / self.p['n'])
            lcorr = self.p['r'] / np.tan(theta)
            rad = (z + lcorr) * np.tan(theta)
            A = np.pi * rad**2
            dz = z[1] - z[0]
            dV = A * dz
            density = self.p['stim_N_ex'] / sum(dV)
            self.p['density'] = density
            N = dV * density
            return N

        def hill(I):
            In = I**self.p['n_hill']
            return self.p['Imax'] * In / (self.p['K']**self.p['n_hill'] + In) # peak amplitude of the current response

        # Simulate one period without stimulation
        nest.Simulate(self.stim_times[0])

        # Set dc stimulation
        stims = []
        z = np.linspace(0, self.p['depth'], self.p['N_pos'])
        N_slice = affected_neurons(z).astype(int)

        I = intensity(z)
        A = hill(self.p['I0'] * I)
        A = A / A.max()
        idx = 0
        self.stim_amps_ex = {}
        self.stim_amps_in = {}
        self.stim_nodes_ex = []
        self.stim_nodes_in = []
        for i, N_stim in enumerate(N_slice):
            nodes = self.nodes_ex[idx:idx + N_stim]
            self.stim_nodes_ex.extend(list(nodes))
            idx += N_stim
            amp = A[i] * self.p['stim_amp_ex']
            self.stim_amps_ex.update({n: amp for n in nodes})
            stim = nest.Create(
                "dc_generator",
                params={'amplitude': amp,
                        'start': 0.,
                        'stop': self.p['stim_duration']})
            nest.Connect(stim, nodes)
            stims.append(stim)

        assert all(np.in1d(self.stim_nodes_ex, self.nodes_ex))
        assert all(np.in1d(self.stim_nodes_in, self.nodes_in))
        # Run multiple trials
        simtimes = np.concatenate((np.diff(self.stim_times),
                                  [np.min(np.diff(self.stim_times))]))

        for s in progress_bar(simtimes):
            for stim in stims:
                nest.SetStatus(stim, {'origin': nest.GetKernelStatus()['time']})
            nest.Simulate(s)

    def simulate(self, save=False, raster=False, state=False,
                 progress_bar=False):
        self.vprint('Setting kernel')
        self.set_kernel()
        self.vprint('Setting neurons')
        self.set_neurons()
        self.vprint('Setting background')
        self.set_background()
        self.vprint('Setting connections')
        self.set_connections()
        self.vprint('Setting stimulation')
        self.set_stimulation()
        if self.p.get('gauss_mean') or self.p.get('gauss_std'):
            self.vprint('Setting channelnoise')
            self.set_channelnoise()
        self.vprint('Setting spike recording')
        self.set_spike_rec()
        if state:
            self.vprint('Setting state recording')
            self.set_state_rec()
        simtime = (np.sum(np.diff(self.stim_times)) +
                   np.min(np.diff(self.stim_times)) +
                   self.stim_times[0])
        self.vprint('Simulating {} trials,'.format(len(self.stim_times)) +
              ' total {} ms'.format(simtime))
        tstart = time.time()
        self.simulate_trials(progress_bar=progress_bar)
        self.vprint('Simulation lapsed {:.2f} s'.format(time.time() - tstart))
        if save:
            self.vprint('Saving data')
            self.save_data(overwrite=True)
        if raster:
            self.vprint('Plotting rasters')
            self.plot_raster(save=save)

    def generate_data_dict(self):
        status = nest.GetKernelStatus()
        include = ['time']
        status = {k:v for k,v in status.items() if k in include}
        self.p['status'] = status
        data = {}
        if hasattrs(self, 'spikes_ex', 'spikes_in'):
            self.vprint('Gathering spikes')
            spiketrains = {}
            for spks, tag in zip([self.spikes_ex, self.spikes_in], ['ex', 'in']):
                spk_status = nest.GetStatus(spks, 'events')[0]
                nsenders = len(set(spk_status['senders']))
                nspikes = len(spk_status['times'])
                if nspikes == 0:
                    self.vprint('Warning: no spikes in {} recorded.'.format(tag))
                rate = 0 if not nsenders else nspikes / (status['time'] / 1000.0) / nsenders
                self.p['rate_'+tag] = rate
                spiketrains[tag] = spk_status
            data['spiketrains'] = spiketrains
        if hasattrs(self, 'vm_ex', 'vm_in'):
            self.vprint('Gathering states')
            exc   = nest.GetStatus(self.vm_ex, "events")[0]
            inh   = nest.GetStatus(self.vm_in, "events")[0]
            data.update({'state':  {'in': inh,
                                    'ex': exc}})
            
        self.vprint('Gathering connections')
        conns = nest.GetConnections(source=self.nodes, target=self.nodes)
        conn_include = ('weight', 'source', 'target')
        conns = list(nest.GetStatus(conns, conn_include))
        data.update({
            'connections': pd.DataFrame(conns, columns=conn_include),
            'nodes': {
                'ex': list(self.nodes_ex),
                'in': list(self.nodes_in)},
        })
        
        data.update({
            'params': self.p
        })
        
        if hasattr(self, 'stim_times'):
            self.vprint('Gathering stim times')
            data.update({
                'epoch': {
                    'times': self.stim_times,
                    'durations': [self.p['stim_duration']] * len(self.stim_times)},
                'stim_nodes': {
                    'ex': self.stim_nodes_ex,
                    'in': self.stim_nodes_in},
                'stim_amps': {
                    'ex': self.stim_amps_ex,
                    'in': self.stim_amps_in}
            })
        return data

    @property
    def data(self):
        if not hasattr(self, '_data'):
            if not hasattr(self, 'nodes'):
                self.vprint('Cannot find data in memory, loading from file...')
                fnameout = os.path.join(self.p['data_path'], self.p['fname'] + '.npz')
                data = np.load(fnameout)['data'][()]
                if not self.p == data['params']:
                    self.vprint('WARNING: Data parameters and self parameters are ' +
                          'not equal. Use "data["params"]"')
            else:
                data = self.generate_data_dict()
            self._data = data
        return self._data

    def save_data(self, overwrite=False):
        fnameout = os.path.join(self.p['data_path'], self.p['fname'] + '.npz')
        if self.mpi:
            fnameout = fnameout.replace('.npz', '_{}.npz'.format(self.comm.Get_rank()))
        if not os.path.exists(self.p['data_path']):
            os.mkdir(self.p['data_path'])
        if not overwrite and os.path.exists(fnameout):
            raise IOError('File {} exist, use overwrite=True.'.format(fnameout))
        np.savez(fnameout, data=self.data)

    # def _raster(self, pop):
    #     spk_status = nest.GetStatus(getattr(self, 'spikes_' + pop), 'events')[0]
    #     senders = spk_status['senders']
    #     times = spk_status['times']
    #     spiketrains = {s: times[senders==s] for s in np.unique(senders)[:500]}
    #     senders_, times_ = [], []
    #     for s, t in spiketrains.items():
    #         senders_.extend([s]*len(t))
    #         times_.extend(t)
    #     plt.scatter(times_, senders_)

    def plot_raster(self, save=False, xlim=None):
        import nest.raster_plot
        for pop in ['in', 'ex']:
            # fig, ax = plt.subplots()
            try:
                nest.raster_plot.from_device(getattr(self, 'spikes_%s' % pop),
                                             hist=True)
            except nest.NESTError:
                self.vprint('Unable to plot ' + pop)
                continue
            fig = plt.gcf()
            ax = plt.gca()
            if xlim is not None:
                ax.set_xlim(xlim)
            # self._raster(pop)
            ax.set_title(pop)
            if save:
                fig.savefig(os.path.join(self.p['data_path'], pop + '.png'))

    def dump_params_to_json(self):
        fnameout = os.path.join(self.p['data_path'], self.p['fname'] + '.json')
        with open(fnameout, 'w') as outfile:
            json.dump(self.data['params'], outfile,
                      sort_keys=True, indent=4,
                      cls=NumpyEncoder)

            
def validate_instrument(x, y, stim_times):
    stim_times = stim_times.astype(float)

    src_x = np.searchsorted(x, stim_times, side='right')
    src_y = np.searchsorted(y, stim_times, side='right')

    remove_idxs, = np.where(
        (src_x==len(x)) | (src_x==0) | (src_y==len(y)) | (src_y==0))
    src_x = np.delete(src_x, remove_idxs)
    src_y = np.delete(src_y, remove_idxs)
    stim_times = np.delete(stim_times, remove_idxs)

    X = x[src_x-1] - stim_times
    Y = y[src_y-1] - stim_times
    
#     cov = np.cov(X, Y)[0,1]
    cor = np.corrcoef(X, Y)[0,1]# * X.std() / Y.std()

    return cor


if __name__ == '__main__':
    import sys
    import imp
    import os.path as op
    from tools_plot import despine, set_style
    from tools_analysis import corrcoef, coef_var

    if len(sys.argv) not in [2, 3]:
        raise IOError('Usage: "python simulator.py parameters <load-data>"')

    param_module = sys.argv[1]
    jobname = param_module.replace('.py', '')
    currdir = op.dirname(op.abspath(__file__))
    f, p, d = imp.find_module(jobname, [currdir])
    parameters = imp.load_module(jobname, f, p, d).parameters

    sim = Simulator(parameters, mpi=False, data_path='results', fname=jobname, verbose=True)
    if not 'load-data' in sys.argv:
        sim.simulate(save=True, progress_bar=True, state=False)
    sim.dump_params_to_json()
    print('Rate ex:', sim.data['params']['rate_ex'])
    print('Rate in:', sim.data['params']['rate_in'])

    set_style('article', w=.4)
    fig, ax = plt.subplots(1, 1)
    conn = sim.data['connections']
    conn.weight.hist(bins=100)
    plt.grid(False)
    despine(ax=fig.gca())
    plt.xlabel('Synaptic weight [pA]')
    # plt.xticks([-1, -.5, 0, .5, 1])
    fig.savefig(sim.p['fname'] + '_syn_dist.pdf', bbox_inches='tight')

    senders = sim.data['spiketrains']['ex']['senders']
    times = sim.data['spiketrains']['ex']['times']
    stim_times = sim.data['epoch']['times']
    sender_ids = np.unique(senders)
    t_stop = sim.p['status']['time']

    spiketrains = {sender: times[sender==senders]
                   for sender in sender_ids}
    from causal_optoconnectics.core import hit_rate
    mu = 2
    sigma = 2 # not used
    hit_rates = [hit_rate(s1, stim_times,  mu=mu, sigma=sigma)
                 for k,s1 in spiketrains.items() if k in sim.data['stim_nodes']['ex']]


    fig, ax = plt.subplots(1, 2)

    amps = np.array(list(sim.data['stim_amps']['ex'].values()))
    ax[0].hist(amps, bins=200)
    ax[0].set_xlabel('Stim amps')

    width = .1
    bins = np.arange(0, 1+width, width)
    hist, bins = np.histogram(hit_rates, bins=bins)
    ax[1].bar(bins[1:], hist, align='edge', width=-width)
    ax[1].set_xlabel('Hit rate')

    fig.savefig(sim.p['fname'] + '_stim_distribution.png')
    
    fig, ax = plt.subplots(1, 2)
    
    binsize_corr = 2.
    spiketrains_list = np.ravel(list(spiketrains.values()))
    spiketrains_sampled = np.random.choice(spiketrains_list, size=400)
    cc = corrcoef(
        spiketrains_sampled, t_stop, binsize=binsize_corr)

    cc = cc[np.triu_indices(len(cc), k=1)]
    cc = cc[~np.isnan(cc)]
    ax[0].hist(cc, bins=100);
    ax[0].set_title('CC = {}'.format(np.mean(cc)))


    cv = np.array(coef_var(spiketrains_sampled))
    cv = cv[~np.isnan(cv)]

    ax[1].hist(cv, bins=100);
    ax[1].set_title('CV = {}'.format(np.mean(cv)))
    
    print('CC = {}, CV = {}'.format(np.mean(cc), np.mean(cv)))
    fig.savefig(sim.p['fname'] + '_network_state.png')
    
    fig = plt.figure()
    rand_nodes = np.random.choice([s for s in sim.data['stim_nodes']['ex'] if s in spiketrains], size=400)
    prestim_cor = [
        validate_instrument(spiketrains[s1], spiketrains[s2], stim_times)
        for s1, s2 in zip(rand_nodes, rand_nodes[1:])]
    plt.hist(prestim_cor)
    fig.savefig(sim.p['fname'] + '_prestim_cor.png')
