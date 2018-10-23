import numpy as np
from method import IV
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


def LambertWm1(x):
    import nest
    nest.sli_push(x)
    nest.sli_run('LambertWm1')
    y = nest.sli_pop()
    return y


def PSPnorm(tau_m, C_m, tau_syn):
    a = (tau_m / tau_syn)
    b = (1.0 / tau_syn - 1.0 / tau_m)

    # time of maximum
    t_max = 1.0 / b * (-LambertWm1(-np.exp(-1.0 / a) / a) - 1.0 / a)
    # maximum of PSP for current of unit amplitude
    return (np.exp(1.0) / (tau_syn * C_m * b) *
            ((np.exp(-t_max / tau_m) - np.exp(-t_max / tau_syn)) / b -
             t_max * np.exp(-t_max / tau_syn)))

def rtrim(val, trim):
    if not val.endswith(trim):
        return val
    else:
        return val[:len(val) - len(trim)]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 0:
                return float(obj)
            else:
                return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def poisson_clipped(N, period, low, high):
    poisson = []
    while len(poisson) < N:
        p = - np.log(1 - np.random.uniform(0, 1)) * period
        if p >= low and p <= high:
            poisson.append(p)
    stim_times = [poisson[0]]
    for idx, isi in enumerate(poisson[1:]):
        stim_times.append(stim_times[idx] + isi)
    return np.array(stim_times).round()


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
        self.nodes = nest.Create('iaf_psc_alpha', self.p['N_neurons'])
        keys = ['t_ref', 'V_m', 'E_L', 'V_reset', 'tau_m', 'C_m', 'V_th',
                'tau_syn_ex', 'tau_syn_in']
        nest.SetStatus(self.nodes, [{k: self.p[k] for k in keys}])
        self.nodes_ex = self.nodes[:self.p['N_ex']]
        self.nodes_in = self.nodes[self.p['N_ex']:]

        self.p['J_ex'] = self.p['J'] / PSPnorm(
            self.p['tau_m'], self.p['C_m'], self.p['tau_syn_ex'])
        self.p['J_high_ex'] = self.p['J_high'] / PSPnorm(
            self.p['tau_m'], self.p['C_m'], self.p['tau_syn_ex'])
        self.p['J_low_ex'] = self.p['J_low'] / PSPnorm(
            self.p['tau_m'], self.p['C_m'], self.p['tau_syn_ex'])

        # Go negative later on
        self.p['J_in'] = self.p['g'] * self.p['J'] / PSPnorm(
            self.p['tau_m'], self.p['C_m'], self.p['tau_syn_in'])
        self.p['J_high_in'] = self.p['J_high'] / PSPnorm(
            self.p['tau_m'], self.p['C_m'], self.p['tau_syn_in'])
        self.p['J_low_in'] = self.p['J_low'] / PSPnorm(
            self.p['tau_m'], self.p['C_m'], self.p['tau_syn_in'])

        self.p['C_ex'] = int(self.p['eps'] * self.p['N_ex'])
        self.p['C_in'] = int(self.p['eps'] * self.p['N_in'])

    def set_connections(self):

        def mu(mean):
            return np.log(mean / np.sqrt(1 + self.p['p_var'] / mean**2))

        def sigma(mean):
            return np.sqrt(np.log(1 + self.p['p_var'] / mean**2))

        def connect(nodes, j, c, high, low):
            nest.Connect(
                nodes, self.nodes,
                {'rule': 'fixed_indegree', 'indegree': c},
                {"weight": {'distribution': 'lognormal_clipped',
                            'mu': mu(j),
                            'sigma': sigma(j),
                            'low': low, 'high': high},
                "delay": self.p['delay']})
        self.vprint('Connecting excitatory neurons J = ', self.p['J_ex'], 'C = ',
              self.p['C_ex'])
        connect(self.nodes_ex, self.p['J_ex'], self.p['C_ex'],
                self.p['J_high_ex'], self.p['J_low_ex'])
        self.vprint('Connecting inhibitory neurons J = ', self.p['J_in'], 'C = ',
              self.p['C_in'])
        connect(self.nodes_in, self.p['J_in'], self.p['C_in'],
                self.p['J_high_in'], self.p['J_low_in'])
        # switch inhibitory neurons
        conns = nest.GetConnections(source=self.nodes_in,
                                    synapse_model='static_synapse')
        conn_vals = nest.GetStatus(conns, 'weight')
        nest.SetStatus(conns, [{'weight': -1.*val} for val in conn_vals])

    def set_background(self):
        den = (self.p['J_ex'] * self.p['C_ex'] * np.exp(1) * self.p['tau_m'] *
               self.p['tau_syn_ex'])
        nu_th = (self.p['V_th'] * self.p['C_m']) / den
        nu_ex = self.p['eta'] * nu_th
        self.p['rate_p'] = 1000.0 * nu_ex * self.p['C_ex']
        background = nest.Create("poisson_generator", 1,
                                  params={"rate": self.p['rate_p']})
        nest.Connect(background, self.nodes,
                     syn_spec={"weight": self.p['J_ex'], "delay": self.p['res']})

    def set_channelnoise(self):
        channelnoise = nest.Create("noise_generator", 1,
                              params={"mean": self.p['gauss_mean'],
                                      'std': self.p['gauss_std']})
        nest.Connect(channelnoise, self.nodes)

    def set_pulsepacket(self):
        simtime = (np.sum(np.diff(self.stim_times)) +
                   np.min(np.diff(self.stim_times)) +
                   self.stim_times[0])
        if self.p['pulse_dist'] == 'poisson':
            pulses = poisson_clipped(1000, 5, 5, 10)
            pulses = pulses[pulses < 1500]
        elif self.p['pulse_dist'] == 'regular':
            pulses = np.arange(0, simtime, self.p['pulse_period'])
        elif self.p['pulse_dist'] == 'uniform':
            pulses = np.sort(np.unique(np.random.uniform(1,1500,10000).round()))
        else:
            raise NotImplementedError
        pulsepacket = nest.Create("pulsepacket_generator", 1,
                              params={"pulse_times": pulses,
                                      'sdev': self.p['pulse_std'],
                                      'activity': self.p['pulse_activity']})
        nest.Connect(pulsepacket, self.nodes,
                     syn_spec={"weight": self.p['pulse_J'], "delay": self.p['res']})

    def set_stimulation(self):
        if self.p['stim_dist'] is None:
            self.stim_times = np.linspace(
                self.p['stim_period'], self.p['stim_N'] * self.p['stim_period'],
                self.p['stim_N'])
        elif self.p['stim_dist'] == 'poisson':
            self.stim_times = poisson_clipped(
                N=self.p['stim_N'], period=self.p['stim_period'],
                low=self.p['stim_period'], high=self.p['stim_max_period'])

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
        def connect(nodes, label, N_rec):
            if not N_rec:
                return
            vm = nest.Create("multimeter",
                params = {"interval": sampling_period,
                          "record_from": ['V_m'],
                          "withgid": True, "to_file": False,
                          "label": label, "withtime": True})
            nest.Connect(vm, nodes[:N_rec])
            return vm
        self.vm_ex = connect(self.nodes_ex, "exc_v",
                             self.p.get('N_rec_state_ex'))
        self.vm_in = connect(self.nodes_in, "inh_v",
                             self.p.get('N_rec_state_in'))

    def simulate_trials(self, progress_bar=False):
        progress_bar = progress_bar and PBAR

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
        if progress_bar: pbar = tqdm(total=len(simtimes))
        for s in simtimes:
            if progress_bar: pbar.update(1)
            for stim in stims:
                nest.SetStatus(stim, {'origin': nest.GetKernelStatus()['time']})
            nest.Simulate(s)
        if progress_bar: pbar.close()

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
        if self.p.get('pulse_activity'):
            self.vprint('Setting pulsepacket')
            self.set_pulsepacket()
        self.vprint('Setting spike recording')
        self.set_spike_rec()
        if state:
            self.vprint('Setting state recording')
            self.set_state_rec()
        simtime = (np.sum(np.diff(self.stim_times)) +
                   np.min(np.diff(self.stim_times)) +
                   self.stim_times[0])
        self.vprint('Simulating {} trials,'.format(self.p['stim_N']) +
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
            exc   = nest.GetStatus(self.vm_ex, "events")[0]
            inh   = nest.GetStatus(self.vm_in, "events")[0]
            data.update({'state':  {'in': inh,
                                    'ex': exc}})
        conns = nest.GetConnections(source=self.nodes, target=self.nodes)
        conn_include = ('weight', 'source', 'target')
        conns = list(nest.GetStatus(conns, conn_include))
        data.update({
            'connections': pd.DataFrame(conns, columns=conn_include),
            'epoch': {
                'times': self.stim_times,
                'durations': [self.p['stim_duration']] * self.p['stim_N']},
            'stim_nodes': {
                'ex': self.stim_nodes_ex,
                'in': self.stim_nodes_in},
            'stim_amps': {
                'ex': self.stim_amps_ex,
                'in': self.stim_amps_in},
            'nodes': {
                'ex': list(self.nodes_ex),
                'in': list(self.nodes_in)},
            'params': self.p
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

    def plot_raster(self, save=False):
        import nest.raster_plot
        for pop in ['in', 'ex']:
            try:
                nest.raster_plot.from_device(getattr(self, 'spikes_%s' % pop),
                                             hist=True)
            except nest.NESTError:
                self.vprint('Unable to plot ' + pop)
                continue
            fig = plt.gcf()
            ax = plt.gca()
            ax.set_title(pop)
            if save:
                fig.savefig(os.path.join(self.p['data_path'], pop + '.png'))

    def dump_params_to_json(self):
        fnameout = os.path.join(self.p['data_path'], self.p['fname'] + '.json')
        with open(fnameout, 'w') as outfile:
            json.dump(self.data['params'], outfile,
                      sort_keys=True, indent=4,
                      cls=NumpyEncoder)


if __name__ == '__main__':
    import sys
    import imp
    import os.path as op
    from tools_plot import despine, set_style

    if len(sys.argv) not in [2, 3]:
        raise IOError('Usage: "python simulator.py parameters <load-data>"')

    param_module = sys.argv[1]
    jobname = param_module.replace('.py', '')
    currdir = op.dirname(op.abspath(__file__))
    f, p, d = imp.find_module(jobname, [currdir])
    parameters = imp.load_module(jobname, f, p, d).parameters

    sim = Simulator(parameters, mpi=False, data_path='results', fname=jobname)
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
    plt.xticks([-1, -.5, 0, .5, 1])
    fig.savefig(sim.p['fname'] + '_syn_dist.pdf', bbox_inches='tight')

    # senders = sim.data['spiketrains']['ex']['senders']
    # times = sim.data['spiketrains']['ex']['times']
    # stim_times = sim.data['epoch']['times']
    # sender_ids = np.unique(senders)
    #
    # spiketrains = {sender: times[sender==senders]
    #                for sender in sender_ids}
    # winsize_iv = 4
    # latency_iv = 4 # not used
    # hit_rates = [IV(s1, [], stim_times,  winsize=winsize_iv, latency=latency_iv).hit_rate
    #              for s1 in spiketrains.values()]
    #
    #
    # fig, ax = plt.subplots(1, 2)
    # width = 1
    # bins = np.arange(0, 11, width)
    # hist, bins = np.histogram(sim.data['stim_amps']['ex'].values(), bins=bins)
    # print(bins)
    # print(hist, sum(hist))
    # ax[0].bar(bins[1:], hist, align='edge', width=-width)
    #
    # width = .1
    # bins = np.arange(0, 1+width, width)
    # hist, bins = np.histogram(hit_rates, bins=bins)
    # plt.bar(bins[1:], hist, align='edge', width=-width)
    # fig.savefig(sim.p['fname'] + '_stim_distribution.png')
