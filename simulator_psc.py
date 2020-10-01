import numpy as np
import pathlib
import nest
import nest.topology as tp
import pandas as pd
import feather
import os
import json
import time
import copy
import itertools
try:
    from tqdm import tqdm
    PBAR = True
except ImportError:
    PBAR = False

nest.set_verbosity(100)


def prime_factors(n):
    result = []
    for i in itertools.chain([2], itertools.count(3, 2)):
        if n <= 1:
            break
        while n % i == 0:
            n //= i
            result.append(i)
    return result


def closest_product(n):
    primes = np.array(prime_factors(n))
    products = np.array([(np.prod(primes[:i]), np.prod(primes[i:]))
                         for i in range(len(primes))])
    idx = np.argmin(abs(np.diff(products, axis=1)))
    return products[idx]


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


class Simulator:
    def __init__(self, parameters, verbose=False, **kwargs):
        parameters = copy.deepcopy(parameters)
        if kwargs:
            parameters.update(kwargs)
        save_to_file = parameters.get('save_to_file')
        self.save_to_file = save_to_file if save_to_file is not None else True
        if 'data_path' not in parameters:
            parameters['data_path'] = os.getcwd()
        self.data_path = pathlib.Path(parameters['data_path']).absolute()
        self.data_path.mkdir(exist_ok=True)
        parameters['verbose'] = verbose
        self.p = parameters

    def vprint(self, *val):
        if self.p['verbose']:
            print('Rank {}'.format(nest.Rank()), *val)

    def set_kernel(self):
        if hasattr(self, '_data'):
            del(self._data)
        nest.ResetKernel()
        nest.SetKernelStatus({"overwrite_files": True,
                              "data_path": self.p['data_path'],
                              "local_num_threads": self.p['num_threads'],
                              "data_prefix": "",
                              "print_time": False})
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

    def set_nodes(self):
        keys = [
            'V_m', #Membrane potential in mV
            'E_L', #Leak reversal potential in mV.
            'C_m', #Capacity of the membrane in pF
            't_ref', #Duration of refractory period in ms.
            'V_th', #Spike threshold in mV.
            'V_reset', #Reset potential of the membrane in mV.
            'tau_m', #Leak conductance in nS;
            'tau_syn_ex', #Rise time of the excitatory synaptic alpha function in ms.
            'tau_syn_in', #Rise time of the inhibitory synaptic alpha function in ms.'w_'
        ]
        nest.CopyModel(
            'iaf_psc_alpha', "nodes_ex",
            {k: self.p[k] for k in keys})
        nest.CopyModel(
            'iaf_psc_alpha', "nodes_in",
            {k: self.p[k] for k in keys})
        self.p['J_in'] = - self.p['g'] * self.p['J_ex']

    def create_nodes(self):
        if 'nodes_ex' not in nest.Models():
            self.set_nodes()
        self.nodes_ex = nest.Create('nodes_ex', self.p['N_ex'])
        self.nodes_in = nest.Create('nodes_in', self.p['N_in'])
        self.nodes = self.nodes_ex + self.nodes_in

    def set_connections_from_file(self, filename):
        self.connections = feather.read_dataframe(filename)
        nest.Connect(
            self.connections.source.values, self.connections.target.values,
            conn_spec={'rule': 'one_to_one'},
             syn_spec={
                'weight': self.connections.weight.values,
                'delay': self.p['delay']})
        if self.save_to_file:
            self.connections.to_feather(
                self.data_path / 'connections_{}.feather'.format(nest.Rank()))

    def set_connections_topology(self):
        assert self.p['topology_dim'] in [1, 2]
        def generate_pos(pop):
            if self.p['position'] == 'random':
                pos = np.random.uniform(
                    - self.p['extent'] / 2., self.p['extent'] / 2.,
                    (2, self.p['N_' + pop]))
                if self.p['topology_dim'] == 1:
                    pos[0, :] = 0
                layer = {
                    'extent': [self.p['extent'], self.p['extent']],
                    'positions': pos.T.tolist(),
                    'elements': 'nodes_' + pop,
                    'edge_wrap': True}
            elif self.p['position'] == 'grid':
                if self.p['topology_dim'] == 2:
                    N_x, N_y = closest_product(self.p['N_'+pop])
                else:
                    N_x, N_y = self.p['N_'+pop], 1
                assert N_x * N_y == self.p['N_'+pop]
                layer = {
                    'extent': [self.p['extent'], self.p['extent']],
                    'rows' : N_x,
                    'columns' : N_y,
                    'elements': 'nodes_' + pop,
                    'edge_wrap': True}
                pos = None
            return layer, pos

        layer_ex, self.pos_ex = generate_pos('ex')
        layer_in, self.pos_in = generate_pos('in')
        self.layer_ex = tp.CreateLayer(layer_ex)
        self.layer_in = tp.CreateLayer(layer_in)

        def connect_layer(source, target, weight):
            if weight == 0: return
            syn_spec = {
                'connection_type': 'divergent',
                'mask': self.p['mask_{}_{}'.format(source, target)],
                'kernel': 1.,
                'weights': weight,
                'sources': {'model': 'nodes_{}'.format(source)},
                'targets': {'model': 'nodes_{}'.format(target)},
                'allow_autapses': False}

            tp.ConnectLayers(
                getattr(self, 'layer_{}'.format(source)),
                getattr(self, 'layer_{}'.format(target)),
                syn_spec)

        self.vprint('Connecting ex to in with', self.p['J_ex'])
        center_doghnut = (
            self.p['mask_ex_in']['doughnut']['outer_radius'] -
            self.p['mask_ex_in']['doughnut']['inner_radius']
        )
        center_doghnut = center_doghnut / 2 + self.p['mask_ex_in']['anchor'][1]
        connect_layer(
        'ex', 'in',
        {"gaussian": {
            "p_center":  self.p['J_ex'],
            'mean': center_doghnut,
            "sigma": .5}})
        self.vprint('Connecting in to ex with', self.p['J_in'])
        connect_layer('in', 'ex', self.p['J_in'])

        self.nodes_ex = nest.GetNodes(self.layer_ex)[0]
        self.nodes_in = nest.GetNodes(self.layer_in)[0]
        self.nodes = self.nodes_ex + self.nodes_in

        if self.pos_ex is None:
            self.pos_ex = np.array(tp.GetPosition(self.nodes_ex)).T
        if self.pos_in is None:
            self.pos_in = np.array(tp.GetPosition(self.nodes_in)).T
        self.positions = pd.DataFrame(
            np.vstack([self.nodes, np.hstack([self.pos_ex, self.pos_in])]).T,
            columns=['nodes', 'x', 'y']
        )

        self.vprint('Gathering connections')
        conns = nest.GetConnections(source=self.nodes, target=self.nodes)
        conn_include = ('weight', 'source', 'target')
        conns = list(nest.GetStatus(conns, conn_include))
        self.connections = pd.DataFrame(conns, columns=conn_include)

        if self.save_to_file:
            self.vprint('Saving positions')
            self.positions.to_feather(self.data_path /
                             'positions_{}.feather'.format(nest.Rank()))
            self.vprint('Saving connections')
            self.connections.to_feather(self.data_path /
                             'connections_{}.feather'.format(nest.Rank()))

    def set_connections(self):
        self.p['C_ex'] = int(self.p['eps'] * self.p['N_ex'])
        self.p['C_in'] = int(self.p['eps'] * self.p['N_in'])

        def mu(mean):
            return np.log(mean / np.sqrt(1 + self.p['p_var'] / mean**2))

        def sigma(mean):
            return np.sqrt(np.log(1 + self.p['p_var'] / mean**2))

        self.vprint(
            'Connecting excitatory neurons J = ', self.p['J_ex'], 'C = ',
            self.p['C_ex'])

        nest.Connect(
            self.nodes_ex, self.nodes,
            {'rule': 'fixed_indegree', 'indegree': self.p['C_ex']},
            {"weight": {'distribution': 'lognormal_clipped',
                        'mu': mu(self.p['J_ex']),
                        'sigma': sigma(self.p['J_ex']),
                        'low': self.p['J_low'], 'high': self.p['J_high']},
            "delay": self.p['delay']})


        self.vprint(
            'Connecting inhibitory neurons J = ', self.p['J_in'], 'C = ',
            self.p['C_in'])


        nest.Connect(
            self.nodes_in, self.nodes,
            {'rule': 'fixed_indegree', 'indegree': self.p['C_in']},
            {"weight": self.p['J_in'],
            "delay": self.p['delay']})

        self.vprint('Gathering connections')
        tstart = time.time()
        conns = nest.GetConnections(source=self.nodes, target=self.nodes)
        conn_include = ('weight', 'source', 'target')
        conns = list(nest.GetStatus(conns, conn_include))
        self.connections = pd.DataFrame(conns, columns=conn_include)

        if self.save_to_file:
            self.vprint('Saving connections')
            self.connections.to_feather(self.data_path /
                             'connections_{}.feather'.format(nest.Rank()))
            self.vprint('Time lapsed {:.2f} s'.format(time.time() - tstart))

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
            {"weight": self.p['J_p'],  "delay": self.p['delay']})

    def set_background_ex(self):
        self.vprint('Connecting background rate = ', self.p['rate_p'])
        background = nest.Create(
            "poisson_generator", 1,
             params={"rate": self.p['rate_p']})
        nest.Connect(
            background, self.nodes_ex,
            {'rule': 'all_to_all'},
            {"weight": self.p['J_p'],  "delay": self.p['delay']})

    def set_spike_rec(self):
        spks = nest.Create("spike_detector", 2,
                         params=[{"label": "ex", "to_file": self.save_to_file},
                                 {"label": "in", "to_file": self.save_to_file}])
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
                          "withgid": True, "to_file": self.save_to_file,
                          "label": label, "withtime": True})
            nest.Connect(vm, nodes[:N_rec])
            return vm
        self.vm_ex = connect(self.nodes_ex, "exc_v", N_rec_ex)
        self.vm_in = connect(self.nodes_in, "inh_v", N_rec_in)

    def set_ac_input(self):
        # Set confounding drive
        self.ac = nest.Create(
            "ac_generator", 1, params={
                "offset": self.p['ac_offset'],
                'amplitude': self.p['ac_amp'],
                'frequency': self.p['ac_freq']
            }
        )
        nest.SetStatus(self.ac, {'origin': self.p['ac_delay']})
        nest.Connect(self.ac, self.nodes_ex)

    def set_ac_poisson_input(self):
        # Set confounding drive
        # self.ac = nest.Create(
        #     'sinusoidal_poisson_generator',
        #     params={
        #         'rate': 0.0,
        #         'amplitude': self.p['ac_amp'],
        #         'frequency': self.p['ac_freq'],
        #         'phase': 0.0,
        #         'individual_spike_trains': True})
        # nest.SetStatus(self.ac, {'origin': self.p['ac_delay']})
        # nest.Connect(
        #     self.ac, self.nodes_ex,
        #     {'rule': 'all_to_all'},
        #     {"weight": self.p['ac_J']})
        approx_simtime = (self.p['init_simtime'] +
            self.p['stim_trials'] * self.p['stim_isi_min'])
        rate_times = np.arange(
            self.p['ac_delay'], approx_simtime, self.p['ac_period'])
        rate_times = rate_times + np.random.uniform(-20, 20, len(rate_times))
        rate_times = rate_times.round(1)
        rate_values = np.zeros(len(rate_times))
        rate_values[::-2] = self.p['ac_rate']
        rate_values[-1] = 0 # zero when simtime > approx_simtime
        self.ac = nest.Create(
            'inhomogeneous_poisson_generator',
            params={
                'rate_times': rate_times,
                'rate_values': rate_values})
        nest.Connect(
            self.ac, self.nodes_ex,
            {'rule': 'all_to_all'},
            {"weight": self.p['ac_J']})

    def set_stim_dependent_poisson_input(self):
        if not hasattr(self, 'stim_times'):
            self.compute_stim_times()
        rate_times = np.random.permutation(
            self.stim_times)[:int(len(self.stim_times) / 2)]
        rate_times = np.sort(np.concatenate((rate_times - 20, rate_times + 20)))
        rate_values = np.zeros(len(rate_times))
        rate_values[::2] = self.p['ac_rate']
        self.ac = nest.Create(
            'inhomogeneous_poisson_generator',
            params={
                'rate_times': rate_times,
                'rate_values': rate_values})
        nest.Connect(
            self.ac, self.nodes_ex,
            {'rule': 'all_to_all'},
            {"weight": self.p['ac_J']})

    def compute_stim_amps(self):
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

        # Set dc stimulation
        z = np.linspace(0, self.p['depth'], self.p['N_pos'])
        N_slice = affected_neurons(z).astype(int)
        I = intensity(z)
        A = hill(self.p['I0'] * I)
        A = A / A.max()
        idx = 0
        self.stim_amps = {}
        nodes_permute = np.random.permutation(self.nodes_ex)
        for i, N_stim in enumerate(N_slice):
            nodes = nodes_permute[idx:idx + N_stim]
            idx += N_stim
            amp = A[i] * self.p['stim_amp_ex']
            self.stim_amps.update({n: amp for n in nodes})

    def compute_stim_amps_constant(self):
        nodes_permute = np.random.permutation(self.nodes_ex)
        self.stim_amps = {
            n: self.p['stim_amp_ex']
            for n in nodes_permute[:self.p['stim_N_ex']]
        }

    def assign_stim_amps(self, stim_amps):
        self.stim_generators = []
        for n, amp in stim_amps.items():
            stim = nest.Create(
                "dc_generator",
                params={'amplitude': amp,
                        'start': 0.,
                        'stop': self.p['stim_duration']})
            nest.Connect(stim, (n,))
            self.stim_generators.append(stim)

    def compute_stim_times(self):
        self.stim_times = [self.p['init_simtime']]
        for i in range(self.p['stim_trials']):
            self.stim_times.append(
                self.stim_times[i] +
                self.p['stim_isi_min'] +
                round(np.random.uniform(-20, 20), 1)
            )

    def simulate_trials(self, progress_bar=False):
        if not hasattr(self, 'stim_times'):
            self.compute_stim_times()

        if progress_bar:
            if not callable(progress_bar):
                progress_bar = tqdm
        else:
            progress_bar = lambda x: x

        nest.Simulate(self.p['init_simtime'])

        self.assign_stim_amps(self.stim_amps)

        for i in progress_bar(range(len(self.stim_times) - 1)):
            stim_time = self.stim_times[i]
            for stim in self.stim_generators:
                nest.SetStatus(stim, {'origin': stim_time})
            nest.Simulate(self.stim_times[i+1] - stim_time)
        for stim in self.stim_generators:
            nest.SetStatus(stim, {'origin': self.stim_times[-1]})
        nest.Simulate(self.p['stim_isi_min'])

    def simulate_trials_branch(self, progress_bar=False):
        if progress_bar:
            if not callable(progress_bar):
                progress_bar = tqdm
        else:
            progress_bar = lambda x: x

        if not hasattr(self, 'stim_times'):
            self.compute_stim_times()

        nest.Simulate(self.p['init_simtime'])

        self.assign_stim_amps(self.stim_amps)

        def get_status(nodes):
            status = nest.GetStatus(nodes)
            status = [{'V_m': s['V_m']} for s in status]
            return status

        status = get_status(self.nodes)
        self.stim_times = [] # TODO fix
        for i in progress_bar(range(self.p['stim_trials'])):
            if i > 0:
                nest.SetStatus(self.nodes, status)
                nest.Simulate(self.p['stim_isi_min'])
                status = get_status(self.nodes)
            stim_time = nest.GetKernelStatus()['time']
            self.stim_times.append(stim_time)
            for stim in self.stim_generators:
                nest.SetStatus(stim, {'origin': stim_time})
            nest.Simulate(self.p['post_stimtime'])

    def simulate(self, progress_bar=False, connfile=None, stim_amps=None, branch_out=False):
        for setup in self.p['setup']:
            self.vprint('Setting', setup)
            if setup == 'set_connections_from_file':
                assert connfile is not None
                getattr(self, setup)(connfile)
            elif setup in ['simulate_trials', 'simulate_trials_branch']:
                getattr(self, setup)(progress_bar=progress_bar)
            elif setup in ['compute_stim_amps', 'compute_stim_amps_constant']:
                if stim_amps is None:
                    getattr(self, setup)()
                else:
                    self.stim_amps = stim_amps
            else:
                getattr(self, setup)()
        # self.set_kernel()
        # self.vprint('Setting neurons')
        # self.set_nodes()
        # self.create_nodes()
        # self.vprint('Setting background')
        # self.set_background()
        # self.vprint('Setting connections')
        # if connfile is None:
        #     self.set_connections()
        # else:
        #     self.set_connections_from_file(connfile)
        # self.vprint('Setting AC input')
        # # self.set_ac_input()
        # self.set_ac_poisson_input()
        # self.vprint('Setting spike recording')
        # self.set_spike_rec()
        # if state:
        #     self.vprint('Setting state recording')
        #     self.set_state_rec()
        # tstart = time.time()
        # if branch_out:
        #     self.simulate_trials_branch(progress_bar=progress_bar, stim_amps=stim_amps)
        # else:
        #     self.simulate_trials(progress_bar=progress_bar, stim_amps=stim_amps)
        # self.vprint('Simulation lapsed {:.2f} s'.format(time.time() - tstart))
        if self.save_to_file:
            self.vprint('Saving parameters')
            self.dump_params_to_json(nest.Rank())
        self.vprint('Gathering stim times')
        self.stim_data = {
            'times': np.array(self.stim_times),
            'durations': [self.p['stim_duration']] * len(self.stim_times),
            'stim_amps': self.stim_amps
        }
        if self.save_to_file:
            self.vprint('Saving stim times')
            np.savez(
                self.data_path / 'stimulation_data_{}.npz'.format(nest.Rank()),
                data=self.stim_data)

    def dump_params_to_json(self, suffix=''):
        status = nest.GetKernelStatus()
        include = ['time']
        status = {k:v for k,v in status.items() if k in include}
        self.p['status'] = status

        fnameout = str(self.data_path / 'params_{}.json'.format(suffix))
        with open(fnameout, 'w') as outfile:
            json.dump(self.p, outfile,
                      sort_keys=True, indent=4,
                      cls=NumpyEncoder)

    def get_spikes(self, pop='all'):
        if pop == 'ex':
            return pd.DataFrame(nest.GetStatus(self.spikes_ex, 'events')[0])
        if pop == 'in':
            return pd.DataFrame(nest.GetStatus(self.spikes_in, 'events')[0])
        if pop == 'all':
            ex_ = pd.DataFrame(nest.GetStatus(self.spikes_ex, 'events')[0])
            in_ = pd.DataFrame(nest.GetStatus(self.spikes_in, 'events')[0])
            return ex_, in_


    def plot(self):
        import nest.raster_plot
        nest.raster_plot.from_device(self.spikes_ex, hist=True)
        plt.gcf().savefig(str(self.data_path / 'raster_ex.png'))
        nest.raster_plot.from_device(self.spikes_in, hist=True)
        plt.gcf().savefig(str(self.data_path / 'raster_in.png'))


if __name__ == '__main__':
    import sys
    import imp
    import os.path as op

    connfile = None
    if len(sys.argv) == 3:
        data_path, param_module = sys.argv[1:]
    elif len(sys.argv) == 4:
        data_path, param_module, connfile = sys.argv[1:]
    else:
        raise IOError('Usage: "python simulator.py data_path parameters [connfile]')

    jobname = param_module.replace('.py', '')
    currdir = op.dirname(op.abspath(__file__))
    f, p, d = imp.find_module(jobname, [currdir])
    parameters = imp.load_module(jobname, f, p, d).parameters

    sim = Simulator(parameters, data_path=data_path, jobname=jobname, verbose=True)
    sim.simulate(progress_bar=True, connfile=connfile)
    import matplotlib.pyplot as plt
    sim.plot()

    fig = plt.figure()
    sim.connections.weight.hist(bins=100)
    fig.savefig(str(sim.data_path / 'connectivity.png'))
