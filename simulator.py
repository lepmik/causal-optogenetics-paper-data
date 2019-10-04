import numpy as np
import pathlib
import nest
import pandas as pd
import os
import json
import time
import copy
try:
    from tqdm import tqdm
    PBAR = True
except ImportError:
    PBAR = False

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
    def __init__(self, parameters, verbose=False, **kwargs):
        parameters = copy.deepcopy(parameters)
        if kwargs:
            parameters.update(kwargs)

        if 'data_path' not in parameters:
            parameters['data_path'] = os.getcwd()
        self.data_path = pathlib.Path(parameters['data_path'])
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

    def set_spike_rec(self):
        spks = nest.Create("spike_detector", 2,
                         params=[{"label": "ex", "to_file": True},
                                 {"label": "in", "to_file": True}])
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
                          "record_from": ['V_m', 'g_ex', 'g_in'],
                          "withgid": True, "to_file": True,
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

        # self.vprint('Gathering connections')
        # tstart = time.time()
        # conns = nest.GetConnections(
        #     source=self.stim_nodes_ex + self.stim_nodes_in,
        #     target=self.nodes)
        # conn_include = ('weight', 'source', 'target')
        # conns = list(nest.GetStatus(conns, conn_include))
        # conns = pd.DataFrame(conns, columns=conn_include)
        # conns.to_feather(self.data_path /
        #                  'connections_{}.feather'.format(nest.Rank()))
        # self.vprint('Time lapsed {:.2f} s'.format(time.time() - tstart))

        # Simulate one period without stimulation
        self.vprint('Simulating initial time')
        tstart = time.time()
        nest.Simulate(self.p['init_simtime'])
        self.vprint('Time lapsed {:.2f} s'.format(time.time() - tstart))

        # Run multiple trials
        self.vprint('Simulating trials')
        tstart = time.time()

        keys = [
              'V_m', #Membrane potential in mV
              'g_ex',
              'g_in']

        def get_status(nodes):
            status = nest.GetStatus(nodes)

            status = [
                {k: s[k] for k in keys}
                for s in status
            ]
            return status

        status = get_status(self.nodes)
        print([s for i, s in enumerate(status) if i < 3])
        raise
        self.stim_times = []
        for i in progress_bar(range(self.p['n_trials'])):
            nest.SetStatus(self.nodes, status)
            if i > 0:
                nest.Simulate(self.p['delay'])
                # status = get_status(self.nodes)
            stim_time = nest.GetKernelStatus()['time']
            self.stim_times.append(stim_time)
            for stim in stims:
                nest.SetStatus(stim, {'origin': stim_time})
            nest.Simulate(self.p['simtime'])

        self.vprint('Time lapsed {:.2f} s'.format(time.time() - tstart))

        self.vprint('Gathering stim times')
        data = {
            'epoch': {
                'times': np.array(self.stim_times),
                'durations': [self.p['stim_duration']] * len(self.stim_times)},
            'stim_nodes': {
                'ex': self.stim_nodes_ex,
                'in': self.stim_nodes_in},
            'stim_amps': {
                'ex': self.stim_amps_ex,
                'in': self.stim_amps_in}
        }
        np.savez(self.data_path / 'stimulation_data_{}.npz'.format(nest.Rank()), data=data)


    def simulate(self, state=False,
                 progress_bar=False):
        self.vprint('Setting kernel')
        self.set_kernel()
        self.vprint('Setting neurons')
        self.set_neurons()
        self.vprint('Setting background')
        self.set_background()
        self.vprint('Setting connections')
        self.set_connections()
        self.vprint('Setting spike recording')
        self.set_spike_rec()
        if state:
            self.vprint('Setting state recording')
            self.set_state_rec()
        simtime = (self.p['simtime'] + self.p['delay']) * self.p['n_trials'] + self.p['init_simtime']
        self.vprint(
            'Simulating {} trials,'.format(self.p['n_trials']),
            'total {} ms'.format(simtime)
        )
        tstart = time.time()
        self.simulate_trials(progress_bar=progress_bar)
        self.vprint('Simulation lapsed {:.2f} s'.format(time.time() - tstart))
        self.vprint('Saving parameters')
        self.dump_params_to_json()

    def dump_params_to_json(self):
        status = nest.GetKernelStatus()
        include = ['time']
        status = {k:v for k,v in status.items() if k in include}
        self.p['status'] = status

        fnameout = str(self.data_path / 'params.json')
        with open(fnameout, 'w') as outfile:
            json.dump(self.p, outfile,
                      sort_keys=True, indent=4,
                      cls=NumpyEncoder)


if __name__ == '__main__':
    import sys
    import imp
    import os.path as op

    if len(sys.argv) not in [2, 3]:
        raise IOError('Usage: "python simulator.py parameters <load-data>"')

    data_path, param_module = sys.argv[1:]
    os.makedirs(data_path, exist_ok=True)
    jobname = param_module.replace('.py', '')
    currdir = op.dirname(op.abspath(__file__))
    f, p, d = imp.find_module(jobname, [currdir])
    parameters = imp.load_module(jobname, f, p, d).parameters

    sim = Simulator(parameters, data_path=data_path, jobname=jobname, verbose=True)
    sim.simulate(state=True, progress_bar=True)
