import numpy as np
import nest
import pandas as pd
import quantities as pq
import neo
import os
import shutil
import matplotlib.pyplot as plt
from pprint import pprint
import json


def hasattrs(object, strings):
    if isinstance(strings, str):
        return hasattr(object, strings)
    for string in strings:
        try:
            getattr(object, string)
        except:
            return False
    return True


def convert_params(params):
    nest_units = ['pF', 'ms', 'mV', 'pA', 'nS', 'Hz']
    success_keys = []
    keys_w_units = [k for k, v in params.items() if isinstance(v, pq.Quantity)]
    for unit in nest_units:
        for key, val in params.items():
            if isinstance(val, pq.Quantity):
                try:
                    params.update({key: val.rescale(unit).magnitude})
                    success_keys.append(key)
                except:
                    pass
            if isinstance(val, dict):
                convert_params(val)
    if set(keys_w_units) != set(success_keys):
        raise ValueError('Parameters not sccessfully converted.')


def dict_equal(d1, d2):
    k1, k2 = [set(d.keys()) for d in (d1, d2)]
    intersect = k1.intersection(k2)
    return set(o for o in intersect if d2[o] != d1[o]) == set()


def rtrim(val, trim):
    if not val.endswith(trim):
        return val
    else:
        return val[:len(val) - len(trim)]


class Simulator:
    def __init__(self, parameters, **kwargs):
        if kwargs:
            parameters.update(kwargs)
        convert_params(parameters)

        if 'data_path' not in parameters:
            parameters['data_path'] = os.getcwd()
        if 'fname' not in parameters:
            parameters['fname'] = 'simulation_output'
        else:
            if len(os.path.splitext(parameters['fname'])[-1]) != 0:
                raise ValueError('fname should not have extention, it is' +
                                 ' properly added when needed')
        self.__dict__.update(parameters)
        def_keys = ['nest_kernel_status', 'N_ex', 'N_in', 'N_rec_spike_in',
                    'N_rec_state_in', 'N_rec_spike_ex', 'N_rec_state_ex',
                    'sampling_period_state']
        self._inp_keys = list(parameters.keys())
        self._param_keys = [k for k in self._inp_keys + def_keys]

    def set_kernel(self):
        if hasattr(self, '_data'):
            del(self._data)
        nest.ResetKernel()
        nest.SetKernelStatus({"overwrite_files": True,
                              "data_path": self.data_path,
                              "data_prefix": "",
                              "print_time": True,
                              "local_num_threads": 1})
        msd = self.msd
        N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
        pyrngs = [np.random.RandomState(s) for s in range(msd, msd + N_vp)]
        nest.SetKernelStatus({'grng_seed' : msd + N_vp})
        nest.SetKernelStatus(
            {'rng_seeds' : range(msd + N_vp + 1, msd + 2 * N_vp + 1)})
        nest.SetStatus([0], [{"resolution": self.res}])

    def set_neurons(self):
        assert self.N_neurons % 2 == 0, 'N_neurons must be even number'
        self.N_ex = int(np.ceil(self.N_neurons * self.P_ex))
        self.N_in = int(np.ceil(self.N_neurons * (1 - self.P_ex)))
        if self.N_ex + self.N_in != self.N_neurons:
            raise ValueError('N_ex=%i, N_in=%i' % (self.N_ex, self.N_in) +
                             ', does not add up to total')
        # Create neurons
        self.nodes   = nest.Create(self.neuron, self.N_neurons)
        self.nodes_ex = self.nodes[:self.N_ex]
        self.nodes_in = self.nodes[self.N_ex:]
        def_parameters = nest.GetStatus(self.nodes)[0]
        for pop in ['ex', 'in']:
            parameters = {rtrim(key, '_' + pop): val
                          for key, val in self.__dict__.items()
                          if key.endswith('_' + pop)}
            neuron_parameters = {key: parameters[key]
                                 for key in def_parameters
                                 if key in parameters}
            nest.SetStatus(getattr(self, 'nodes_' + pop),
                           [neuron_parameters])

    def set_connections(self):
        if not self.J_in <= 0:
            raise ValueError('inhibitory weights must be negative.')
        nest.CopyModel("static_synapse", "excitatory",
                       {"weight": self.J_ex, "delay": self.delay})
        nest.CopyModel("static_synapse", "inhibitory",
                       {"weight": self.J_in, "delay": self.delay})
        syn_params_ex = {"model": "excitatory",
                         "receptor_type": {"distribution": "uniform_int",
                                           "low": 1, "high": self.nr_ports}
                         }
        syn_params_in = {"model": "inhibitory",
                         "receptor_type": {"distribution": "uniform_int",
                                           "low": 1, "high": self.nr_ports}
                         }
        conn_params_ex = {'rule': 'fixed_indegree', 'indegree': self.C_ex}
        conn_params_in = {'rule': 'fixed_indegree', 'indegree': self.C_in}
        noise = nest.Create("poisson_generator", 1,
                            params={"rate": self.p_rate})

        nest.Connect(noise, self.nodes_ex, syn_spec=syn_params_ex)
        nest.Connect(noise, self.nodes_in, syn_spec=syn_params_ex)
        nest.Connect(self.nodes_ex, self.nodes,
                     conn_params_ex, syn_params_ex)
        nest.Connect(self.nodes_in, self.nodes,
                     conn_params_in, syn_params_in)

    def set_stimulation(self):
        self.stim = nest.Create(
            "dc_generator",
            params={'amplitude': self.dc_amp,
                    'start': self.trial_start,
                    'stop': self.trial_start + self.trial_duration})
        nest.Connect(self.stim, self.nodes[:self.N_stim])

    def set_spike_rec(self, N_rec=None):
        N_rec = N_rec or self.N_rec
        for pop in ['ex', 'in']:
            if N_rec < getattr(self, 'N_' +  pop):
                setattr(self, 'N_rec_spike_' + pop, N_rec)
            else:
                setattr(self, 'N_rec_spike_' +  pop, getattr(self, 'N_' +  pop))
        assert isinstance(N_rec, int) and N_rec > 0, 'N_rec must be int > 0'
        spikes = nest.Create("spike_detector", 2,
                           params=[{"label": "Exc", "to_file": False},
                                   {"label": "Inh", "to_file": False}])
        self.spikes_ex = spikes[:1]
        self.spikes_in = spikes[1:]
        # connect using all_to_all: all recorded excitatory neurons to one detector
        nest.Connect(self.nodes_ex[:self.N_rec_spike_ex], self.spikes_ex)
        nest.Connect(self.nodes_in[:self.N_rec_spike_in], self.spikes_in)

    def set_state_rec(self, N_rec=None, sampling_period=None):
        N_rec = N_rec or self.N_rec
        assert isinstance(N_rec, int) and N_rec > 0, 'N_rec must be int > 0'
        for pop in ['ex', 'in']:
            if N_rec < getattr(self, 'N_' +  pop):
                setattr(self, 'N_rec_state_' +  pop,  N_rec)
            else:
                setattr(self, 'N_rec_state_' +  pop, getattr(self, 'N_' +  pop))
        if sampling_period is None:
            sampling_period = self.res
        self.sampling_period_state = sampling_period
        self.vm_ex = nest.Create("multimeter",
                        params = {"interval": sampling_period,
                                  "record_from": self.record_from,
                                  "withgid": True,
                                  "to_file": False,
                                  "label": "exc_v",
                                  "withtime": True})
        nest.Connect(self.vm_ex, self.nodes_ex[:self.N_rec_state_ex])
        self.vm_in = nest.Create("multimeter",
                        params = {"interval": sampling_period,
                                  "record_from": self.record_from,
                                  "withgid": True,
                                  "to_file": False,
                                  "label": "inh_v",
                                  "withtime": True})
        # connect using all_to_all: all recorded excitatory neurons to one detector
        nest.Connect(self.vm_in, self.nodes_in[:self.N_rec_state_in])

    def simulate_trials(self):
        self.events = []
        self.simtime = self.trial_start + self.trial_duration
        for n in range(self.num_trials):
            nest.SetStatus(self.stim, {'origin': nest.GetKernelStatus()['time']})
            nest.Simulate(self.simtime)
            self.events.append(self.trial_start + self.simtime * n)
        nest.Simulate(self.simtime)

    def simulate(self):
        self.simtime = (self.trial_start + self.trial_duration) * self.num_trials
        nest.Simulate(self.simtime)

    def generate_data_dict(self):
        self.nest_kernel_status = nest.GetKernelStatus()
        data = {}
        if any(hasattr(self, k) for k in ['spikes_ex', 'spikes_in', 'vm_ex', 'vm_in']):
            data['params'] = {k: getattr(self, k) for k in self._param_keys
                              if hasattr(self, k)}
        else:
            return data
        if hasattrs(self, ('spikes_ex', 'spikes_in')):
            exc = nest.GetStatus(self.spikes_ex, 'events')[0]
            inh = nest.GetStatus(self.spikes_in, 'events')[0]
            for a, t in zip([exc, inh], ['exc', 'inh']):
                if len(a['times']) == 0:
                    print('Warning: no spikes in {} recorded.'.format(t))
            data.update({'spiketrains': {'excitatory': exc,
                                        'inhibitory': inh}})
        if hasattrs(self, ('vm_ex', 'vm_in')):
            exc   = nest.GetStatus(self.vm_ex, "events")[0]
            inh   = nest.GetStatus(self.vm_in, "events")[0]
            data.update({'state':  {'inhibitory': inh,
                                    'excitatory': exc}})
        if hasattr(self, 'events'):
            data.update({
                'epochs': {
                    'times': self.events,
                    'durations': [self.trial_duration] * self.num_trials}})
        conns = nest.GetConnections(source=self.nodes, target=self.nodes)
        data.update({'connections': list(nest.GetStatus(conns))})
        return data

    @property
    def data(self):
        if not hasattr(self, '_data'):
            data = self.generate_data_dict()
            if 'state' not in data and 'spiketrains' not in data:
                fnameout = os.path.join(self.data_path, self.fname + '.npz')
                data = np.load(fnameout)['data'][()]
                params = {k: getattr(self, k) for k in self._param_keys
                          if hasattr(self, k)}
                if not dict_equal(params, data['params']):
                    print('WARNING: Data parameters and self parameters are ' +
                          'not equal. Use "data["params"]"')
            self._data = data
        return self._data

    def save_data(self, overwrite=False):
        fnameout = os.path.join(self.data_path, self.fname + '.npz')
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if not overwrite and os.path.exists(fnameout):
            raise IOError('File {} exist, use overwrite=True.'.format(fnameout))
        np.savez(fnameout, data=self.data)

    def save_raster(self):
        import nest.raster_plot
        for pop in ['in', 'ex']:
            try:
                nest.raster_plot.from_device(getattr(self, 'spikes_%s' % pop),
                                             hist=True)
            except nest.NESTError:
                print('Unable to plot ' + pop)
                continue
            fig = plt.gcf()
            ax = plt.gca()
            ax.set_title(pop)
            fig.savefig(os.path.join(self.data_path, pop + '.png'))

    def dump_params_to_json(self):
        parameters = {k: getattr(self, k) for k in self._param_keys
                      if hasattr(self, k)}
        fnameout = os.path.join(self.data_path, self.fname + '.npz')
        parameters_no_units = {}
        for key, val in parameters.items():
            if isinstance(val, np.array):
                if val.ndim == 0:
                    parameters_no_units[key] = float(val)
                else:
                    parameters_no_units[key] = val.tolist()
            else:
                parameters_no_units[key] = val
        with open(fnameout, 'w') as outfile:
            json.dump(parameters_no_units, outfile,
                      sort_keys=True, indent=4)
