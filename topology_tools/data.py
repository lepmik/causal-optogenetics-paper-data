import numpy as np
import neo
import pandas as pd
import os
import os.path as op
import scipy
import elephant
from elephant import kernels
from elephant.statistics import instantaneous_rate
from elephant.statistics import time_histogram
import sys


def myround(x, base=5):
    if x < 0:
        f = np.floor
    else:
        f = np.ceil
    return int(base * f(float(x)/base))


class Data:
    def __init__(self, data_path=None, job_name=None, data=None):
        if data is None:
            fname = data_path / (job_name + '.npz')
            data = np.load(fname)['data'][()]
        self._data = data

    def __getitem__(self, name):
        return self._data[name]

    def __contains__(self, name):
        return name in self._data

    def state_extrema(self, pop, t_start=None, t_stop=None):
        V = self['state'][pop]['V_m']
        w = self['state'][pop]['w']
        t = self['state'][pop]['times']
        if (t_start, t_stop) != (None, None):
            mask = (t > t_start) & (t <= t_stop)
            v, w, t = v[mask], w[mask], t[mask]
        Vmax = self['params']['V_peak_'+pop[:2]] + 5
        Vmin = myround(V.min(), base=10)
        wmin = 0 if w.min() >= 0 else myround(w.min(), base=50)
        wmax = 1 if w.max() <= 1 else myround(w.max(), base=50)
        return Vmin, Vmax, wmin, wmax

    def state(self, pop, t_start=None, t_stop=None):

        def result():
            if (t_start, t_stop) != (None, None):
                if not all(isinstance(a, float) for a in [t_start, t_stop]):
                    raise TypeError(
                        't_start, t_stop must be "float" not ' +
                        '{}'.format([type(a) for a in [t_start, t_stop]]))
                v, w, t = self._state[pop]
                mask = (t > t_start) & (t <= t_stop)
                return v[:, mask], w[:, mask], t[mask]
            else:
                return self._state[pop]

        if hasattr(self, '_state'):
            if pop in self._state:
                return result()
        else:
            self._state = {}
        V = self['state'][pop]['V_m']
        w = self['state'][pop]['w']
        t = self['state'][pop]['times']
        assert len(t) == len(w) == len(V)
        df = pd.DataFrame(self['state'][pop])
        grp = df.groupby('senders')
        senders = np.unique(self['location'][pop]['gid'])
        n_samples = int(self['params']['simtime'] / self['params']['sampling_period'])
        v = np.zeros((len(senders), n_samples-1))
        w = np.zeros((len(senders), n_samples-1))
        for sender, attr in grp:
            idx = self['location'][pop]['gid'].index(sender)
            v[idx, :] = np.array(attr['V_m'])
            w[idx, :] = np.array(attr['w'])
        t = np.array(attr['times'])
        self._state[pop] = (v, w, t)
        return result()

    def pos_state(self, pop, t_start=None, t_stop=None):

        if t_start and t_stop:
            times = attr['times']
            mask = (times > t_start) & (times <= t_stop)
            return V[:, mask], w[:, mask]
        else:
            return V, w

    def position(self, pop):
        return self['location'][pop]['pos']

    def pop_rate(self, pop, binsize, t_start=None, t_stop=None):
        bins = np.arange(0, self['params']['simtime'], binsize)
        histogram, _ = np.histogram(self['spiketrains'][pop]['times'],
                                       bins=bins)
        rate = histogram / (self['params']['N_rec_spikes_'+pop[:2]] * (binsize / 1000.))
        time = bins[1:] - (binsize / 2.)
        if t_start and t_stop:
            times = bins[-1]
            mask = (times > t_start) & (times <= t_stop)
            return rate[mask], time[mask]
        else:
            return rate, time

    def rate(self, pop, binsize, t_start=None, t_stop=None):
        def result():
            if (t_start, t_stop) != (None, None):
                if not all(isinstance(a, float) for a in [t_start, t_stop]):
                    raise TypeError(
                        't_start, t_stop must be "float" not ' +
                        '{}'.format([type(a) for a in [t_start, t_stop]]))
                times = self._rate[pop][1][:-1]
                mask = (times > t_start) & (times <= t_stop)
                return self._rate[pop][0][:, mask], times[mask]
            else:
                return self._rate[pop]

        if hasattr(self, '_rate'):
            if self._rate['settings'] == (binsize,):
                if pop in self._rate:
                    return result()
        else:
            self._rate = {}
        df = pd.DataFrame(self['spiketrains'][pop])
        grp = df.groupby('senders')
        bins = np.arange(0., self['params']['simtime'] + binsize, binsize)
        rates = np.zeros(
            (self['params']['N_rec_spikes_'+pop[:2]], len(bins) - 1))
        for ii, (sender, attr) in enumerate(grp):
            r, _ = np.histogram(attr['times'], bins=bins)
            r = r / (binsize / 1000.)
            if 'location' in self:
                idx = self['location'][pop]['gid'].index(sender)
            else:
                idx = ii
            rates[idx, :] = r
        self._rate[pop] = (rates, bins)
        self._rate['settings'] = (binsize,)
        return result()
