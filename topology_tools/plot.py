import numpy as np
import neo
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as op
import scipy
import seaborn as sns
import elephant
from elephant import kernels
from elephant.statistics import instantaneous_rate
from elephant.statistics import time_histogram
import matplotlib.lines as mlines
import sys
from data import Data
from tools import despine


class Plot:
    def __init__(self, data_path=None, job_name=None, fig_ext='.png',
                 data=None, t_start=0, t_stop=None):
        print ('Initializing plotting class')
        if  job_name is not None and data_path is not None:
            print ('Saving output to {} with base name {}'.format(data_path, job_name))
        self.fig_ext = fig_ext
        self.job_name = job_name
        self.data_path = data_path

        if isinstance(data, Data):
            self.data = data
            if self.job_name is None:
                self.job_name = data['params'].get('job_name')
            if self.data_path is None:
                self.data_path = data['params'].get('data_path')
        else:
            self.data = Data(data_path, job_name, data)
        self.job_name = self.job_name or ''
        self.data_path = self.data_path or ''
        self.params = self.data['params']
        self.t_start = t_start or 0.
        self.t_stop = t_stop or float(self.params['simtime'])

    def display(self, fname=None, fig=None, save=True, show=False):
        fig.suptitle('{:.1f} ms - {:.1f} ms'.format(self.t_start, self.t_stop))
        if save:
            try:
                os.mkdir(self.job_name)
            except:
                pass
            ext = self.fig_ext
            if ext[0] != '.':
                ext = '.' + ext
            fname = op.join(
                self.data_path, self.job_name, self.job_name + '_' + fname + ext)
            if ext == '.pdf':
                fig.savefig(fname, bbox_inches='tight')
            if ext == '.png':
                fig.savefig(fname, bbox_inches='tight', dpi=300)
            plt.close(fig)
        if show:
            plt.show()

    def voltage(self, pop, N=1, xlim=None, color_th='r', color='k', ax=None,
                save=True, show=False):
        print('Plotting %s %s membrane potential from example neuron' %
              (self.job_name, pop))
        if ax is None:
            fig, ax = plt.subplots()
        data = pd.DataFrame(self.data['state'][pop])
        grp = data.groupby('senders')
        vmin_prev = np.inf
        for cnt, (sender, attr) in enumerate(grp):
            if cnt == N:
                break
            times = np.array(attr['times'])
            ax.plot(times, attr['V_m'])
            vmin = attr['V_m'].min()
            if vmin < vmin_prev:
                vmin_prev = vmin
        ax.set_ylim([vmin, self.params['V_peak_'+pop] + 5])
        if xlim: ax.set_xlim(xlim)
        V_th = self.params['V_th_%s' % pop[:2]]
        ax.axhspan(V_th, V_th, color=color_th, lw=2)
        line = mlines.Line2D([], [], color=color_th, label='V_th')
        ax.set_ylabel('Membrane potential [mV]')
        ax.set_xlabel('Time [ms]')
        self.display(
            fname='{}_voltage'.format(pop),
            fig=plt.gcf(),
            save=save,
            show=show)

    def psp(self, N=1, xlim=None, color_th='r', color='k', ax=None,
            save=True, show=False):
        if ax is None:
            fig, axs = plt.subplots(1,2)
        conns = df = pd.DataFrame(self.data['connections'])
        for source_pop, target_pop, ax in zip(['ex', 'in'], ['in', 'ex'], axs):
            state = pd.DataFrame(self.data['state'][target_pop])
            spikes = pd.DataFrame(self.data['spiketrains'][source_pop])
            state_grp = state.groupby('senders')
            spikes_grp = spikes.groupby('senders')
            prev_ = 0
            for s, attr in spikes_grp:
                if len(attr['times']) > prev_:
                    source = s
                prev_ = len(attr['times'])
            target = conns.groupby('source').get_group(source).target.iloc[0]
            var = state_grp.get_group(target)
            for t in spikes_grp.get_group(source).times:
                mask = (var['times'] >= t) & (var['times'] < t + 10)
                baseline = var['V_m'][mask].iloc[0]
                ax.plot(var['times'][mask] - t, var['V_m'][mask] - baseline)
            ax.set_ylabel('Membrane potential [mV]')
            ax.set_xlabel('Time from spike [ms]')
            ax.set_title('PSP of ' + source_pop + ' to ' + target_pop)
        # ax.set_ylim([vmin, self.params['V_peak_'+pop] + 5])
        # if xlim: ax.set_xlim(xlim)
        # V_th = self.params['V_th_%s' % pop[:2]]
        # ax.axhspan(V_th, V_th, color=color_th, lw=2)
        # line = mlines.Line2D([], [], color=color_th, label='V_th')
        self.display(
            fname='PSP',
            fig=plt.gcf(),
            save=save,
            show=show)

    def pop_rate(self, pop, binsize=2, stimulation=False, ax=None, hist=False,
            title=None, xlim=None, ylim=None, fname=None,
            label=None, legend_loc=2, save=True, show=False, **kwargs):
        print ('Plotting %s %s rate' % (self.job_name, pop))
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        rate, bins = self.data.rate(pop=pop, binsize=binsize)
        rate = rate.mean(axis=0)
        if hist:
            ax.bar(bins[:-1], rate, align='edge',
                   width=bins[1]-bins[0], **kwargs)
        else:
            times = bins[1:] - (binsize / 2.)
            ax.plot(times, rate, **kwargs)

        if stimulation:
            if 'epochs' in self.data:
                epochs = self.data['epochs']
                for t, d in zip(epochs['times'], epochs['durations']):
                    ax.axvspan(t, t+d, color='r', alpha=.5)
            else:
                import warnings
                warnings.warn('No epocharrays avail')
        if title is not None: ax.set_title(title)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is None: ylim = [0, ax.get_ylim()[1]]
        ax.set_ylim(ylim)
        ax.set_ylabel('Rate [Hz]')
        ax.set_xlabel('Time [ms]')
        if label is not None:
            plt.legend(loc=legend_loc)
        self.display(
            fname=fname or '{}_poprate'.format(pop),
            fig=plt.gcf(),
            save=save,
            show=show)

    def pop_rate_spectrum(self, pop, binsize=2, ax=None, nperseg=1024,
                title=None, xlim=None, ylim=None, fname=None,
                label=None, legend_loc=2, save=True, show=False, **kwargs):
        print ('Plotting %s %s population rate spectrum' % (self.job_name, pop))
        import scipy.signal as signal
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        rate, bins = self.data.pop_rate(pop=pop, binsize=binsize)
        fs = 1. / (bins[1] - bins[0]) * 1000.
        freqs, psd = signal.welch(rate, fs=fs, nperseg=nperseg)
        ax.plot(freqs, psd)
        if title is not None: ax.set_title(title)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is None: ylim = [0, ax.get_ylim()[1]]
        ax.set_ylim(ylim)
        ax.set_ylabel('PSD')
        ax.set_xlabel('Frequency [Hz]')
        if label is not None: plt.legend(loc=legend_loc)
        self.display(
            fname=fname or '{}_spectrum'.format(pop),
            fig=plt.gcf(),
            save=save,
            show=show)

    def single_rate_spectrum(self, pop, node=0, binsize=2, nperseg=1024,
                             xlim=None, ylim=None, color='k', ax=None,
                             save=True, show=False, fname=None, title=None):
        print('Plotting {} {} single rate spectrum'.format(self.job_name, pop))
        import scipy.signal as signal
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        rate, bins = self.data.rate(pop=pop, binsize=binsize)
        fs = 1. / (bins[1] - bins[0]) * 1000.
        freqs, psd = signal.welch(rate[node, :], fs=fs, nperseg=nperseg)
        ax.plot(freqs, psd)
        if title is not None: ax.set_title(title)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is None: ylim = [0, ax.get_ylim()[1]]
        ax.set_ylim(ylim)
        ax.set_ylabel('PSD')
        ax.set_xlabel('Frequency [Hz]')
        self.display(
            fname=fname or '{}_spectrum'.format(pop),
            fig=plt.gcf(),
            save=save,
            show=show)

    def raster(self, pop, ax=None, title=None, xlim=None, fname=None,
               N=10, stimulation=False, save=True, show=False, *args):
        print ('Plotting %s %s raster' % (self.job_name, pop))
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        trials = []
        data = self.data['spiketrains'][pop]
        ax.scatter(data['times'], data['senders'], marker='.', markersize=10)

        if stimulation:
            if 'epochs' in self.data:
                epochs = self.data['epochs']
                for t, d in zip(epochs['times'], epochs['durations']):
                    ax.axvspan(t, t+d, color='r', alpha=.5)
            else:
                import warnings
                warnings.warn('No epocharrays avail')
        if title is not None: ax.set_title(title)
        if xlim is not None: ax.set_xlim(xlim)
        ax.set_ylabel('Sender id')
        ax.set_xlabel('Time [ms]')
        self.display(
            fname=fname or '{}_raster'.format(pop),
            fig=plt.gcf(),
            save=save,
            show=show)

    def connectivity(self, ax=None, title=None, xlim=None, fname=None,
                     save=True, show=False):
        print ('Plotting %s connectivity' % (self.job_name))
        df = pd.DataFrame(self.data['connections'])
        if ax is None:
            fig, ax = plt.subplots()
        df.weight.hist(ax=ax, bins=100)
        self.display(
            fname=fname or '{}_connectivity'.format(pop),
            fig=plt.gcf(),
            save=save,
            show=show)

    def scatter_rate_2D(self, pop, binsize=10, ax=None, fname=None, save=True,
                       show=False, cmap='viridis'):
        print ('Plotting ' + self.job_name + ' ' + pop +' mean 2D rate')
        if ax is None:
            fig, ax = plt.subplots()
        rates, _ = self.data.rate(
            pop=pop, binsize=binsize, t_start=self.t_start, t_stop=self.t_stop)
        x, y = self.data.position(pop)
        line_gr = ax.scatter(x, y, c=rates.mean(axis=1), cmap=cmap)
        # dx, dy = x[1] - x[0], y[1] - y[0]
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])
        ax.set_aspect(1)
        self.display(
            fname=fname or '{}_mean_grid_rate'.format(pop),
            fig=plt.gcf(),
            save=save,
            show=show)

    def hist_rate_2D(self, pop, binsize=10, ax=None, fname=None, save=True,
                       show=False, cmap='viridis'):
        print ('Plotting ' + self.job_name + ' ' + pop +' mean 2D rate')
        if ax is None:
            fig, ax = plt.subplots()
        rates, _ = self.data.rate(
            pop=pop, binsize=binsize, t_start=self.t_start, t_stop=self.t_stop)
        x, y = self.data.position(pop)
        ax.hist2d(
            x, y, bins=[len(np.unique(x)), len(np.unique(y))],
            weights=rates.mean(axis=1), cmap=cmap)
        ax.grid(False)
        # line_gr = ax.scatter(x, y, c=rates.mean(axis=1), cmap=cmap)
        # dx, dy = x[1] - x[0], y[1] - y[0]
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])
        ax.set_aspect(1)
        self.display(
            fname=fname or '{}_mean_grid_rate'.format(pop),
            fig=plt.gcf(),
            save=save,
            show=show)

    def mean_rate_1D(self, pop, binsize=10, ax=None, save=True, show=False):
        print ('Plotting %s %s hd rate' % (self.job_name, pop))
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='polar')
        rate, _ = self.data.rate(
            pop=pop, binsize=binsize, t_start=self.t_start, t_stop=self.t_stop)
        pos = self.data.position(pop)[1, :]
        idxs = np.argsort(pos)
        ax.plot(pos[idxs], rate.mean(axis=1)[idxs], 'r')
        self.display(
            fname = '{}_hd_rate'.format(pop),
            fig=plt.gcf(),
            save=save,
            show=show)
