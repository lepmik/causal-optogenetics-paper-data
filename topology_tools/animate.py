import numpy as np
import neo
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import scipy
import seaborn as sns
import elephant
from elephant import kernels
from elephant.statistics import instantaneous_rate
from elephant.statistics import time_histogram
import matplotlib.lines as mlines
import sys
from data import Data
try:
    from tqdm import tqdm
    PBAR = tqdm
except ImportError:
    PBAR = lambda x: x


def prob_color(p, cmap):
    n = mpl.colors.Normalize(vmin=min(p), vmax=max(p))
    m = mpl.cm.ScalarMappable(norm=n, cmap=cmap)
    # return m.to_rgba(p)
    return m


def norm(a):
    l = np.log10(a.ravel() + 1e-6)
    l = (l - l.min()) / (l.max() - l.min())
    return l.reshape(a.shape)


def simpleaxis(ax, left=True, right=True, top=True, bottom=True, ticks=True):
    """
    Removes axis lines
    """
    try:
        ax.spines['top'].set_visible(top)
        ax.spines['right'].set_visible(right)
        ax.spines['left'].set_visible(left)
        ax.spines['bottom'].set_visible(bottom)
    except AttributeError:
        pass
    except:
        raise
    if not bottom and not ticks:
        ax.get_xaxis().tick_bottom()
        plt.setp(ax.get_xticklabels(), visible=False)
    if not left and not ticks:
        ax.get_yaxis().tick_left()
        plt.setp(ax.get_yticklabels(), visible=False)


rc = {
    'figure.figsize': (16, 9),
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.labelsize': 30,
    'axes.titlesize': 25,
    'font.size': 25,
    'legend.frameon': False,
    'legend.fontsize': 15,
    'image.cmap': 'jet'
}


class Animate:
    def __init__(self, data_path=None, job_name=None, save_figs=True, dt=2.,
                 fig_ext='.png', show=False, data=None, t_start=0, t_stop=None):
        print ('Initializing animation class')
        if save_figs:
            assert job_name is not None
            assert data_path is not None
            print ('Saving output to %s with base name %s' % (data_path, job_name))
            sns.set(rc=rc)
            sns.set_color_codes()
            plt.rcParams.update(rc)
        self.dt = dt
        self.fig_ext = fig_ext
        self.show = show
        self.save_figs = save_figs
        self.job_name = job_name
        self.data_path = data_path
        self.t_start = t_start or 0.
        try:
            os.mkdir(self.data_path)
        except:
            pass
        try:
            os.mkdir(self.job_name)
        except:
            pass
        if isinstance(data, Data):
            self.data = data
        else:
            self.data = Data(data_path, job_name, data)
        self.params = self.data['params']
        self.t_stop = t_stop or self.params['nest_kernel_status']['time']

    def display(self, fname=None, save=False, show=False):
        fig = plt.gcf()
        if show:
            fig.canvas.draw()
            plt.pause(1e-17)
        if save:
            fname = fname.with_suffix(self.fig_ext)
            if self.fig_ext == '.pdf':
                fig.savefig(str(fname), bbox_inches='tight')
            if self.fig_ext == '.png':
                fig.savefig(str(fname), bbox_inches='tight') # possibility to add dpi

    def generate_path(self, name, save):
        path = self.data_path / (self.job_name + '_' + name + '.ani')
        if save:
            path.mkdir(exist_ok=True)
        fname = path / self.job_name
        return path, fname

    def updatefig(self, fname, save, show, updates):
        if show:
            plt.ion()
        for t in PBAR(np.arange(self.t_start, self.t_stop - self.dt, self.dt)):
            for update in updates:
                update(t, t + self.dt)
            plt.suptitle(
                r"Time: {}, $\Delta t$: {}".format(t, self.dt))
            self.display(
                fname=fname.with_name(fname.stem +'_' + str(int(t / self.dt))),
                save=save, show=show)

    def scatter_rate_2D(self, pop,  binsize=1., cmap='viridis', save=False, show=False,
                ax=None, multiple=False):
        print ('Animating ' + self.job_name + ' ' + pop +' grid rate')
        assert binsize <= self.dt
        path, fname = self.generate_path(
            '_' + pop + '_rate_2D', save=save)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        x, y = self.data.position(pop=pop)
        r, _ = self.data.rate(
            pop=pop, binsize=binsize,
            t_start=self.t_start, t_stop=self.t_stop)
        cs = prob_color(r.mean(axis=1), cmap=cmap)
        line = ax.scatter(x, y)
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])
        ax.set_aspect(1)

        def update(t_start, t_stop):
            rate, _ = self.data.rate(
                pop=pop, binsize=binsize, t_start=t_start, t_stop=t_stop)
            if rate.size > 0:
                line.set_facecolor(cs.to_rgba(rate[:, 0]))

        if multiple:
            return update
        else:
            self.updatefig(fname=fname, save=save, show=show, updates=[update])


    def hist_rate_2D(self, pop,  binsize=1., cmap='viridis', save=False, show=False,
                ax=None, multiple=False):
        print ('Animating ' + self.job_name + ' ' + pop +' grid rate')

        assert binsize <= self.dt
        path, fname = self.generate_path(
            '_' + pop + '_rate_2D', save=save)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        x, y = self.data.position(pop=pop)
        bins = [len(np.unique(x)), len(np.unique(y))] # TODO extract numbers from params
        r, _ = self.data.rate(
            pop=pop, binsize=binsize,
            t_start=self.t_start, t_stop=self.t_stop)
        range = r.mean(axis=1)
        norm = mpl.colors.Normalize(vmin=min(range), vmax=max(range))
        image = ax.imshow(
            [[], []], extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap,
            norm=norm)
        ax.grid(False)
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])
        ax.set_aspect(1)

        def update(t_start, t_stop):
            rate, _ = self.data.rate(
                pop=pop, binsize=binsize, t_start=t_start, t_stop=t_stop)
            if rate.size > 0:
                data, _, _ = np.histogram2d(
                    x, y, bins=bins,
                    weights=rate[:, 0])
                image.set_data(data)

        if multiple:
            return update
        else:
            self.updatefig(fname=fname, save=save, show=show, updates=[update])

    def voltage_2D(self, pop, cmap='viridis', save=False, show=False,
                 ax=None, multiple=False):
        print ('Animating ' + self.job_name + ' ' + pop + 'voltage_2D')
        import matplotlib as mpl
        path, fname = self.generate_path(
            '_' + pop + '_voltage_2D', save=save)
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        x, y = self.data.position(pop=pop)
        bins = [len(np.unique(x)), len(np.unique(y))] # TODO extract numbers from params
        v, w, _ = self.data.state(
            pop=pop, t_start=self.t_start, t_stop=self.t_stop)
        range = v.ravel()
        norm = mpl.colors.Normalize(vmin=min(range), vmax=max(range))
        image = ax.imshow(
            [[], []], extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap,
            norm=norm)
        ax.grid(False)
        inc_x, inc_y = x.max() * 0.01, y.max() * 0.01
        ax.set_xlim([x.min()- inc_x, x.max() + inc_x])
        ax.set_ylim([y.min() - inc_y, y.max() + inc_y])
        ax.set_aspect(1)

        def update(t_start, t_stop):
            v, w, _ = self.data.state(
                pop=pop, t_start=t_start, t_stop=t_stop)
            data, _, _ = np.histogram2d(
                x, y, bins=bins,
                weights=v[:, 0])
            image.set_data(data)

        if multiple:
            return update
        else:
            self.updatefig(fname=fname, save=save, show=show, updates=[update])

    def rate(self, pop,  binsize=1., cmap='viridis', color='k',
             lw=2, save=False, show=False, ax=None, multiple=False):
        print ('Animating %s %s state and rate' % (self.job_name, pop))
        import matplotlib.cm as cm
        assert binsize <= self.dt
        if cmap is None:
            cmap = plt.rcParams['image.cmap']
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        rate, rate_bins = self.data.pop_rate(
            pop=pop, binsize=binsize)
        path, fname = self.generate_path(
            '_' + pop + '_rate_density', save=save)
        line, = ax.plot([], [], color=color, lw=lw)

        tmax, rmax = rate_bins.max(), rate.max()
        ax.set_xlim([0, tmax])
        ax.set_ylim([0, rmax * 1.001])
        aspect = tmax / (rmax + rmax * 0.001)
        ax.set_aspect(aspect)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Rate [Hz]')
        simpleaxis(ax, right=False, top=False)

        def update(t_start, t_stop):
            rate_update, rate_bins_update = self.data.rate(
                pop=pop, binsize=binsize, t_start=0., t_stop=t_stop)
            line.set_data(rate_bins_update, rate_update.mean(axis=0))

        if multiple:
            return update
        else:
            self.updatefig(fname=fname, save=save, show=show, updates=[update])

    def rate_1D(self, pop, binsize=1., color='k', lw=2, save=False,
                show=False, ax=None, multiple=False):
        print ('Animating ' + self.job_name + ' ' + pop + ' 1D rate')
        import matplotlib.animation as animation
        if ax is None:
            fig, ax = plt.subplots()
        path, fname = self.generate_path(
            '_' + pop + '_1D_rate', save=save)
        rate, _ = self.data.rate(
            pop=pop, binsize=binsize, t_start=self.t_start, t_stop=self.t_stop)
        pos = self.data.position(pop)[1, :]
        idxs = np.argsort(pos)
        line, = ax.plot([], [], color=color, lw=lw)
        ax.set_xlim([pos.min(), pos.max()])
        m = rate.max(axis=1).max()
        ax.set_ylim([0 - m * 0.01, m + m * 0.01])
        ax.set_ylabel('rate ' + pop)

        def update(t_start, t_stop):
            rate_update, _ = self.data.rate(
                pop=pop, binsize=binsize, t_start=t_start, t_stop=t_stop)
            line.set_data(pos[idxs], rate_update[idxs, 0])

        if multiple:
            return update
        else:
            self.updatefig(fname=fname, save=save, show=show, updates=[update])

    def voltage_1D(self, pop, color='k', lw=2, save=False,
                 show=False, ax=None, multiple=False):
        print ('Animating %s %s voltage' % (self.job_name, pop))
        import matplotlib.animation as animation
        if ax is None:
            fig, ax = plt.subplots()
        path, fname = self.generate_path(
            '_' + pop + '_1D_voltage', save=save)
        v, w, _ = self.data.state(
            pop=pop, t_start=self.t_start, t_stop=self.t_stop)
        vmin, vmax, wmin, wmax = self.data.state_extrema(pop)
        var, lim = v, (vmin, vmax)
        pos = self.data.position(pop)[1, :]
        bars = ax.bar(pos, np.zeros_like(pos), width=abs(pos[0] - pos[1]))
        ax.set_xlim([pos.min(), pos.max()])
        ax.set_ylim(lim)
        ax.set_ylabel('voltage ' + pop)

        def update(t_start, t_stop):
            v, w, _ = self.data.state(
                pop=pop, t_start=t_start, t_stop=t_stop)
            for i, b in enumerate(bars):
                b.set_height(v[i, 0])

        if multiple:
            return update
        else:
            self.updatefig(fname=fname, save=save, show=show, updates=[update])
