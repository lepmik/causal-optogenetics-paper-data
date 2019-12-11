import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import quantities as pq
from tools_analysis import make_spiketrain_trials
import neo
import os.path as op
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def fix_figure(padLegend=None, padHeight=0):
    plt.draw() #to know size of legend
    fig = plt.gcf()
    ax = plt.gca()
    dpi = fig.get_dpi()
    if padLegend is None:
        padLegend = ax.get_legend().get_frame().get_width() / dpi
    else:
        assert isinstance(padLegend, float)

    widthAx, heightAx = fig.get_size_inches()
    pos = ax.get_position()
    padLeft   = pos.x0 * widthAx
    padBottom = pos.y0 * heightAx
    padTop    = (1 - pos.y0 - pos.height) * widthAx
    padRight  = (1 - pos.x0 - pos.width) * heightAx

    widthTot = widthAx + padLeft + padRight + padLegend
    heightTot = heightAx + padTop + padBottom

    # set figure size and ax position
    fig.set_size_inches(widthTot, heightTot)
    ax.set_position([padLeft / widthTot, padBottom / heightTot,
                     widthAx / widthTot, heightAx / heightTot + padHeight])
    plt.draw()

def savefig(fig, fname, ext='.svg', **kwargs):
    if ext[0] != '.':
        ext = '.' + ext
    fname = op.splitext(fname)[0]
    fig.savefig(op.join('manuscript',fname + ext), **kwargs)


def despine(ax=None, left=False, right=True, top=True, bottom=False,
            xticks=True, yticks=True):
    """
    Removes axis lines
    """
    if ax is None:
        ax = plt.gcf().get_axes()
    if not isinstance(ax, (list, tuple)):
        ax = [ax]
    for a in ax:
        try:
            a.spines['top'].set_visible(not top)
            a.spines['right'].set_visible(not right)
            a.spines['left'].set_visible(not left)
            a.spines['bottom'].set_visible(not bottom)
        except KeyError:
            pass

        a.get_xaxis().tick_bottom()
        plt.setp(a.get_xticklabels(), visible=xticks)
        if not xticks:
            a.xaxis.set_ticks_position('none')
        a.get_yaxis().tick_left()
        plt.setp(a.get_yticklabels(), visible=yticks)
        if not yticks:
            a.yaxis.set_ticks_position('none')


def set_style(style='article', sns_style='white', w=1, h=1):
    sdict = {
        'article': {
            # (11pt font = 360pt, 4.98) (10pt font = 345pt, 4.77)
            'figure.figsize' : (4.98 * w, 2 * h),
            'figure.autolayout': False,
            'lines.linewidth': 2,
            'font.size'      : 11,
            'legend.frameon' : False,
            'legend.fontsize': 11,
            'font.family'    : 'serif',
            'text.usetex'    : True
        },
        'notebook': {
            'figure.figsize' : (12, 6),
            'axes.labelsize' : 'xx-large',
            'lines.linewidth': 2,
            'xtick.labelsize': 'xx-large',
            'ytick.labelsize': 'xx-large',
            'axes.titlesize' : 'xx-large',
            'legend.frameon' : False,
            'legend.fontsize': 'xx-large',
            'font.family'    : 'serif',
            'text.usetex'    : True
        }
    }
    rc = sdict[style]
    plt.rcParams.update(rc)
    sns.set(rc=rc, style=sns_style,
            color_codes=True)


def add_caption(ax, start='a', vspace=1, **kw):
    '''
    Adds a caption e.g. (a) underneath the xlabel.

    Parameters
    ----------
    axs : list of matplotlib axes
    start : str
        The starting letter
    vspace : int
        Vertical padding between xlabel and caption, uses latex "\\["vspace"ex]"
    kw : dict
        Keywords given to matplotlib.axes.Axes.set_xlabel
    '''
    import string
    alph = string.ascii_lowercase
    alph = alph[alph.index(start.lower()):]
    ax = ax if isinstance(ax, (list, tuple)) else [ax]
    for a, capt in zip(ax, alph):
        label = a.get_xlabel()
        if label == '':
            a.set_xlabel(r'({})'.format(capt), **kw)
        else:
            a.set_xlabel(
                r'\begin{{center}}{}\\[{}ex]({})\end{{center}}'.format(label, vspace, capt),
                **kw)


def label_diff(x1, x2, y, text, ax, color='k', pad_txt=.03, pad_line=0.01):
    '''
    Make line connected to x1 and x2 with text on top

    Parameters
    ----------
    x1 : float
        Postion one
    x2 : float
        Postion two
    y : float
        Hight
    text : str
        Text on top of line
    '''
    h = y * pad_line
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color=color)
    ax.text((x1 + x2) * .5, y + y * pad_txt, text, ha='center', va='bottom',
            color=color)


def plot_psth(spike_train=None, epoch=None, trials=None, xlim=[None, None],
              fig=None, axs=None, legend_loc=1, color='b',
              rast_ylabel='Trials', rast_size=10,
              hist_color=None, hist_edgecolor=None,
              hist_ylim=None,  hist_ylabel=None,
              hist_output='counts', hist_binsize=None, hist_nbins=100,
              hist_alpha=1., hist_log=False):
    """
    Visualize clustering on amplitude at detection point

    Parameters
    ----------
    spike_train : neo.SpikeTrain
    epoch : neo.Epoch
    trials : list of cut neo.SpikeTrains with same number of recording channels
    xlim : list
        limit of x axis
    fig : matplotlib figure
    axs : matplotlib axes (must be 2)
    legend_loc : 'outside' or matplotlib standard loc
    color : color of spikes
    stim_alpha : float
    stim_color : str
    stim_label : str
    stim_style : 'patch' or 'line'
        The amount of offset for the stimulus relative to epoch.
    rast_ylabel : str
    hist_color : str
    hist_edgecolor : str
    hist_ylim : list
    hist_ylabel : str
    hist_output : str
        Accepts 'counts', 'rate' or 'mean'.
    hist_binsize : pq.Quantity
    hist_nbins : int


    Returns
    -------
    out : fig
    """
    if fig is None and axs is None:
        fig, (hist_ax, rast_ax) = plt.subplots(2, 1, sharex=True)
    elif fig is not None and axs is None:
        hist_ax = fig.add_subplot(2, 1, 1)
        rast_ax = fig.add_subplot(2, 1, 2, sharex=hist_ax)
    else:
        assert len(axs) == 2
        hist_ax, rast_ax = axs

    if trials is None:
        assert spike_train is not None and epoch is not None
        t_start = xlim[0] or 0 * pq.s
        t_stop = xlim[1] or epoch.durations[0]
        trials = make_spiketrain_trials(epoch=epoch, t_start=t_start,
                                        t_stop=t_stop, spike_train=spike_train)
    else:
        assert spike_train is None
    dim = trials[0].times.dimensionality

    # raster
    plot_raster(trials, color=color, ax=rast_ax, ylabel=rast_ylabel,
                marker_size=rast_size)
    # histogram
    hist_color = color if hist_color is None else hist_color
    hist_ylabel = hist_output if hist_ylabel is None else hist_ylabel
    plot_spike_histogram(trials, color=hist_color, ax=hist_ax,
                         output=hist_output, binsize=hist_binsize,
                         nbins=hist_nbins, edgecolor=hist_edgecolor,
                         ylabel=hist_ylabel, alpha=hist_alpha,
                         log=hist_log)
    if hist_ylim is not None: hist_ax.set_ylim(hist_ylim)
    return fig


def plot_spike_histogram(trials, color='b', ax=None, binsize=None, bins=None,
                         output='counts', edgecolor=None, alpha=1., ylabel=None,
                         nbins=None, log=False):
    """
    histogram plot of trials

    Parameters
    ----------
    trials : list of neo.SpikeTrains
    color : str
        Color of histogram.
    edgecolor : str
        Color of histogram edges.
    ax : matplotlib axes
    output : str
        Accepts 'counts', 'rate' or 'mean'.
    binsize :
        Binsize of spike rate histogram, default None, if not None then
        bins are overridden.
    nbins : int
        Number of bins, defaults to 100 if binsize is None.
    ylabel : str
        The ylabel of the plot, if None defaults to output type.

    Examples
    --------
    >>> import neo
    >>> from numpy.random import rand
    >>> from exana.stimulus import make_spiketrain_trials
    >>> spike_train = neo.SpikeTrain(rand(1000) * 10, t_stop=10, units='s')
    >>> epoch = neo.Epoch(times=np.arange(0, 10, 1) * pq.s,
    ...                   durations=[.5] * 10 * pq.s)
    >>> trials = make_spiketrain_trials(spike_train, epoch)
    >>> ax = plot_spike_histogram(trials, color='r', edgecolor='b',
    ...                           binsize=1 * pq.ms, output='rate', alpha=.5)

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import quantities as pq
        import neo
        from numpy.random import rand
        from exana.stimulus import make_spiketrain_trials
        from exana.statistics import plot_spike_histogram
        spike_train = neo.SpikeTrain(rand(1000) * 10, t_stop=10, units='s')
        epoch = neo.Epoch(times=np.arange(0, 10, 1) * pq.s, durations=[.5] * 10 * pq.s)
        trials = make_spiketrain_trials(spike_train, epoch)
        ax = plot_spike_histogram(trials, color='r', edgecolor='b', binsize=1 * pq.ms, output='rate', alpha=.5)
        plt.show()

    Returns
    -------
    out : axes
    """
    ### TODO
    if bins is not None:
        assert isinstance(bins, int)
        warnings.warn('The variable "bins" is deprecated, use nbins in stead.')
        nbins = bins
    ###
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    from elephant.statistics import time_histogram
    dim = trials[0].times.dimensionality
    t_start = trials[0].t_start.rescale(dim)
    t_stop = trials[0].t_stop.rescale(dim)
    if binsize is None:
        if nbins is None:
            nbins = 100
        binsize = (abs(t_start)+abs(t_stop))/float(nbins)
    else:
        binsize = binsize.rescale(dim)
    time_hist = time_histogram(trials, binsize, t_start=t_start,
                               t_stop=t_stop, output=output, binary=False)
    bs = np.arange(t_start.magnitude, t_stop.magnitude, binsize.magnitude)
    if ylabel is None:
        if output == 'counts':
            ax.set_ylabel('count')
        elif output == 'rate':
            time_hist = time_hist.rescale('Hz')
            if ylabel:
                ax.set_ylabel('rate [%s]' % time_hist.dimensionality)
        elif output == 'mean':
            ax.set_ylabel('mean count')
    elif isinstance(ylabel, str):
        ax.set_ylabel(ylabel)
    else:
        raise TypeError('ylabel must be str not "' + str(type(ylabel)) + '"')
    ax.bar(bs[:len(time_hist)], time_hist.magnitude.flatten(), width=bs[0]-bs[1],
           edgecolor=edgecolor, facecolor=color, alpha=alpha, align='edge')
    if log:
        ax.set_yscale('log')
    return ax


def plot_raster(trials, color="#3498db", lw=1, ax=None, marker='.', marker_size=10,
                ylabel='Trials', id_start=0, ylim=None):
    """
    Raster plot of trials

    Parameters
    ----------
    trials : list of neo.SpikeTrains
    color : color of spikes
    lw : line width
    ax : matplotlib axes

    Returns
    -------
    out : axes
    """
    from matplotlib.ticker import MaxNLocator
    if ax is None:
        fig, ax = plt.subplots()
    trial_id = []
    spikes = []
    dim = trials[0].times.dimensionality
    for n, trial in enumerate(trials):  # TODO what about empty trials?
        n += id_start
        spikes.extend(trial.times.magnitude)
        trial_id.extend([n]*len(trial.times))
    if marker_size is None:
        heights = 6000./len(trials)
        if heights < 0.9:
            heights = 1.  # min size
    else:
        heights = marker_size
    ax.scatter(spikes, trial_id, marker=marker, s=heights, lw=lw, color=color,
               edgecolors='face')
    if ylim is None:
        ax.set_ylim(-0.5, len(trials)-0.5)
    elif ylim is True:
        ax.set_ylim(ylim)
    else:
        pass
    y_ax = ax.axes.get_yaxis()  # Get X axis
    y_ax.set_major_locator(MaxNLocator(integer=True))
    t_start = trials[0].t_start.rescale(dim)
    t_stop = trials[0].t_stop.rescale(dim)
    ax.set_xlim([t_start, t_stop])
    ax.set_xlabel("Times [{}]".format(dim))
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax


def plot_xcorr(spike_trains, colors=None, edgecolors=None, fig=None,
               density=True, alpha=1., gs=None, binsize=1*pq.ms,
               time_limit=1*pq.s, split_colors=True, xcolor='k',
               xedgecolor='k', xticksvisible=True, yticksvisible=True,
               acorr=True, ylim=None, names=None):
    """
    Bar plot of crosscorrelation of multiple spiketrians

    Parameters
    ----------
    spike_trains : list of neo.SpikeTrain or neo.SpikeTrain
    colors : list or str
        colors of histogram
    edgecolors : list or str
        edgecolor of histogram
    ax : matplotlib axes
    alpha : float
        opacity
    binsize : Quantity
        binsize of spike rate histogram, default 2 ms
    time_limit : Quantity
        end time of histogram x limit, default 100 ms
    gs : instance of matplotlib.gridspec
    split_colors : bool
        if True splits crosscorrelations into colors from respective
        autocorrelations
    xcolor : str
        color of crosscorrelations
    xedgecolor : str
        edgecolor of crosscorrelations
    xticksvisible : bool
        show xtics on crosscorrelations, (True by default)
    yticksvisible : bool
        show ytics on crosscorrelations, (True by default)
    acorr : bool
        show autocorrelations, (True by default)

    Examples
    --------
    >>> import neo
    >>> from numpy.random import rand
    >>> sptr1 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
    >>> sptr2 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
    >>> sptr3 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
    >>> fig = plot_xcorr([sptr1, sptr2, sptr3])

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import quantities as pq
        import neo
        from numpy.random import rand
        from exana.statistics import plot_xcorr
        sptr1 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
        sptr2 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
        sptr3 = neo.SpikeTrain(rand(100) * 2, t_stop=2, units='s')
        plot_xcorr([sptr1, sptr2, sptr3])
        plt.show()

    Returns
    -------
    out : fig
    """
    if len(spike_trains) == 1:
        spike_trains = [spike_trains]
    from tools_analysis import correlogram
    import matplotlib.gridspec as gridspec
    if colors is None:
        from matplotlib.pyplot import cm
        colors = cm.rainbow(np.linspace(0, 1, len(spike_trains)))
    elif not isinstance(colors, list):
        colors = [colors] * len(spike_trains)
    if edgecolors is None:
        edgecolors = colors
    elif not isinstance(edgecolors, list):
        edgecolors = [edgecolors] * len(spike_trains)

    def get_name(sptr, idx):
        if hasattr(sptr, 'name'):
            name = sptr.name
        elif names is not None:
            assert len(names) == len(spike_trains)
            name = names[idx]
        else:
            name = 'idx {}'.format(idx)
        return name
    if fig is None:
        fig = plt.figure()

    nrc = len(spike_trains)
    if gs is None:
        gs0 = gridspec.GridSpec(nrc, nrc)
    else:
        gs0 = gridspec.GridSpecFromSubplotSpec(nrc, nrc, subplot_spec=gs)
    axs, cnt = [], 0
    for x in range(nrc):
        for y in range(nrc):
            if (y > x) or (y == x):
                if not acorr and y == x:
                    continue
                prev_ax = None if len(axs) == 0 else axs[cnt-1]
                ax = fig.add_subplot(gs0[x, y], sharex=prev_ax, sharey=prev_ax)
                axs.append(ax)
            if y > x:
                plt.setp(ax.get_xticklabels(), visible=xticksvisible)
                plt.setp(ax.get_yticklabels(), visible=yticksvisible)
    cnt = 0
    ccgs = []
    for x in range(nrc):
        for y in range(nrc):
            if y > x:
                sptr1 = spike_trains[x]
                sptr2 = spike_trains[y]

                count, bins = correlogram(
                    t1=sptr1,
                    t2=sptr2,
                    binsize=binsize, limit=time_limit,  auto=False,
                    density=density)
                ccgs.append(count)
                if split_colors:
                    c1, c2 = colors[x], colors[y]
                    e1, e2 = edgecolors[x], edgecolors[y]
                    c1_n = sum(bins <= 0)
                    c2_n = len(bins) - c1_n
                    cs = [c1] * c1_n + [c2] * c2_n
                    es = [e1] * c1_n + [e2] * c2_n
                else:
                    cs, es = xcolor, xedgecolor
                axs[cnt].bar(bins, count, align='edge',
                             width=-binsize, color=cs,
                             edgecolor=es)
                axs[cnt].set_xlim([-time_limit, time_limit])
                name1 = get_name(sptr1, x)
                name2 = get_name(sptr2, y)
                axs[cnt].set_xlabel(name1 + ' ' + name2)
                cnt += 1
            elif y == x and acorr:
                sptr = spike_trains[x]
                count, bins = correlogram(
                    t1=sptr, t2=None,
                    binsize=binsize, limit=time_limit,
                    auto=True, density=density)
                ccgs.append(count)
                axs[cnt].bar(bins, count, width=-binsize, align='edge',
                             color=colors[x], edgecolor=edgecolors[x])
                axs[cnt].set_xlim([-time_limit, time_limit])
                name = get_name(sptr, x)
                axs[cnt].set_xlabel(name + ' ' + name)
                cnt += 1
    if ylim is not None: axs[0].set_ylim(ylim)
    plt.tight_layout()
    return ccgs, bins, axs


def post_stim_spikes(x, stim_times, mu, sigma):
    """Makes binary classification of response in windows"""
    stim_times = stim_times.astype(float)

    src_x = np.searchsorted(x, stim_times, side='right')

    remove_idxs, = np.where((src_x==len(x)) | (src_x==0))
    src_x = np.delete(src_x, remove_idxs)
    stim_times = np.delete(stim_times, remove_idxs)
    X = x[src_x] - stim_times
    return ((X > mu - sigma) & (X < mu + sigma)).astype(int), stim_times


class response_plotter:
    def __init__(self, source, target, stim_times, x_mu, x_sigma, y_mu, y_sigma):
        '''
        Parameters
        ----------
        source : array
            putative sender neuron
        target : array
            putative receiver neuron
        stim_times : array
            stimulation times
        x_mu : float
            Average stimulus response time for upstream spikes (y)
        y_sigma : float
            Standard deviation of upstream stimulus response times.
        y_mu : float
            Average stimulus response time for downstream spikes (y)
        y_sigma : float
            Standard deviation of downstream stimulus response times.
        '''
        self.x_mu = x_mu
        self.x_sigma = x_sigma
        self.y_mu = y_mu
        self.y_sigma = y_sigma
        self.period = np.min(np.diff(stim_times))
        self.source = source
        self.target = target
        self.responses, self.stim_times = post_stim_spikes(source, stim_times, x_mu, x_sigma)
        self.z0 = self.responses == 0
        self.z1 = self.responses == 1


    def trials(self, node):
        assert node in ['source', 'target']
        if not hasattr(self, '_trials_' + node):
            sptr = getattr(self, node)
            result = [
                sptr[(sptr > t) & (sptr  <= t + self.period)] - t
                for t in self.stim_times]
            setattr(self, '_trials_' + node, result)
        return getattr(self, '_trials_' + node)

    def plot(self, node, idxs=None, gs=None, **kwargs):
        binsize = 1e-3
        bins = np.arange(0, self.period + binsize, binsize)
        trials = self.trials(node)
        idxs = np.ones(len(trials)).astype(bool) if idxs is None else idxs
        assert idxs.dtype == bool
        assert len(idxs) == len(trials), (len(idxs), len(trials))
        trials = [t for t, i in zip(trials, idxs) if i]
        trial_num = np.arange(len(trials))
        ids = [np.ones(len(t)) * idx for idx, t in zip(trial_num, trials)]
        times = [t for trial in trials for t in trial]
        nums = [i for ii in ids for i in ii]

        if gs is None:
            fig = plt.figure()
            gs = GridSpec(2, 1, hspace=0.05)
        else:
            fig = plt.gcf()
            gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs, hspace=0.05)
        ax2 = fig.add_subplot(gs[1, 0])
        ax1 = fig.add_subplot(gs[0, 0], sharex=ax2)

        ax2.scatter(times, nums, color='k', s=1)
        ax1.hist(times, bins=bins, width=binsize)

        sns.despine()
        sns.despine(ax=ax1, bottom=True)
        plt.setp(ax1.get_xticklabels(), visible=False)

        for ax in (ax1, ax2):
            if node == 'target':
                ax.axvspan(self.y_mu - self.y_sigma, self.y_mu + self.y_sigma,
                           color='r', alpha=.5)
            if node == 'source':
                ax.axvspan(self.x_mu - self.x_sigma, self.x_mu + self.x_sigma,
                           color='cyan', alpha=.5)
        return ax1, ax2


def regplot(x, y, data, model=None, ci=95., scatter_color=None, model_color='k', ax=None,
            scatter_kws={}, regplot_kws={}, cmap=None, cax=None, clabel=None,
            xlabel=True, ylabel=True, colorbar=False, **kwargs):
    if model is None:
        import statsmodels.api as sm
        model = sm.OLS
    from seaborn import utils
    from seaborn import algorithms as algo
    if ax is None:
        fig, ax = plt.subplots()
    _x = data[x]
    _y = data[y]
    grid = np.linspace(_x.min(), _x.max(), 100)

    X = np.c_[np.ones(len(_x)), _x]
    G = np.c_[np.ones(len(grid)), grid]

    results = model(_y, X, **kwargs).fit()

    def reg_func(xx, yy):
        yhat = model(yy, xx, **kwargs).fit().predict(G)
        return yhat
    yhat = results.predict(G)
    yhat_boots = algo.bootstrap(
        X, _y, func=reg_func, n_boot=1000, units=None)
    err_bands = utils.ci(yhat_boots, ci, axis=0)
    ax.plot(grid, yhat, color=model_color, **regplot_kws)
    sc = ax.scatter(_x, _y, c=scatter_color, **scatter_kws)
    ax.fill_between(grid, *err_bands, facecolor=model_color, alpha=.15)
    if colorbar:
        cb = plt.colorbar(mappable=sc, cax=cax, ax=ax)
        cb.ax.yaxis.set_ticks_position('right')
        if clabel: cb.set_label(clabel)

    if xlabel:
        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
    if ylabel:
        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(y)
    return results


def scatterplot(x, y, data, scatter_color=None, ax=None,
            cmap=None, cax=None, clabel=None,
            xlabel=True, ylabel=True, colorbar=False, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()
    _x = data[x]
    _y = data[y]

    sc = ax.scatter(_x, _y, c=scatter_color, **kwargs)
    if colorbar:
        cb = plt.colorbar(mappable=sc, cax=cax, ax=ax)
        cb.ax.yaxis.set_ticks_position('right')
        if clabel: cb.set_label(clabel)

    if xlabel:
        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
    if ylabel:
        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(y)
