import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import quantities as pq
from tools_analysis import make_spiketrain_trials
import neo
import os.path as op


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
            xticks=True, yticks=True, all_sides=False):
    """
    Removes axis lines
    """
    if all_sides:
        left, right, top, bottom = [True] * 4
    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, (list, tuple)):
        ax = [ax]
    for a in ax:
        try:
            a.spines['top'].set_visible(not top)
            a.spines['right'].set_visible(not right)
            a.spines['left'].set_visible(not left)
            a.spines['bottom'].set_visible(not bottom)
        except AttributeError:
            raise
        except:
            raise
        if not xticks:
            a.get_xaxis().tick_bottom()
            plt.setp(a.get_xticklabels(), visible=False)
        if not yticks:
            a.get_yaxis().tick_left()
            plt.setp(a.get_yticklabels(), visible=False)


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
            'figure.figsize' : (16, 9),
            'axes.labelsize' : 50,
            'lines.linewidth': 4,
            'lines.markersize': 20,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'axes.titlesize' : 20,
            'font.size'      : 20,
            'legend.frameon' : False,
            'legend.fontsize': 35,
            'font.family'    : 'serif',
            'text.usetex'    : True
        }
    }
    rc = sdict[style]
    plt.rcParams.update(rc)
    sns.set(rc=rc, style=sns_style,
            color_codes=True)


def add_caption(axs, start='a', vspace=1, **kw):
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
    for ax, capt in zip(axs, alph):
        label = ax.get_xlabel()
        ax.set_xlabel(
            r'\begin{center}{}\\[{}ex]({})\end{center}'.format(label, vspace, capt),
            **kw)


def plot_psth(spike_train=None, epoch=None, trials=None, xlim=[None, None],
              fig=None, axs=None, legend_loc=1, color='b',
              title='', stim_alpha=.2, stim_color=None,
              stim_label='Stim on', stim_style='patch', stim_offset=0*pq.s,
              rast_ylabel='Trials', rast_size=10.,
              hist_color=None, hist_edgecolor=None, hist_ylim=None,
              hist_ylabel=None, hist_output='counts', hist_binsize=None,
              hist_nbins=100, hist_alpha=1.):
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
    title : figure title
    stim_alpha : float
    stim_color : str
    stim_label : str
    stim_style : 'patch' or 'line'
    stim_offset : pq.Quantity
        The amount of offset for the stimulus relative to epoch.
    rast_ylabel : str
    rast_size : float
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
    if stim_style == 'patch':
        if epoch is not None:
            stim_duration = epoch.durations.rescale(dim).magnitude.max()
        else:
            warnings.warn('Unable to acquire stimulus duration, setting ' +
                          'stim_style to "line". Please provede "epoch"' +
                          ' in order to use stim_style "patch".')
            stim_style = 'line'
    # raster
    plot_raster(trials, color=color, ax=rast_ax, ylabel=rast_ylabel,
                marker_size=rast_size)
    # histogram
    hist_color = color if hist_color is None else hist_color
    hist_ylabel = hist_output if hist_ylabel is None else hist_ylabel
    plot_spike_histogram(trials, color=hist_color, ax=hist_ax,
                         output=hist_output, binsize=hist_binsize,
                         nbins=hist_nbins, edgecolor=hist_edgecolor,
                         ylabel=hist_ylabel, alpha=hist_alpha)
    if hist_ylim is not None: hist_ax.set_ylim(hist_ylim)
    # stim representation
    stim_color = color if stim_color is None else stim_color
    if stim_style == 'patch':
        fill_stop = stim_duration
        import matplotlib.patches as mpatches
        line = mpatches.Patch([], [], color=stim_color, label=stim_label,
                              alpha=stim_alpha)
    elif stim_style == 'line':
        fill_stop = 0
        import matplotlib.lines as mlines
        line = mlines.Line2D([], [], color=stim_color, label=stim_label)
    stim_offset = stim_offset.rescale('s').magnitude
    hist_ax.axvspan(stim_offset, fill_stop + stim_offset, color=stim_color,
                    alpha=stim_alpha, zorder=0)
    rast_ax.axvspan(stim_offset, fill_stop + stim_offset, color=stim_color,
                    alpha=stim_alpha, zorder=0)
    if legend_loc == 'outside':
        hist_ax.legend(handles=[line], bbox_to_anchor=(0., 1.02, 1., .102),
                       loc=4, ncol=2, borderaxespad=0.)
    else:
        hist_ax.legend(handles=[line], loc=legend_loc, ncol=2, borderaxespad=0.)
    if title is not None: hist_ax.set_title(title)
    return fig


def plot_spike_histogram(trials, color='b', ax=None, binsize=None, bins=None,
                         output='counts', edgecolor=None, alpha=1., ylabel=None,
                         nbins=None):
    """
    Histogram plot of trials

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
    ax.bar(bs[:len(time_hist)], time_hist.magnitude.flatten(), width=bs[1]-bs[0],
           edgecolor=edgecolor, facecolor=color, alpha=alpha)
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
