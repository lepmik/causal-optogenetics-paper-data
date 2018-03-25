import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

def coef_var(spike_trains):
    """
    Calculate the coefficient of variation in inter spike interval (ISI)
    distribution over several spike_trains

    Parameters
    ----------
    spike_trains : list
        contains spike trains where each spike train is an array of spike times

    Returns
    -------
    out : list
        Coefficient of variations for each spike_train, nan if len(spike_train) == 0

    Examples
    --------
    >>> np.random.seed(12345)
    >>> spike_trains = [np.arange(10), np.random.random((10))]
    >>> print('{d[0]:.2f}, {d[1]:.2f}'.format(d=coeff_var(spike_trains)))
    0.00, -9.53

    """
    cvs = []
    for spike_train in spike_trains:
        isi = np.diff(spike_train)
        if len(isi) > 0:
            cvs.append(np.std(isi) / np.mean(isi))
        else:
            cvs.append(np.nan)
    return cvs


def corrcoef(spike_trains, t_stop, binsize=10.):
    """
    Calculate the pairwise Pearson correlation coefficients

    Parameters
    ----------
    spike_trains : list
        contains spike trains where each spike train is an array of spike times
    t_stop : float
        stop time of spike trains
    binsize : float
        binsize for histograms to be correlated


    Returns
    -------
    out : array
        Pearson correlation coefficients
    """
    N = len(spike_trains)
    cmat = [corr_spikes(m1, m2, t_stop, binsize)
            for m1 in spike_trains for m2 in spike_trains]
    return np.array(cmat).reshape((N, N))

def corr_spikes(s1, s2, t_stop, binsize):
    bins = np.arange(0, t_stop + binsize, binsize)
    mat = [np.histogram(t, bins=bins)[0] for t in [s1, s2]]
    return corr(*mat)

def corr(a, b):
    mat = [(m - m.mean()) / m.std() for m in [a, b]]
    return cov(*mat)

def cov(a, b):
    return np.mean((a - a.mean())*(b - b.mean()))

def csv_append_dict(fname, dictionary):
    assert isinstance(dictionary, dict)
    df = pd.DataFrame([dictionary])
    if not fname.endswith('.csv'):
        fname = fname + '.csv'
    if not op.exists(fname):
        df.to_csv(fname, index=False, mode='w', header=True)
    else:
        df.to_csv(fname, index=False, mode='a', header=False)


def despine(ax=None, left=False, right=True, top=True, bottom=False,
            xticks=True, yticks=True, all_sides=False):
    """
    Removes axis lines
    """
    if all_sides:
        left, right, top, bottom = [True] * 4
    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, list):
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
            'axes.labelsize' : 25,
            'lines.linewidth': 2,
            'xtick.labelsize': 25,
            'ytick.labelsize': 25,
            'axes.titlesize' : 20,
            'font.size'      : 20,
            'legend.frameon' : False,
            'legend.fontsize': 20,
            'font.family'    : 'serif',
            'text.usetex'    : True
        }
    }
    rc = sdict[style]
    plt.rcParams.update(rc)
    sns.set(rc=rc, style=sns_style,
            color_codes=True)
