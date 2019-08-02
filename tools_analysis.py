import quantities as pq
import numpy as np
import neo
import pdb
from causal_optoconnectics.cch import histogram


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
    mat = [histogram(t, bins=bins)[0] for t in [s1, s2]]
    return corr(*mat)


def corr(a, b):
    mat = [(m - m.mean()) / m.std() for m in [a, b]]
    return cov(*mat)


def cov(a, b):
    return np.mean((a - a.mean())*(b - b.mean()))


def make_spiketrain_trials(spike_train, epoch, t_start=None, t_stop=None,
                           dim=None):
    '''
    Makes trials based on an Epoch and given temporal bound

    Parameters
    ----------
    spike_train : neo.SpikeTrain, neo.Unit, numpy.array, quantities.Quantity
    epoch : neo.Epoch
    t_start : quantities.Quantity
        time before epochs, default is 0 s
    t_stop : quantities.Quantity
        time after epochs default is duration of epoch
    dim : str
        if spike_train is numpy.array, the unit must be provided, e.g. "s"

    Returns
    -------
    out : list of neo.SpikeTrains
    '''

    if isinstance(spike_train, neo.Unit):
        sptr = []
        dim = unit.spiketrains[0].times.dimensionality
        unit = unit.spiketrains[0].times.units
        for st in unit.spiketrains:
            sptr.append(spike_train.rescale(dim).magnitude)
        sptr = np.sort(sptr) * unit
    elif isinstance(spike_train, neo.SpikeTrain):
        sptr = spike_train.times
        dim = sptr.dimensionality
        unit = sptr.units
    elif isinstance(spike_train, pq.Quantity):
        assert is_quantities(spike_train, 'vector')
        sptr = spike_train
        dim = sptr.dimensionality
        unit = sptr.units
    elif isinstance(spike_train, np.array):
        sptr = spike_train * pq.Quantity(1, unit)
        dim = sptr.dimensionality
        unit = sptr.units
    else:
        raise TypeError('Expected (neo.Unit, neo.SpikeTrain, ' +
                        'quantities.Quantity, numpy.array), got "' +
                        str(type(spike_train)) + '"')

    from neo.core import SpikeTrain
    if t_start is None:
        t_start = 0 * unit
    if t_start.ndim == 0:
        t_starts = t_start * np.ones(len(epoch.times))
    else:
        t_starts = t_start
        assert len(epoch.times) == len(t_starts), 'epoch.times and t_starts have different size'
    if t_stop is None:
        t_stop = epoch.durations
    if t_stop.ndim == 0:
        t_stops = t_stop * np.ones(len(epoch.times))
    else:
        t_stops = t_stop
        assert len(epoch.times) == len(t_stops), 'epoch.times and t_stops have different size'


    if not isinstance(epoch, neo.Epoch):
        raise TypeError('Expected "neo.Epoch" got "' + str(type(epoch)) + '"')

    trials = []
    for j, t in enumerate(epoch.times.rescale(dim)):
        t_start = t_starts[j].rescale(dim)
        t_stop = t_stops[j].rescale(dim)
        spikes = []
        for spike in sptr[(t+t_start < sptr) & (sptr < t+t_stop)]:
            spikes.append(spike-t)
        trials.append(SpikeTrain(times=spikes * unit,
                                 t_start=t_start,
                                 t_stop=t_stop))
    return trials


def permutation_resampling(case, control, num_samples=10000, statistic=None):
    """
    Simulation-based statistical calculation of p-value that statistic for case
    is different from statistic for control under the null hypothesis that the
    groups are invariant under label permutation. That is, case and control is
    combined and shuffeled randomly `num_samples` times and given statistic is
    calculated after each shuffle. Given the observed differece as the absulete
    differece between the statistic of the case and control. Then the p-value is
    calculated as the number of occurences where the shuffled statistic is
    greater than the observed differece pluss the number of occurences where
    the shuffled statistic is less than the negative observed differece, divided
    by the number of shuffles.

    For example, in a case-control study, it can be used to find the p-value
    under the hypothesis that the mean of the case group is different from that
    of the control group, and we cannot use the t-test because the distributions
    are highly skewed.

    Adapted from http://people.duke.edu/~ccc14/pcfb/analysis.html

    Parameters
    ----------
    case : 1D array like
        Samples from the case study.
    control : 1D array like
        Samples from the control study.
    num_samples : int
        Number of permutations
    statistic : function(2darray, axis)
        The statistic function to compare case and control. Default is mean

    Returns
    -------
    pval : float
        The calculated p-value.
    observed_diff : float
        Absolute difference between statistic of `case` and statistic of
        `control`.
    diffs : list
        A list of length equal to `num_samples` with differences between
        statistic of permutated case and statistic of permutated control.

    Examples
    --------
    Make up some data

    >>> np.random.seed(12345)
    >>> case = [94, 38, 23, 197, 99, 16, 141]
    >>> control = [52, 10, 40, 104, 51, 27, 146, 30, 46]

    Find the p-value by permutation resampling

    >>> pval, observed_diff, diffs = permutation_resampling(
    ...     case, control, 10000, np.mean)

    .. plot::

        import matplotlib.pylab as plt
        import numpy as np
        from exana.statistics import permutation_resampling
        case = [94, 38, 23, 197, 99, 16, 141]
        control = [52, 10, 40, 104, 51, 27, 146, 30, 46]
        pval, observed_diff, diffs = permutation_resampling(
            case, control, 10000, np.mean)
        plt.title('Empirical null distribution for differences in mean')
        plt.hist(diffs, bins=100, histtype='step', normed=True)
        plt.axvline(observed_diff, c='red', label='diff')
        plt.axvline(-observed_diff, c='green', label='-diff')
        plt.text(60, 0.01, 'p = %.3f' % pval, fontsize=16)
        plt.legend()
        plt.show()

    """
    statistic = statistic or np.mean

    observed_diff = abs(statistic(case) - statistic(control))
    num_case = len(case)

    combined = np.concatenate([case, control])
    diffs = []
    for i in range(num_samples):
        xs = np.random.permutation(combined)
        diff = np.mean(xs[:num_case]) - np.mean(xs[num_case:])
        diffs.append(diff)

    pval = (np.sum(diffs > observed_diff) +
            np.sum(diffs < -observed_diff))/float(num_samples)
    return pval, observed_diff, diffs
