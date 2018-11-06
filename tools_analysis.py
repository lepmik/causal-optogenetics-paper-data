import quantities as pq
import numpy as np
import neo
import pdb


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


def poisson_continuity_correction(n, observed):
    """
    n : array
        Likelihood to observe n or more events
    observed : array
        Rate of Poisson process
    References
    ----------
    Stark, E., & Abeles, M. (2009). Unbiased estimation of precise temporal
    correlations between spike trains. Journal of neuroscience methods, 179(1),
    90-100.
    """
    from scipy.stats import poisson
    if n.ndim == 0:
        n = np.array([n])
    assert n.ndim == 1
    assert np.all(n >= 0)
    n = n.astype(int)
    result = np.zeros(n.shape)
    if n.shape != observed.shape:
        observed = np.repeat(observed, n.size)
    for i, (n_i, rate) in enumerate(zip(n, observed)):
        if n_i == 0:
            result[i] = 1.
        else:
            rates = [poisson.pmf(j, rate) for j in range(n_i)]
            result[i] = 1 - np.sum(rates) - 0.5 * poisson.pmf(n_i, rate)
    return result


def hollow_kernel(kernlen, width, hollow_fraction=0.6, kerntype='gaussian'):
    '''
    Returns a hollow kernel normalized to it's sum
    Parameters
    ----------
    kernlen : int
        Length of kernel, must be uneven (kernlen % 2 == 1)
    width : float
        Width of kernel (std if gaussian)
    hollow_fraction : float
        Fractoin of the central bin to removed.
    Returns
    -------
    kernel : array
    '''
    if kerntype == 'gaussian':
        from scipy.signal import gaussian
        assert kernlen % 2 == 1
        kernel = gaussian(kernlen, width)
        kernel[int(kernlen / 2.)] *= (1 - hollow_fraction)
    else:
        raise NotImplementedError
    return kernel / sum(kernel)


def cch_convolve(cch, width, hollow_fraction, kerntype):
    import scipy.signal as scs
    kernlen = len(cch) - 1
    kernel = hollow_kernel(kernlen, width, hollow_fraction, kerntype)
    # padd edges
    len_padd = int(kernlen / 2.)
    cch_padded = np.zeros(len(cch) + 2 * len_padd)
    # "firstW/2 bins (excluding the very first bin) are duplicated,
    # reversed in time, and prepended to the cch prior to convolving"
    cch_padded[0:len_padd] = cch[1:len_padd+1][::-1]
    cch_padded[len_padd: - len_padd] = cch
    # # "Likewise, the lastW/2 bins aresymmetrically appended to the cch."
    cch_padded[-len_padd:] = cch[-len_padd-1:-1][::-1]
    # convolve cch with kernel
    result = scs.fftconvolve(cch_padded, kernel, mode='valid')
    assert len(cch) == len(result)
    return result


def cch_significance(t1, t2, binsize, limit, hollow_fraction, width,
                     kerntype='gaussian'):
    """
    Parameters
    ---------
    t1 : np.array, or neo.SpikeTrain
        First spiketrain, raw spike times in seconds.
    t2 : np.array, or neo.SpikeTrain
        Second spiketrain, raw spike times in seconds.
    binsize : float, or quantities.Quantity
        Width of each bar in histogram in seconds.
    limit : float, or quantities.Quantity
        Positive and negative extent of histogram, in seconds.
    kernlen : int
        Length of kernel, must be uneven (kernlen % 2 == 1)
    width : float
        Width of kernel (std if gaussian)
    hollow_fraction : float
        Fraction of the central bin to removed.
    References
    ----------
    Stark, E., & Abeles, M. (2009). Unbiased estimation of precise temporal
    correlations between spike trains. Journal of neuroscience methods, 179(1),
    90-100.
    English et al. 2017, Neuron, Pyramidal Cell-Interneuron Circuit Architecture
    and Dynamics in Hippocampal Networks
    """
    cch, bins = correlogram(t1, t2, binsize=binsize, limit=limit,
                            density=False)
    pfast = np.zeros(cch.shape)
    cch_smooth = cch_convolve(cch=cch, width=width,
                              hollow_fraction=hollow_fraction,
                              kerntype=kerntype)
    pfast = poisson_continuity_correction(cch, cch_smooth)
    # ppeak describes the probability of obtaining a peak with positive lag
    # of the histogram, that is signficantly larger than the largest peak
    # in the negative lag direction.
    ppeak = np.zeros(cch.shape)
    max_vals = np.zeros(cch.shape)
    cch_half_len = int(np.floor(len(cch) / 2.))
    max_vals[cch_half_len:] = np.max(cch[:cch_half_len])
    max_vals[:cch_half_len] = np.max(cch[cch_half_len:])
    ppeak = poisson_continuity_correction(cch, max_vals)
    return ppeak, pfast, bins, cch, cch_smooth


def transfer_probability(t1, t2, binsize, limit, hollow_fraction, width,
                         latency, winsize, kerntype='gaussian'):
    cch, bins = correlogram(t1, t2, binsize=binsize, limit=limit,
                            density=False)
    cch_s = cch_convolve(cch=cch, width=width,
                              hollow_fraction=hollow_fraction,
                              kerntype=kerntype)

    mask = (bins >= latency) & (bins <= latency + winsize)
    cmax = np.max(cch[mask])
    idx, = np.where(cch==cmax * mask)
    idx = idx if len(idx) == 1 else idx[0]
    pfast, = poisson_continuity_correction(cmax, cch_s[idx])
    cch_half_len = int(np.floor(len(cch) / 2.))
    max_pre = np.max(cch[:cch_half_len])
    ppeak, = poisson_continuity_correction(cmax, max_pre)
    ptime = float(bins[idx])
    trans_prob = sum(cch[mask] - cch_s[mask]) / len(t1)
    return trans_prob, ppeak, pfast, ptime, cmax


def correlogram(t1, t2=None, binsize=.001, limit=.02, auto=False,
                density=False):
    """Return crosscorrelogram of two spike trains.
    Essentially, this algorithm subtracts each spike time in `t1`
    from all of `t2` and bins the results with np.histogram, though
    several tweaks were made for efficiency.
    Originally authored by Chris Rodger, copied from OpenElectrophy, licenced
    with CeCill-B. Examples and testing written by exana team.
    Parameters
    ---------
    t1 : np.array, or neo.SpikeTrain
        First spiketrain, raw spike times in seconds.
    t2 : np.array, or neo.SpikeTrain
        Second spiketrain, raw spike times in seconds.
    binsize : float, or quantities.Quantity
        Width of each bar in histogram in seconds.
    limit : float, or quantities.Quantity
        Positive and negative extent of histogram, in seconds.
    auto : bool
        If True, then returns autocorrelogram of `t1` and in
        this case `t2` can be None. Default is False.
    density : bool
        If True, then returns the probability density function.
    See also
    --------
    :func:`numpy.histogram` : The histogram function in use.
    Returns
    -------
    (count, bins) : tuple
        A tuple containing the bin right edges and the
        count/density of spikes in each bin.
    Note
    ----
    `bins` are relative to `t1`. That is, if `t1` leads `t2`, then
    `count` will peak in a positive time bin.
    Examples
    --------
    >>> t1 = np.arange(0, .5, .1)
    >>> t2 = np.arange(0.1, .6, .1)
    >>> limit = 1
    >>> binsize = .1
    >>> counts, bins = correlogram(t1=t1, t2=t2, binsize=binsize,
    ...                            limit=limit, auto=False)
    >>> counts
    array([0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0])
    The interpretation of this result is that there are 5 occurences where
    in the bin 0 to 0.1, i.e.
    >>> idx = np.argmax(counts)
    >>> '%.1f, %.1f' % (abs(bins[idx - 1]), bins[idx])
    '0.0, 0.1'
    The correlogram algorithm is identical to, but computationally faster than
    the histogram of differences of each timepoint, i.e.
    >>> diff = [t2 - t for t in t1]
    >>> counts2, bins = np.histogram(diff, bins=bins)
    >>> np.array_equal(counts2, counts)
    True
    """
    if auto: t2 = t1
    lot = [t1, t2, limit, binsize]
    if any(isinstance(a, pq.Quantity) for a in lot):
        if not all(isinstance(a, pq.Quantity) for a in lot):
            raise ValueError('If any is quantity all must be ' +
                             '{}'.format([type(d) for d in lot]))
        dim = t1.dimensionality
        t1, t2, limit, binsize = [a.rescale(dim).magnitude for a in lot]
    # For auto-CCGs, make sure we use the same exact values
    # Otherwise numerical issues may arise when we compensate for zeros later
    if not int(limit * 1e10) % int(binsize * 1e10) == 0:
        raise ValueError('Time limit {} must be a '.format(limit) +
                         'multiple of binsize {}'.format(binsize) +
                         ' remainder = {}'.format(limit % binsize))
    # For efficiency, `t1` should be no longer than `t2`
    swap_args = False
    if len(t1) > len(t2):
        swap_args = True
        t1, t2 = t2, t1

    # Sort both arguments (this takes negligible time)
    t1 = np.sort(t1)
    t2 = np.sort(t2)

    # Determine the bin edges for the histogram
    # Later we will rely on the symmetry of `bins` for undoing `swap_args`
    limit = float(limit)

    # The numpy.arange method overshoots slightly the edges i.e. binsize + epsilon
    # which leads to inclusion of spikes falling on edges.
    bins = np.arange(-limit, limit + binsize, binsize)

    # Determine the indexes into `t2` that are relevant for each spike in `t1`
    ii2 = np.searchsorted(t2, t1 - limit)
    jj2 = np.searchsorted(t2, t1 + limit)

    # Concatenate the recentered spike times into a big array
    # We have excluded spikes outside of the histogram range to limit
    # memory use here.
    big = np.concatenate([t2[i:j] - t for t, i, j in zip(t1, ii2, jj2)])

    # Actually do the histogram. Note that calls to np.histogram are
    # expensive because it does not assume sorted data.
    count, bins = np.histogram(big, bins=bins, density=density)

    if auto:
        # Compensate for the peak at time zero that results in autocorrelations
        # by subtracting the total number of spikes from that bin. Note
        # possible numerical issue here because 0.0 may fall at a bin edge.
        c_temp, bins_temp = np.histogram([0.], bins=bins)
        bin_containing_zero = np.nonzero(c_temp)[0][0]
        count[bin_containing_zero] = 0#-= len(t1)

    # Finally compensate for the swapping of t1 and t2
    if swap_args:
        # Here we rely on being able to simply reverse `counts`. This is only
        # possible because of the way `bins` was defined (bins = -bins[::-1])
        count = count[::-1]

    return count, bins[1:]


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
