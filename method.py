import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.linear_model import LogisticRegression
from cross_correlation import (correlogram, poisson_continuity_correction,
                               cch_convolve)
import seaborn as sns


def hist_stim(stim_times, source, target, winsize, latency):
    hist = np.zeros([len(stim_times), 2])
    src = np.searchsorted
    hist[:, 0] = (
        # stim response
        src(source, stim_times, 'left') <
        src(source, stim_times + winsize, 'right'))
    hist[:, 1] = (
        # stim synaptic response
        src(target, stim_times + latency, 'left') <
        src(target, stim_times + latency + winsize, 'right'))
    return hist


class IV:
    def __init__(self, source, target, stim_times,
                 winsize, latency, **parameters):
        '''
        Parameters
        ----------
        source : array
            putative sender neuron
        target : array
            putative receiver neuron
        stim_times : array
            stimulation times
        '''
        p = {
            'width': 10,
            'hollow_fraction': .6,
            'kerntype': 'gaussian'
        }
        if parameters:
            p.update(parameters)
        self.p = p
        self.period = np.min(np.diff(stim_times))
        self.stim_times = stim_times
        self.source = source
        self.target = target
        self.latency = latency
        self.winsize = winsize
        self.lams = np.array(
            hist_stim(stim_times, source, target,
                      self.winsize, self.latency))
        self.StimRef = self.lams[:, 0] == 0
        self.Stim = self.StimRef == False

    @property
    def wald(self):
        ys = self.lams[self.Stim, 1]
        ysr = self.lams[self.StimRef, 1]
        return(ys.mean() - ysr.mean())

    @property
    def logreg(self):
        X, Y = self.lams[:, 0], self.lams[:, 1]
        lr = LogisticRegression()
        lr.fit(X.reshape(-1, 1).astype(int), Y.astype(int))
        return lr

    @property
    def trans_prob(self):
        mask = ((self.cch['bins'] >= self.latency) &
                (self.cch['bins'] <= self.latency + self.winsize))
        trans_prob = sum(self.cch['cch_hit'][mask] / sum(self.Stim) -
                         self.cch['cch_miss'][mask] / sum(self.StimRef))
        return trans_prob

    @property
    def cch(self):
        if hasattr(self, '_cch'):
            return getattr(self, '_cch')
        binsize = 1.
        limit = np.ceil(self.latency + self.winsize) * 2
        stim_hit = self.stim_times[self.Stim]
        stim_miss = self.stim_times[self.StimRef]
        cch_hit, bins = correlogram(
            stim_hit, self.target,
            binsize=binsize, limit=limit, density=False)
        cch_miss, bins_ = correlogram(
            stim_miss, self.target,
            binsize=binsize, limit=limit, density=False)
        cch_diff = cch_hit - cch_miss
        assert np.array_equal(bins, bins_)
        result = {'cch_hit': cch_hit,
                  'cch_miss': cch_miss,
                  'cch_diff': cch_diff,
                  'bins': bins}
        setattr(self, '_cch', result)
        return result

    @property
    def prob(self):
        if hasattr(self, '_p'):
            return getattr(self, '_p')
        cch_s = cch_convolve(
            cch=self.cch['cch_diff'], width=self.p['width'],
            hollow_fraction=self.p['hollow_fraction'],
            kerntype=self.p['kerntype'])
        mask = ((self.cch['bins'] >= self.latency) &
                (self.cch['bins'] <= self.latency + self.winsize))
        hit_max = np.max(self.cch['cch_hit'][mask])
        idx, = np.where(self.cch['cch_hit']==hit_max * mask)
        idx = idx if len(idx) == 1 else idx[0]
        pfast, = poisson_continuity_correction(hit_max, cch_s[idx])
        cch_half_len = int(np.floor(len(self.cch['cch_diff']) / 2.))
        ppeak, = poisson_continuity_correction(
            np.max(self.cch['cch_diff'][:cch_half_len]), hit_max)
        ptime = float(self.cch['bins'][idx])
        pcausal, = poisson_continuity_correction(
            self.cch['cch_hit'][idx], self.cch['cch_miss'][idx])
        result = {'pcausal': pcausal,
                  'pfast': pfast,
                  'ppeak': ppeak,
                  'ptime': ptime}
        setattr(self, '_p', result)
        return result

    @property
    def hit_rate(self):
        return sum(self.Stim) / len(self.Stim)

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
        period = np.min(np.diff(self.stim_times))
        bins = np.arange(0, period + 1, 1)
        trials = self.trials(node)
        idxs = np.ones(len(trials)).astype(bool) if idxs is None else idxs
        assert idxs.dtype == bool
        assert len(idxs) == len(trials)
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
        ax1.hist(times, bins=bins, width=1)
        sns.despine()
        sns.despine(ax=ax1, bottom=True)
        plt.setp(ax1.get_xticklabels(), visible=False)
        for ax in (ax1, ax2):
            if node == 'target':
                ax.axvspan(self.latency, self.latency + self.winsize,
                           color='r', alpha=.5)
            if node == 'source':
                ax.axvspan(0, self.winsize,
                           color='cyan', alpha=.5)
        return ax1, ax2
