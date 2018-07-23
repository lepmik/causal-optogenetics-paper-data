import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.linear_model import LogisticRegression
from cross_correlation import correlogram, poisson_continuity_correction
import seaborn as sns

def hist_stim(stim_times, source, target, period, winsize, latency):
    hist = np.zeros([len(stim_times), 3])
    src = np.searchsorted
    for idx, t in enumerate(stim_times):
        hist[idx,:] = (
          # stim response
          src(source, t, 'left') <
          src(source, t + winsize, 'right'),
          # stim synaptic response
          src(target, t + latency, 'left') <
          src(target, t + latency + winsize, 'right'),
          # no stim response
          src(source, t - 2*winsize, 'left') <
          src(source, t - winsize, 'right')
          )
    return hist

class IV:
    def __init__(self, source, target, stim_times,
                 winsize, latency):
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
        self.period = np.min(np.diff(stim_times))
        self.stim_times = stim_times
        self.source = source
        self.target = target
        self.latency = latency
        self.winsize = winsize
        self.lams = np.array(
            hist_stim(stim_times, source, target,
                      self.period, self.winsize, self.latency))
        self.StimRef = self.lams[:, 0] == 0
        self.Stim = self.StimRef == False
        self.NoStimRef = self.lams[:, 2] == 0
        self.NoStim = self.NoStimRef == False

    @property
    def wald(self):
        ys = self.lams[self.Stim, 1]
        ysr = self.lams[self.StimRef, 1]
        return(ys.mean() - ysr.mean())

    @property
    def wald_ns(self):
        ys = self.lams[self.NoStim, 1]
        ysr = self.lams[self.NoStimRef, 1]
        return(ys.mean() - ysr.mean())

    @property
    def logreg(self):
        X, Y = self.lams[:, 0], self.lams[:, 1]
        lr = LogisticRegression()
        lr.fit(X.reshape(-1, 1).astype(int), Y.astype(int))
        return lr

    @property
    def logreg_ns(self):
        X, Y = self.lams[:-1, 2], self.lams[1:, 1]
        lr = LogisticRegression()
        lr.fit(X.reshape(-1, 1).astype(int), Y.astype(int))
        return lr

    @property
    def cch(self):
        return self.__cch__('Stim', 'StimRef')

    @property
    def cch_ns(self):
        return self.__cch__('NoStim', 'NoStimRef')

    def __cch__(self, hit_ref, miss_ref):
        '''
        hit_ref: e.g. "Stim"
        miss_ref: e.g. "StimRef"
        '''
        if hasattr(self, hit_ref + miss_ref):
            return getattr(self, hit_ref + miss_ref)
        binsize = 1.
        limit = np.ceil(self.latency + self.winsize)
        stim_hit = self.stim_times[getattr(self, hit_ref)]
        stim_miss = self.stim_times[getattr(self, miss_ref)]
        cch_hit, bins = correlogram(
            stim_hit, self.target,
            binsize=binsize, limit=limit, density=False)
        cch_miss, bins_ = correlogram(
            stim_miss, self.target,
            binsize=binsize, limit=limit, density=False)
        assert np.array_equal(bins, bins_)
        mask = (bins >= self.latency) & (bins <= self.latency + self.winsize)
        idx, = np.where(cch_hit==np.max(cch_hit[mask]) * mask)
        idx = idx if len(idx) == 1 else idx[0]
        ptime = float(bins[idx])
        pcausal = poisson_continuity_correction(cch_hit[idx], cch_miss[idx])
        trans_prob = sum(cch_hit[mask] / len(stim_hit) -
                         cch_miss[mask] / len(stim_miss))
        setattr(self, hit_ref + miss_ref, (trans_prob, pcausal, ptime))
        return trans_prob, pcausal, ptime

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
