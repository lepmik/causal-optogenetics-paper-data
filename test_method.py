import pytest
import numpy as np
import pdb


def test_hist_stim():
    # make sure that the vectorized function
    # does exaclty the same as the loop version
    from method import hist_stim
    rand = np.random.RandomState(10)
    stim_times = np.sort(rand.uniform(0, 100., 1000))
    sender = np.sort(rand.uniform(0, 100., 100))
    target = np.sort(rand.uniform(0, 100., 100))
    winsize = 0.4,
    latency = 0.1

    hist_ret = hist_stim(stim_times,
                         sender,
                         target,
                         winsize,
                         latency)

    hist = np.zeros([len(stim_times), 2])
    src = np.searchsorted
    for idx, t in enumerate(stim_times):
        hist[idx, :] = (
          # stim response
          src(sender, t, 'left') <
          src(sender, t + winsize, 'right'),
          # stim synaptic response
          src(target, t + latency, 'left') <
          src(target, t + latency + winsize, 'right')
          )
    try:
        assert np.array_equal(hist, hist_ret)
    except:
        pdb.set_trace()


if __name__ == "__main__":
    test_hist_stim()
