import numpy as np
import neo
import quantities as pq
import pytest
import pdb


def test_get_first_spikes():
    from tools_experimentaldata import get_first_spikes
    
    spk_ts = np.array([0.1, 0.2, 0.3, 0.4]) * pq.s
    spktr = neo.SpikeTrain(times=spk_ts, t_stop=0.5*pq.s)

    event_ts = np.array([0.15, 0.29, 0.4]) * pq.s
    event_dur = np.array([0.1]*len(event_ts))
    epc = neo.Epoch(times=event_ts, durations=event_dur)

    sol = np.array([0.05, 0.01, 0.]) * pq.s
    res = get_first_spikes(spktr, epc)

    np.testing.assert_almost_equal(res, sol, decimal=13)


def test_get_first_spikes_more_stim_than_spks():
    from tools_experimentaldata import get_first_spikes
    
    spk_ts = np.array([0.1, 0.2]) * pq.s
    spktr = neo.SpikeTrain(times=spk_ts, t_stop=0.5*pq.s)

    event_ts = np.array([0.15, 0.29, 0.4]) * pq.s
    event_dur = np.array([0.1]*len(event_ts))
    epc = neo.Epoch(times=event_ts, durations=event_dur)

    sol = np.array([0.05]) * pq.s
    res = get_first_spikes(spktr, epc)

    np.testing.assert_almost_equal(res, sol, decimal=13)


def test_get_pseudo_epoch():
    from tools_experimentaldata import get_pseudo_epoch

    ts = np.array([1, 10, 100, 1000])*pq.s
    dur = np.array([1, 1, 1, 1])*pq.s
    ts_dist_after = 5.*pq.s
    ts_dist_before = 5.*pq.s
    epc = neo.Epoch(ts, dur)

    epc_pseudo = get_pseudo_epoch(epc,
                                  ts_dist_after,
                                  ts_dist_before)

    assert len(epc_pseudo.times) == 2
    assert epc_pseudo.times[0] > ts[1]
    assert epc_pseudo.times[0] < ts[2]
    

def test_select_times_from_epoch():
    from tools_experimentaldata import select_times_from_epoch

    # general check of sorting
    ts = np.array([0.1, 1., 2., 5,]) * pq.s
    durations = np.array([0.01, 0.5, 2.5, 200])*pq.s
    labels = np.array(['pulse', 'pulse', 'pulse', 'sine'])
    prop_a = np.array([10, 100, 1000, 10000])
    prop_b = np.array([600, 300, 100, 1])

    epc = neo.Epoch(times=ts,
                    durations=durations,
                    labels=labels,
                    prop_a=prop_a,
                    prop_b=prop_b)

    conditions = {'durations': [0., 0.6] * pq.s,
                  'labels': 'pulse',
                  'annotations.prop_a': [6, 1000],
                  'annotations.prop_b': 600}
    epc_sel = select_times_from_epoch(epc, conditions)
    assert epc_sel.times == np.array([0.1]) * pq.s

    conditions = {'durations': [0., 0.6],
                  'labels': 'pulse',
                  'annotations.prop_a': [6, 1000],
                  'annotations.prob_b': 600}

    # check whether value check works
    with pytest.raises(ValueError):
        epc_sel = select_times_from_epoch(epc, conditions)


def test_multislice():
    from tools_experimentaldata import multislice
    x = np.array([0, 1, 2, 6, 7, 8])

    starts = np.array([0.5, 5.5])
    stops = np.array([2.5, 6.5])

    x_sliced = multislice(x, starts, stops)
    assert np.array_equal(x_sliced,  np.array([1, 2, 6]))

    
def test_multislice_quantities():
    from tools_experimentaldata import multislice
    x = np.array([0, 1, 2, 6, 7, 8])*pq.s

    starts = np.array([0.5, 5.5])*pq.s.rescale(pq.ms)
    stops = np.array([2.5, 6.5])*pq.s.rescale(pq.ms)

    x_sliced = multislice(x, starts, stops)
    assert np.array_equal(x_sliced,  np.array([1, 2, 6]))
    

def test_n_events_in_multislice():
    from tools_experimentaldata import n_events_in_multislice
    x = np.array([0, 1, 2, 6, 7, 8])

    starts = np.array([0.5, 5.5, 7.5])
    stops = np.array([2.5, 6.5, 7.7])

    nevents = n_events_in_multislice(x, starts, stops)
    assert np.array_equal(nevents,  np.array([2, 1, 0]))


def test_determine_if_spktr_is_tagged():
    from tools_experimentaldata import\
        determine_if_spktr_is_tagged as def_tag
    ls_spk_ts = [np.arange(0, 200, 0.1)]
    stim_ts = np.arange(2, 200, 3)
    stim_dur = 0.1
    for t_i in stim_ts:
        ls_spk_ts.append(np.arange(t_i, t_i+stim_dur, 0.0001))
    spk_ts = np.concatenate(ls_spk_ts)
    spk_ts = np.sort(spk_ts)
    spktr = neo.SpikeTrain(spk_ts*pq.s,
                           t_stop=np.max(spk_ts)*pq.s)
    epc = neo.Epoch(stim_ts*pq.s,
                    stim_dur*pq.s)
    res = def_tag(spktr, epc)
    assert res is True

    
def test_keep_spikes_by_stim():
    from tools_experimentaldata import\
        keep_spikes_by_stim as keep
    blk = neo.Block()
    seg = neo.Segment()
    blk.segments.append(seg)
    unit = neo.Unit()
    chx_units = neo.ChannelIndex(index=0)
    chx_units.units.append(unit)
    blk.channel_indexes.append(chx_units)
    spkts = np.array([1, 2, 3, 4, 5, 6])*pq.s

    spktr = neo.SpikeTrain(spkts,
                           t_start=0*pq.s,
                           t_stop=8*pq.s)
    unit.spiketrains.append(spktr)
    
    stimts = np.array([1.5, 4.2])*pq.s
    stimdurs = np.array([2., 1.])*pq.s
    epc = neo.Epoch(times=stimts,
                    durations=stimdurs)
    seg.epochs.append(epc)
    
    blks = [blk]
    blks_stim = keep(blks, 'stim')
    blks_nostim = keep(blks, 'nostim')

    # check stim
    blk_stim = blks_stim[0]
    unit = blk_stim.channel_indexes[0].children[0]
    spktr = unit.spiketrains[0]
    spkts = spktr.times
    assert np.array_equal(spkts,
                          np.array([2., 3., 5.]) * pq.s)

    # check nostim
    blk_nostim = blks_nostim[0]
    unit = blk_nostim.channel_indexes[0].children[0]
    spktr = unit.spiketrains[0]
    spkts = spktr.times
    assert np.array_equal(spkts,
                          np.array([1., 4., 6.])*pq.s)

def test_group_stimulations():
    from tools_experimentaldata import\
        group_stimulations
    blk = neo.Block()
    seg = neo.Segment()
    blk.segments.append(seg)
    n_stim1 = 30
    n_stim2 = 50
    intens = np.concatenate([
        np.linspace(0, 30, n_stim1),
        np.linspace(200, 250, n_stim2)])
    ts = np.arange(0, len(intens), 1)*pq.s
    epc = neo.Epoch(times=ts,
                    durations=np.ones(len(ts))*pq.s,
                    intensity=intens)
    seg.epochs.append(epc)

    blks = [blk]
    bins = np.arange(0, 300, 1)
    kernel_width = 5
    threshold = 0.
    group_stimulations(blks,
                       bins,
                       kernel_width,
                       threshold)
    assert np.sum(epc.annotations['stim_group'] == 1) == n_stim1
    assert np.sum(epc.annotations['stim_group'] == 2) == n_stim2
    
if __name__ == "__main__":
    test_get_first_spikes()
    test_get_first_spikes_more_stim_than_spks()
    test_get_pseudo_epoch()
    test_select_times_from_epoch()
    test_multislice()
    test_n_events_in_multislice()
    test_determine_if_spktr_is_tagged()
    test_keep_spikes_by_stim()
    test_multislice_quantities()

