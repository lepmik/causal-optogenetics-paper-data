import numpy as np
import neo
import quantities as pq
import pytest


def test_create_neo_structure(tmpdir):
    date = '00000000'
    f_clu1 = tmpdir.join(date + ".clu.1")
    clu1_ids = [3, 0, 0, 1, 2, 1]
    f_clu1.writelines([str(id) + '\n' for id in clu1_ids])
    f_clu2 = tmpdir.join(date + ".clu.2")
    clu2_ids = [4, 0, 1, 2, 5]
    f_clu2.writelines([str(id) + '\n' for id in clu2_ids])
    f_res1 = tmpdir.join(date + ".res.1")
    res1_ts = [10, 100, 1000, 10000]
    f_res1.writelines([str(t) + '\n' for t in res1_ts])
    f_res2 = tmpdir.join(date + ".res.2")
    res2_ts = [90, 900, 9000, 90000]
    f_res2.writelines([str(t) + '\n' for t in res2_ts])


def test_merge_epochs():
    from tools_experimentaldata import merge_epochs
    ts = np.array([1, 2, 3]) * pq.s
    durations = np.array([11, 12, 13])*pq.s
    labels = np.array(['asdf', 'asdf', 'qwer'])
    prop_a = np.array([10, 100, 1000])
    prop_b = np.array([33, 66, 99])

    epc1 = neo.Epoch(
        times=ts,
        durations=durations,
        labels=labels,
        prop_a=prop_a,
        prop_b=prop_b)

    ts = np.array([-2, -1, 6]) * pq.s
    durations = np.array([-11, -12, -13])*pq.s
    labels = np.array(['asdf', 'qwer', 'qwer'])
    prop_a = np.array([-10, -100, -1000])
    prop_b = np.array([-33, -66, -99])

    epc2 = neo.Epoch(
        times=ts,
        durations=durations,
        labels=labels,
        prop_a=prop_a,
        prop_b=prop_b)
    epc_m = merge_epochs([epc1, epc2])

    assert np.array_equal(
        epc_m.times, np.array([-2, -1, 1, 2, 3, 6]) * pq.s)
    assert np.array_equal(
        epc_m.durations, np.array([-11, -12, 11, 12, 13, -13]) * pq.s)
    assert np.array_equal(
        epc_m.labels, np.array(
            ['asdf', 'qwer', 'asdf', 'asdf', 'qwer', 'qwer']))
    assert np.array_equal(
        epc_m.annotations['prop_a'], np.array(
            [-10, -100, 10, 100, 1000, -1000]))
    assert np.array_equal(
        epc_m.annotations['prop_b'], np.array(
            [-33, -66, 33, 66, 99, -99]))
    

    

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


def test_spktr_multislice():
    from tools_experimentaldata import\
        spktr_multislice

    spk_ts = np.arange(0, 10, 1)
    spktr = neo.SpikeTrain(spk_ts*pq.s,
                           t_stop=np.max(spk_ts)*pq.s)

    starts = np.array([3, 7])
    stops = np.array([2.4, 1.99])

    spkts_out = np.array([3., 4., 5., 7, 8.])*pq.s
    spktr_out = spktr_multislice(spktr, starts, stops)
    assert np.array_equal(spkts_out, spktr_out.times)
    

if __name__ == "__main__":
    test_multislice()
    test_n_events_in_multislice()
    test_determine_if_spktr_is_tagged()
    test_multislice_quantities()
