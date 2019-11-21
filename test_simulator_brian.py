import numpy as np
from simulator_brian import Simulator
import imp
import os
import os.path as op
import pdb
import brian2 as br2
import pytest

@pytest.fixture(scope='session')
def sim():
    data_path = '/home/jovyan/work/instrumentalVariable/test_data_sim/'
    param_module = 'params_test_fast_brian2.py'
    os.makedirs(data_path, exist_ok=True)
    jobname = param_module.replace('.py', '')
    currdir = op.dirname(op.abspath(__file__))
    f, p, d = imp.find_module(jobname, [currdir])
    p = imp.load_module(jobname, f, p, d).parameters

    sim = Simulator(
        p,
        data_path=data_path,
        jobname=jobname,
        verbose=True)
    
    sim.set_neurons()
    return sim

def test_create_connection_matrix(sim):
    from simulator_brian import create_connection_matrix as ccm    
    m = ccm(sim.p, 1234)
    sim.set_connections_from_matrix(m)
    # test for indegree
    indegree = np.count_nonzero(m/br2.nS, axis=1)
    assert np.all(indegree == sim.p['C_ex'] + sim.p['C_in'])

    # test that all interneuron weights are the same if not zero
    
    w_in = np.unique(m[:, sim.p['N_ex']:])
    assert len(w_in) <= 2
    assert np.sort(w_in)[-1] == sim.p['J_in']


def test_set_connections_from_matrix(sim):
    data_path = '/home/jovyan/work/instrumentalVariable/test_data_sim/'
    param_module = 'params_test_fast_brian2.py'
    os.makedirs(data_path, exist_ok=True)
    jobname = param_module.replace('.py', '')
    currdir = op.dirname(op.abspath(__file__))
    f, p, d = imp.find_module(jobname, [currdir])
    p = imp.load_module(jobname, f, p, d).parameters

    p['N_neurons'] = 6
    p['N_ex'] = 4
    p['N_in'] = 2
    
    sim = Simulator(
        p,
        data_path=data_path,
        jobname=jobname,
        verbose=True)
    sim.set_neurons()
    
    m = np.zeros((6,6))
    m[1,0] = 0.4
    m[5,4] = 0.1

    m = m * br2.nS

    sim.set_connections_from_matrix(m)
    syn_ex = sim.syn_ex
    syn_in = sim.syn_in

    assert np.array_equal(np.array(syn_ex.i), np.array([0]))
    assert np.array_equal(np.array(syn_ex.j), np.array([1]))
    assert np.array(syn_ex.w)- np.array([0.4e-9]) < 10e-10

    assert np.array_equal(np.array(syn_in.i), np.array([4-p['N_ex']]))
    assert np.array_equal(np.array(syn_in.j), np.array([5]))
    assert np.array(syn_in.w)- np.array([0.1e-9]) < 10e-10                                        


def test_set_network_event_monitor(sim):
    from simulator_brian import create_connection_matrix as ccm

    m = ccm(sim.p)
    sim.set_connections_from_matrix(m)
    sim.set_network_event_monitor()
    sim.set_spike_rec()

    sim.net.run(20*br2.ms)

    t_spks = sim.spk_mon.t
    t_evnts = sim.event_mon.t
    # select arbitrary unit
    bool_i = sim.event_mon.i==1
    assert np.array_equal(
        t_evnts[bool_i],
        np.unique(t_spks))

def test_generate_poisson_spike_trains_seed():
    # test whether global numpy seed affects realisation without seed
    # but not the one with seed
    from simulator_brian import generate_poisson_spike_trains as gpst
    n_neurons = 4
    n_inputs = 3
    rate = 0.5 * br2.Hz
    start = 0 * br2.second
    stop = 10 * br2.second
    dt = 1 * br2.second

    s = 1234  # seed
    idcs_s0, ts_s0 = gpst(n_neurons, n_inputs, rate, start, stop, dt, s)
    # realize without seed
    idcs_n0, ts_n0 = gpst(n_neurons, n_inputs, rate, start, stop, dt)    
    
    np.random.seed(123)
    
    idcs_s1, ts_s1 = gpst(n_neurons, n_inputs, rate, start, stop, dt, s)
    idcs_n1, ts_n1 = gpst(n_neurons, n_inputs, rate, start, stop, dt)    

    assert np.array_equal(idcs_s0, idcs_s1)
    assert np.array_equal(ts_s0, ts_s1)
    assert not np.array_equal(idcs_n0, idcs_n1)
    assert not np.array_equal(ts_n0, ts_n1)

def test_generate_poisson_spike_trains_units():
    from simulator_brian import generate_poisson_spike_trains as gpst
    n_neurons = 4
    n_inputs = 3
    rate = 500 * br2.Hz
    start = 0 * br2.ms
    stop = 10 * br2.ms
    dt = 1 * br2.ms

    # test several times to avoid that it happens that no spike is generated
    ts = []
    for i in range(10):
        seed = i
        idcs_i, ts_i = gpst(n_neurons, n_inputs, rate, start, stop, dt, seed)
        ts.append(ts_i)
    dims = []
    for ts_i in ts:
        if len(ts_i)>0:
            dims.append(ts_i.dimensions)
    for d in dims:
        assert d == br2.second.dimensions
    
    
