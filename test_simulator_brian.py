import numpy as np
from simulator_brian import simulator as sim

def test_record_state_variables():
    data_path = '/home/jovyan/work/instrumentalVariable/test_data_sim/'
    param_module = 'params_test_fast_brian2.py'
    os.makedirs(data_path, exist_ok=True)
    jobname = param_module.replace('.py', '')
    currdir = op.dirname(op.abspath(__file__))
    f, p, d = imp.find_module(jobname, [currdir])
    parameters = imp.load_module(jobname, f, p, d).parameters

    parameters['N_neurons'] = 400
    parameters['N_ex'] = 320
    parameters['N_in'] = 80
    
    sim = sim(parameters, data_path=data_path, jobname=jobname, verbose=True)
    sim.set_neurons()
    sim.set_background()
    sim.set_connections()
    sim.set_spike_rec()
    #
    sim.set_state_rec()
