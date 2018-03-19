import sys
from simulator import Simulator
import imp
import os.path as op

if len(sys.argv) != 2:
    raise IOError('Usage: "python run.py parameters"')

param_module = sys.argv[1]
jobname = param_module.replace('.py', '')
currdir = op.dirname(op.abspath(__file__))
f, p, d = imp.find_module(jobname, [currdir])
parameters = imp.load_module(jobname, f, p, d).parameters

sim = Simulator(parameters, mpi=False, data_path='results',
                fname=jobname)
sim.simulate(save=True, progress_bar=True)
