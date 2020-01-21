import numpy as np
import pandas as pd
from simulator_psc import Simulator
import sys
import imp
import os.path as op
import nest
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    if len(sys.argv) == 5:
        data_path, param_module, connfile, n_runs = sys.argv[1:]
    else:
        raise IOError('Usage: "python simulator.py data_path parameters [connfile]')

    jobname = param_module.replace('.py', '')
    currdir = op.dirname(op.abspath(__file__))
    f, p, d = imp.find_module(jobname, [currdir])
    parameters = imp.load_module(jobname, f, p, d).parameters

    stim_amps = None
    for n in tqdm(range(int(n_runs))):
        seed = n + 1000
        sim = Simulator(
            parameters,
            data_path=data_path + '_' + str(n),
            jobname=jobname,
            verbose=False,
            save_to_file=True,
            msd=seed
        )
        sim.simulate(
            state=False,
            progress_bar=True,
            connfile=connfile,
            stim_amps=stim_amps,
            branch_out=True
        )
        if stim_amps is None:
            stim_amps = sim.stim_amps
