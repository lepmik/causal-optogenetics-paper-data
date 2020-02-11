import numpy as np
import pandas as pd
from simulator_psc import Simulator
import sys
import imp
import os.path as op
import nest
from tqdm import tqdm
import warnings
import pathlib
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    prev_path = None
    if len(sys.argv) == 5:
        data_path, param_module, connfile, n_runs = sys.argv[1:]
    elif len(sys.argv) == 6:
        data_path, param_module, connfile, n_runs, prev_path = sys.argv[1:]
        prev_path = pathlib.Path(prev_path)
    else:
        raise IOError('Usage: "python simulator.py data_path parameters connfile [prev_path]')

    jobname = param_module.replace('.py', '')
    currdir = op.dirname(op.abspath(__file__))
    f, p, d = imp.find_module(jobname, [currdir])
    parameters = imp.load_module(jobname, f, p, d).parameters
    
    if prev_path is not None:
        stim_data = np.load(prev_path / 'stimulation_data_0.npz', allow_pickle=True)['data'][()]
        stim_amps = stim_data['stim_amps']
        n_start = int(prev_path.stem.split('_')[-1]) + 1
    else:
        stim_amps = None
        n_start = 0
    for n in tqdm(range(n_start, int(n_runs))):
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
            progress_bar=True,
            connfile=connfile,
            stim_amps=stim_amps,
        )
        if stim_amps is None:
            stim_amps = sim.stim_amps
