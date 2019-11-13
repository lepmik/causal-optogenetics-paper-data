import os
from tools import create_connection_matrix
import imp
import pickle as pkl
import pdb
import sys
import time

# Load parameters
data_path = '/home/jovyan/work/instrumentalVariable/data_branched_stim/'
param_module = 'params_brian2.py'
os.makedirs(data_path, exist_ok=True)
jobname = param_module.replace('.py', '')
currdir = '/home/jovyan/work/instrumentalVariable/causal-optogenetics-paper-data/'
f, p, d = imp.find_module(jobname, [currdir])
p = imp.load_module(jobname, f, p, d).parameters

# calculate some dependent variables
p['C_ex'] = int(p['eps'] * p['N_ex'])
p['C_in'] = int(p['eps'] * p['N_in'])
n = p['tau_syn_in'] * abs(p['E_L'] - p['E_in'])
d = p['J_ex'] * p['tau_syn_ex'] * abs(p['E_L'] - p['E_ex'])
p['J_in'] = p['g'] * d / n

# create network layout
m = create_connection_matrix(p, p['msd'])
# store it
np.savez(str(data_path)+'/'+'m.pkl', data=m)

for i in range(p['num_threads']):
    seed_i = i
    print('Start process with seed: ' + str(seed_i))

    # create_subfolder
    data_path_i = data_path + str(i) +'/'
    try:
     os.makedirs(os.path.join(data_path_i))
    except OSError:
       pass
    cmd = ("python simulation_with_branched_stimulation.py " +
        str(seed_i) + " " + str(data_path) +
           " " + param_module + " &")
    os.system(cmd)
    time.sleep(3)
