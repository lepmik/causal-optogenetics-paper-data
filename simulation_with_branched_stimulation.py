import os
import imp
import numpy as np
from brian2 import *
import sys

if len(sys.argv) not in [2, 3, 4]:
    raise IOError('Usage: "python script.py seed data_path param_module"')


seed_i = int(sys.argv[1])
data_path = sys.argv[2]
data_path_i = data_path + '/' + str(seed_i) + '/'
param_module = sys.argv[3]
jobname = param_module.replace('.py', '')
currdir = '/home/jovyan/work/instrumentalVariable/causal-optogenetics-brian2/'
f, p, d = imp.find_module(jobname, [currdir])
p = imp.load_module(jobname, f, p, d).parameters

logging.file_log = False

# set global seed
seed(seed_i)

# %%
'''
### Set neurons

'''

# %%
eqs = '''dV_m/dt = (g_L*(E_L-V_m)+Ie+Ii+I+Ix)/(C_m) : volt
         Ie = ge*(E_ex-V_m) : amp
         Ii = gi*(E_in-V_m) : amp
         dge/dt = -ge/(tau_syn_ex) : siemens
         dgi/dt = -gi/(tau_syn_in) : siemens
         Ix = sizes*0.5*(1+sin(2*pi*rates*t)) : amp
         rates : Hz
         sizes : amp
         I : amp'''
nodes = NeuronGroup(
    p['N_neurons'],
    model=eqs,
    threshold='V_m > V_th',
    reset='V_m = V_reset',
    refractory=p['t_ref'],
    namespace=p,
    method='euler',
)

nodes.sizes = p['s_sin']
nodes.rates= p['r_sin']

nodes_ex = nodes[:p['N_ex']]
nodes_in = nodes[p['N_ex']:]

nodes_ex_stim = nodes_ex[:p['stim_N_ex']]

# Load connections from matrix
m = np.load(str(data_path)+'m.npz')['m']*nS

# %%
syn_ex = Synapses(
    nodes_ex,
    nodes,
    model='w:siemens',
    on_pre='ge+=w',
    delay=p['syn_delay'])

syn_in = Synapses(
    nodes_in,
    nodes,
    model='w:siemens',
    on_pre='gi+=w',
    delay=p['syn_delay'])

N_ex = p['N_ex']
j, i = np.where(m)
j_ex, i_ex = j[i<N_ex], i[i<N_ex]
j_in, i_in = j[i>=N_ex], i[i>=N_ex]       

syn_ex.connect(i=i_ex, j=j_ex)
syn_ex.w = m[j_ex, i_ex]

syn_in.connect(i=i_in-N_ex, j=j_in)
syn_in.w = m[j_in, i_in]

# %%
'''
### Set background stimulation
'''

# %%
poissInp = PoissonInput(
    nodes, 'ge',
    N=p['N_p'],
    rate=p['rate_p'],
    weight=p['J_ex'])

# %%
'''
### Set spike monitors
'''

# %%
# 1 for times without stim
spk_mon1 = SpikeMonitor(nodes)

# 2 for stimulation periods
spk_mon2 = SpikeMonitor(nodes)
spk_mon2.active = False

# %%
'''
### Simulate
'''


# %%
# run init time without stimulation
run(p['init_simtime'])
sys.stdout.write('\r'+str(defaultclock.t/ms))
t2 = defaultclock.t/ms

# now with stimulation

# initialize random dealy time between branching points
t_dist_i = np.random.uniform(p['t_dist_min']/ms, p['t_dist_max']/ms)

stim_amp = p['stim_amp_ex'] * np.linspace(0., 1., p['stim_N_ex'])

cnt = 0

while defaultclock.t < p['runtime']:
    # get timepoint of branching, shift by the delay of 0.1 ms
    t2 = defaultclock.t/ms - 0.1

    # store spikes of baseline simulation every nth trial
    if cnt % p['n_save_spikes'] == 0:
        t1, i1 = np.array(spk_mon1.t/ms), np.array(spk_mon1.i).astype(int)
        data = {
            't': defaultclock.t/ms,
            'spk_ids': i1,
            'spk_ts': t1}
        np.savez(data_path_i + 'spks1.npz', data=data)    

    # store network state before stimulation
    store()
    # We'll need an explicit seed here, otherwise we'll continue with different
    # random numbers after the restore
    use_seed = randint(iinfo(np.int32).max)
    seed(use_seed)
    # change spike monitors
    spk_mon1.active = False
    spk_mon2.active = True
    run(p['t_pre_stim'])
    # stimulate
    nodes_ex_stim.I = stim_amp
    run(p['t_stim'])
    # turn stimuli off, but keep on simulation
    nodes_ex_stim.I = 0.*pA
    run(p['t_after_stim'])
    # store data of intermittent run
    spk_mon2_t = np.array(spk_mon2.t/ms)
    spk_mon2_i = np.array(spk_mon2.i).astype(int)
    data = {
        't': t2,
        'spk_ids': spk_mon2_i,
        'spk_ts': spk_mon2_t}
    np.savez(data_path_i + 'stimulation_data_{}.npz'.format(str(int(t2*10))), data=data)

    # restore previous network state and continue with simulation
    restore()
    seed(use_seed)
    spk_mon1.active = True
    spk_mon2.active = False
    cnt += 1
    with open(data_path_i + 'log', 'w') as f:
        f.write("t: " + str(int(defaultclock.t/ms))+ " cnt: " + str(cnt))
    # generate waiting time
    t_dist_i = np.random.uniform(p['t_dist_min']/ms, p['t_dist_max']/ms)
    t_dist_i = t_dist_i*ms
    sys.stdout.write('\r'+str(seed_i)+': '+str(defaultclock.t/ms))
    run(t_dist_i)

    t1, i1 = np.array(spk_mon1.t/ms), np.array(spk_mon1.i).astype(int)
    data = {
        't': defaultclock.t/ms,
        'spk_ids': i1,
        'spk_ts': t1}
    np.savez(data_path_i + 'spks1.npz', data=data)    



        
