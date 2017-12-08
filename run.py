from simulator import Simulator
from params import parameters
import time

working_name = 'cond'

sim = Simulator(parameters, fname=working_name)
sim.set_kernel()
sim.set_neurons()
sim.set_connections()
sim.set_stimulation()
sim.set_spike_rec()
sim.set_state_rec()
sim.simulate_trials()
sim.save_data(overwrite=True)
sim.save_raster()
