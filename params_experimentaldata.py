import quantities as pq
import numpy as np
import pandas as pd

# parameters for loading data
# ------------------------------
link_db = 'https://buzsakilab.nyumc.org/datasets/McKenzieS'
data_dir = '/home/jovyan/work/instrumentalVariable/data/OptoData/'

# files to be downloaded
files_ext_by_shank = [
    '.clu.',
    '.res.']
            
files_ext_general = [
             '.xml',
             '.evt.ait',
             '.evt.aip'
#             '_analogin.xml',
#             '_analogin.nrs',
#             '_analogin.dat',
             ]

unit_spiketime = pq.s
n_shanks = 4
sampling_rate = 20000.

# parameter for analysis
# -----------------------

# autocorrelation
autocorr_limit = 0.008 * pq.s
autocorr_binsize =  0.0001 * pq.s

# generation of pseudo stimulations
# time after and before stimulations
t_dist_after = 100*pq.ms
t_dist_before = 600*pq.ms

# bins for time of first spike after stimulation
bins_first_spike = np.arange(0, 500, 1)

# minimal number of stimulations to considered for
# analysis of first spike after stimulation
min_n_stims = 10

# cch
ccg_time_limit = 0.0028*pq.s
ccg_binsize = 0.0004*pq.s
ccg_width = 0.001*pq.s/ccg_binsize
ltnc_syn = 0.001 * pq.s  # in s
ccg_hollow_fraction = 0.6
ccg_peak_wndw = [0.0008, 0.0028]
ccg_sig_level_causal = 0.01
ccg_sig_level_fast = 0.01

# cch in stim-pre-post example plots
ccg_binsize_stimprepost = 0.002*pq.s
ccg_limit_stimprepost = 0.008*pq.s

# iv
iv_window = 0.0075 * pq.s  # in s
iv_min_n_stim = 2500
iv_ltnc = 0.001*pq.s

stim_dat = np.array([
#    ['camkii4', '20160823', 1774, 1786],
    ['camkii4', '20160823', 2060, 2070],
    ['camkii4', '20160824', 2040, 2055],
    ['camkii4', '20160824', 2255, 2273],
    ['camkii4', '20160824', 2325, 2337],
#    ['camkii5', '20160930', 3400, 3406],
#    ['camkii5', '20160930', 1600, 1615]
])

iv_df_stim = pd.DataFrame(data=stim_dat,
                          columns=[
                              'animal', 'date',
                              'intensity_start',
                              'intensity_stop'])

# plotting settings
sdict = {
    # (11pt font = 360pt, 4.98) (10pt font = 345pt, 4.77)
    'figure.figsize': (4.98, 2),
    'figure.autolayout': False,
    'lines.linewidth': 2,
    'font.size': 11,
    'legend.frameon': False,
    'legend.fontsize': 11,
    'font.family': 'serif',
    'text.usetex': True
}

# detect fastresponding units
fastres_pthres = 10e-3
fastres_binwidth = 5 * pq.ms

# separation of stimulations
sep_bins = np.arange(0, 10000, 1)
sep_kernel_width = 10
sep_threshold = 10

# ccg between spikes and stim
stimccg_binsize = 3. * pq.ms
stimccg_limit = 45 * pq.ms
stimccg_pthres = 10e-3
stimccg_maxt = 3  # in ms

# exclude runs from blacklist
blk_blacklist = [
    {'animal': 'camkii4',
     'date': '20160816',
     'reason': 'intensity values missing'},
    {'animal': 'camkii4',
     'date': '20160817',
     'reason': 'intensity values missing'},
    {'animal': 'camkii4',
     'date': '20160817',
     'reason': 'intensity values missing'},
    {'animal': 'camkii4',
     'date': '20160819',
     'reason': 'intensity values missing'}]
            

# determine whether units are tagged
annot_delta_t_before = 30*pq.ms
annot_p_sign = 10e-10

# Bootstrapping
btstrp_n = 1000
btstrp_binsize = 100 * pq.s
