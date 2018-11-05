import os
import numpy as np
import neo
import scipy
import scipy.signal as scs
import pandas as pd
import quantities as pq
import pdb
import requests
import collections
import scipy as sc
from method import IV
from tqdm import tqdm
import copy
from skimage import measure

from tools_analysis import (cch_significance,
                            hollow_kernel,
                            correlogram)
from tools_analysis import poisson_continuity_correction as pcc


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def _download_file(url, fname):
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return True


def load_unitlabels(path_to_optolabels, data_dir):
    # Load matlab file that directs us to the right cells
    """
    "tags" is a struct describing which neurons went into our analysis. There
    are 5 fields

    1) tagged_post = the unit ID label for the neuron that was post synaptic to
    a connection that was significant during optogenetic stimulation of the
    presynaptic cell (N = 118 pairs)
    2) tagged_pre = the unit ID label for the neuron that was pre synaptic to
    an optogenetically confirmed connection, this unit should be
    optogenetically driven (N = 118)
    3) nottagged_pre = the unit ID label for the presynaptic neuron that did
    not show significant connecitivity during light stimulation
    4) nottagged_post = the unit ID label for the postsynaptic neuron that did
    not show significant connecitivity during light stimulation


    The labels are an Nx4 matrix.
    Column 1: session index into tags.fils
    Column 2: shank ID
    Column 3: unit ID
    Column 4: Cel ID (indexed 1-N, where N = number of cells recorded that day)

    fils is a path for another matlab file with more information, but you can
    use that to refer to the correct session in the database folder i sent you.

    --
    so for example, for the first opto-tagged cell pair,
    the presynaptic cell is
    >> tags.tagged_pre(1,:)
    16    2     4    16

     session 16, shank 2, unit 4 (on that shank), unit 16 (for that session)

    >> tags.fils{16}


    '/home/sam/Dropbox/buzsaki/data/SeqGen/camkii2_20160421_NWB.mat'


    so now, you can go to the directory for camkii2 on the data
    share I sent you

    https://buzsakilab.nyumc.org/datasets/McKenzieS/camkii2/20160421/

    and you can get the spike times for all units in shank 2 using these two
    text files. the suffix gives the shank number.

    20160421.res.2
    20160421.clu.2

    20160421.res.2 is in units of samples, so you need to divide by 20000 to
    get seconds.
    20160421.clu.2 gives the unit IDs. every row is a spike, except the first
    entry is the number of units. So, if you delete this you will get the same
    name of rows in the *.clu and the *.res files, which are matched tuples.
    Therefore you can search for just the samples in which unit 4 fires by
    doing something like this:

    >> ts = res(clu == 4);

    Have you worked with these before? I can give you the loading scripts if
    you need them.


    I also attach another struct, optostim. This gives the times of stimulation
    for every session.  this has two fields

    optostim(1).name = the session name
    optostim(1).stim = stimulation start and stop.  this is a 5x1 cell array.
    The Nth element is for stimulation [start stop] on shank N

    You can see this data in the analogin file.

    """

    optolabels = loadmat(data_dir + path_to_optolabels)
    tagged_pre = optolabels['tags']['tagged_pre']
    tagged_post = optolabels['tags']['tagged_post']
    nottagged_pre = optolabels['tags']['nottagged_pre']
    nottagged_post = optolabels['tags']['nottagged_post']

    units_db_tagged_pre = pd.DataFrame(tagged_pre,
                                       columns=['Sess_id',
                                                'shank_id',
                                                'unit_id',
                                                'cell_id'])
    units_db_tagged_pre['class'] = 'tagged_pre'
    units_db_tagged_pre['class_index'] = units_db_tagged_pre.index

    units_db_tagged_post = pd.DataFrame(tagged_post,
                                        columns=['Sess_id',
                                                 'shank_id',
                                                 'unit_id',
                                                 'cell_id'])
    units_db_tagged_post['class'] = 'tagged_post'
    units_db_tagged_post['class_index'] = units_db_tagged_post.index

    units_db_nottagged_pre = pd.DataFrame(nottagged_pre,
                                          columns=['Sess_id',
                                                   'shank_id',
                                                   'unit_id',
                                                   'cell_id'])
    units_db_nottagged_pre['class'] = 'nottagged_pre'
    units_db_nottagged_pre['class_index'] = units_db_nottagged_pre.index

    units_db_nottagged_post = pd.DataFrame(nottagged_post,
                                           columns=['Sess_id',
                                                    'shank_id',
                                                    'unit_id',
                                                    'cell_id'])
    units_db_nottagged_post['class'] = 'nottagged_post'
    units_db_nottagged_post['class_index'] = units_db_nottagged_post.index

    units_db = pd.DataFrame(columns=['Sess_id',
                                     'shank_id',
                                     'unit_id',
                                     'cell_id',
                                     'class',
                                     'class_index'])
    units_db = units_db.append([units_db_tagged_pre,
                                units_db_tagged_post,
                                units_db_nottagged_pre,
                                units_db_nottagged_post],
                               ignore_index=True)

    # get file name, date and animal.
    db_tmp = pd.DataFrame(index=range(len(units_db)),
                          columns=['animal', 'date', 'path', 'file'])
    # -1 for conversion from matlab indices to python
    relevant_files = optolabels['tags']['fils'][
        units_db['Sess_id'].values.astype(int)-1]
    for i, file_i in enumerate(relevant_files):
        animal_i = file_i[38:45]
        date_i = file_i[46:54]
        db_tmp['animal'][i] = animal_i
        db_tmp['date'][i] = date_i
        db_tmp['path'][i] = data_dir + animal_i + '/' + date_i + '/'
        db_tmp['file'][i] = file_i

    units_db = units_db.join(db_tmp)

    return units_db


def download_files_by_dict(file_dict,
                           data_dir,
                           n_shanks,
                           files_ext_general,
                           files_ext_by_shank,
                           link_db):
    for animal_i in file_dict.keys():
        for date_i in file_dict[animal_i]:
            data_dir_i = data_dir + animal_i + '/' + date_i + '/'
            if not os.path.exists(data_dir_i):
                os.makedirs(data_dir_i)

            for shank_i in range(1, n_shanks+1):
                for ext_i in files_ext_by_shank:
                    file_name_url = date_i + ext_i + str(shank_i)
                    if not os.path.exists(data_dir_i + file_name_url):
                        print('Downloading: ' + data_dir_i + file_name_url)
                        url = (link_db + '/' + animal_i + '/' + date_i +
                               '/' + file_name_url)
                        _download_file(url, data_dir_i + file_name_url)

            for ext_gen_i in files_ext_general:
                file_name_url = date_i + ext_gen_i
                if not os.path.exists(data_dir_i + file_name_url):
                    print('Downloading: ' + data_dir_i + file_name_url)
                    url = (link_db + '/' + animal_i + '/' + date_i +
                           '/' + file_name_url)
                    _download_file(url, data_dir_i + file_name_url)
    print('Finished downloading')
    return True


def create_neo_structure(file_dict,
                         data_dir,
                         n_shanks,
                         sampling_rate,
                         unit_spiketime):
    blks = []

    # get stimlation data
    for animal_i in file_dict.keys():
        for date_i in file_dict[animal_i]:
            path_i = data_dir + animal_i + '/' + date_i + '/'

            blk = neo.Block(animal=animal_i,
                            date=date_i,
                            path=path_i)
            seg = neo.Segment(name=date_i,
                              animal=animal_i)
            blk.segments.append(seg)
            chx_units = neo.ChannelIndex(index=0,
                                         name='units')
            blk.channel_indexes.append(chx_units)

            # add units by shanks
            for shank_i in range(1, n_shanks+1):
                clusters_f = open(path_i + date_i + '.clu.' + str(shank_i),
                                  'r')
                try:
                    clusters_i = np.array([
                        int(clu_id) for clu_id in clusters_f.readlines()])
                except:
                    pdb.set_trace()
                clusters_f.close()
                # first line contains number of clusters in file
                n_clusters_i = clusters_i[0]
                clusters_i = clusters_i[1:]
                # make sure that number of clusters match
                assert n_clusters_i == len(np.unique(clusters_i))

                times_f = open(path_i + date_i + '.res.' + str(shank_i), 'r')
                # get times of spikes
                times_i = np.array([
                            float(time_j) for time_j
                            in times_f.readlines()])
                times_f.close()

                # divide by sampling rate
                times_i /= sampling_rate

                # from documentation:
                # cluster 0 corresponds to mechanical noise (the wave shapes
                # do not look like neuron's spike). Cluster 1 corresponds to
                # small, unsortable spikes. These two clusters (0 and 1) should
                # not be used for analysis of neural data since they do not
                # correspond to successfully sorted spikes.
                # remove clusters == 0 and == 1
                pos_cluster_not_0_or_1 = np.where(clusters_i >= 2)[0]
                clusters_i = clusters_i[pos_cluster_not_0_or_1]
                times_i = times_i[pos_cluster_not_0_or_1]
                clusters_unique = np.unique(clusters_i)
                if times_i.size > 0:
                    max_time = np.max(times_i)
                    for clu_i in clusters_unique:
                        unit_i = neo.Unit(shank=shank_i,
                                          cluster=clu_i,
                                          session=date_i,
                                          name=(date_i + '/' +
                                                str(shank_i) + '/' +
                                                str(clu_i)),
                                          animal=animal_i,
                                          date=date_i,
                                          path=path_i,
                                          file=(date_i + '.clu.' +
                                                str(shank_i)))

                        spk_ts = times_i[clusters_i == clu_i]
                        train_i = neo.SpikeTrain(
                            times=spk_ts,
                            units=unit_spiketime,
                            t_start=0.*unit_spiketime,
                            t_stop=max_time*unit_spiketime)
                        train_i.unit = unit_i
                        # add unit and spiketrains only if spktrn
                        # is not empty
                        if len(train_i) > 0:
                            seg.spiketrains.append(train_i)
                            unit_i.spiketrains.append(train_i)
                        chx_units.units.append(unit_i)
            blks.append(blk)
    print('Neo structure created')
    return blks


def add_stimulation_data_to_blocks(blks):
    for blk in blks:
        seg = blk.segments[0]
        # extract stimulation times
        path_stimf = (blk.annotations['path'] +
                      blk.annotations['date'] + '.evt.ait')

        try:
            df = pd.read_csv(path_stimf,
                              header=None,
                              sep='\s|_', engine='python',
                              names=['t', 'type', 'switch', 'shank', 'extra'])
        except:
            pdb.set_trace()
        df = df.drop_duplicates()
        if len(df) > 0:
            # check if datatype matches
            assert np.issubdtype(df['t'].dtype, np.number)
            assert isinstance(df['type'][0], str)
            assert isinstance(df['switch'][0], str)
            assert np.issubdtype(df['shank'].dtype, np.number)

            df_on = df.loc[df['switch'] == 'on']
            df_off = df.loc[df['switch'] == 'off']
            df_extra = df.loc[df['switch'] == 'center']
            df_extra = df_extra.dropna()
            if len(df_extra) > 0:
                df_intens = df_extra.rename(columns={'extra': 'intensity',
                                                     'switch': 'stim_loc'})
            else:
                # extract stimulation intensities
                path_intensityf = (blk.annotations['path'] +
                                   blk.annotations['date'] + '.evt.aip')
                try:
                    df_intens = pd.read_csv(path_intensityf,
                                            header=None,
                                            sep='\s|_', engine='python',
                                            names=['t',
                                                   'type',
                                                   'stim_loc',
                                                   'shank',
                                                   'intensity'])
                except:
                    df_intens = None
                    pass
            # collect_data
            stim_start = []
            stim_dur = []
            stim_intensity = []
            stim_label = []
            stim_shank = []
            stim_loc = []
            for _, row_i in df_on.iterrows():
                stim_start_i = row_i['t'] * pq.ms
                # find stop
                a = df_off.loc[(df_off['t'] > row_i['t']) &
                               (df_off['type'] == row_i['type']) &
                               (df_off['shank'] == row_i['shank'])]
                stim_stop_i = a.iloc[0]['t'] * pq.ms
                stim_dur_i = stim_stop_i - stim_start_i

                # find intensity of stimulation in file
                intensity_i = np.nan
                stim_loc_i = np.nan
                if isinstance(df_intens, pd.DataFrame):
                    df_intens_sel = df_intens.loc[
                        (df_intens['t'] >= stim_start_i.magnitude) &
                        (df_intens['t'] <= stim_stop_i.magnitude) &
                        (df_intens['shank'] == row_i['shank']) &
                        (df_intens['type'] == row_i['type'])]

                    if len(df_intens_sel) > 1:
                        pdb.set_trace()
                        raise Exception
                    elif len(df_intens_sel) == 1:
                        intensity_i = df_intens_sel['intensity'].iloc[0]
                        stim_loc_i = df_intens_sel['stim_loc'].iloc[0]
                        assert isinstance(intensity_i, float)
                        assert isinstance(stim_loc_i, str)

                stim_start.append(stim_start_i)
                stim_dur.append(stim_dur_i)
                stim_intensity.append(intensity_i)
                stim_label.append(row_i['type'])
                stim_shank.append(row_i['shank'])
                stim_loc.append(stim_loc_i)

            stim_start = np.array(stim_start) * stim_start[0].units
            stim_dur = np.array(stim_dur) * stim_dur[0].units
            stim_intensity = np.array(stim_intensity)
            stim_label = np.array(stim_label)
            stim_shank = np.array(stim_shank)
            stim_loc = np.array(stim_loc)

            # make sure they are all the same length
            lengths = [len(stim_start), len(stim_dur),
                       len(stim_intensity), len(stim_label),
                       len(stim_shank), len(stim_loc)]
            assert all(x == lengths[0] for x in lengths)

            epc = neo.Epoch(
                times=stim_start,
                durations=stim_dur,
                shank=stim_shank,
                labels=stim_label,
                intensity=stim_intensity,
                stim_loc=stim_loc)
            seg.epochs.append(epc)
    print('Stimulation data added')
    return True


def select_times_from_epoch(epoch, conditions):
    '''
    Return a copy of the epoch, where events
    are selected based on certain conditions.

    Parameters
    ----------
    epoch : neo.Epoch
    conditions : dict, contains conditions to select epoch
                 form is {attribute: value}
                 if value is list with len=2, then it is considered a range.
                 left edge is included

    Returns
    -------
    epoch_out : neo.Epoch
    '''
    epoch = copy.deepcopy(epoch)
    ts = epoch.times
    bools = []  # holds conditions
    for key, val in conditions.items():
        pos_dot = key.find('.')
        if pos_dot >= 0:
            attr = key[:pos_dot]
            key_dict = key[pos_dot+1:]
            vals_epc = epoch.__dict__[attr][key_dict]
        else:
            vals_epc = epoch.__dict__[key]

        # assert that we have units on conditions
        # if we have them on attributes
        if isinstance(vals_epc, pq.Quantity) or isinstance(val, pq.Quantity):
            try:
                val = val.rescale(vals_epc.units)
            except:
                raise ValueError('Units of conditions must match')

        # check whether condition i has two values and describes a range
        if isinstance(val, collections.Iterable) and not isinstance(val, str):
            val = np.array(val)
            if len(val.shape) == 1:
                assert val.shape[0] == 2
                start = val[0]
                stop = val[1]
                bool_i = np.logical_and(vals_epc >= start,
                                        vals_epc < stop)
                bools.append(bool_i)

            elif len(val.shape) == 2:
                bools_sub = []
                assert val.shape[1] == 2
                for val_i in val:
                    start = val_i[0]
                    stop = val_i[1]
                    bool_i = np.logical_and(vals_epc >= start,
                                            vals_epc < stop)
                    bools_sub.append(bool_i)
                bools_sub = np.logical_or.reduce(bools_sub)
                bools.append(bools_sub)
        else:
            bools.append(vals_epc == val)
    bool_sel = np.logical_and.reduce(bools)
    ts_sel = ts[bool_sel]
    durations = epoch.durations[bool_sel]

    epoch_out = neo.Epoch(times=ts_sel,
                          labels=epoch.labels,
                          durations=durations,
                          name=epoch.name,
                          file_origin=epoch.file_origin,
                          description=epoch.description)
    epoch_out.annotations = epoch.annotations

    # modify other data of epoch as well
    lengths = [len(ts_sel)]
    for key, val in conditions.items():
        pos_dot = key.find('.')
        if pos_dot >= 0:
            attr = key[:pos_dot]
            key_dict = key[pos_dot+1:]
            vals_epc = epoch.__dict__[attr][key_dict]
            epoch_out.__dict__[attr][key_dict] = vals_epc[bool_sel]
            lengths.append(len(epoch_out.__dict__[attr][key_dict]))
        else:
            vals_epc = epoch.__dict__[key]
            epoch_out.__dict__[key] = vals_epc[bool_sel]
            lengths.append(len(epoch_out.__dict__[key]))

    assert all(x == lengths[0] for x in lengths)
    return epoch_out


def select_epoch_by_multiple_conditions(epoch,
                                        cond_list):
    """
    Merge individual calls of function select_times_from_epoch

    Params
    ------
    epoch : neo.Epoch
    cond_list : list of conditions, see select_times_from_epoch

    Returns
    -------
    epoch_out : neo.Epoch
    """
    epoch_list = []
    for cond in cond_list:
        epc_i = select_times_from_epoch(epoch, cond)
        epoch_list.append(epc_i)
    epoch_out = merge_epochs(epoch_list)
    return epoch_out


def merge_epochs(epoch_list):
    epc_out = epoch_list[0]
    for epc_i in epoch_list[1:]:
        epc_out = epc_out.merge(epc_i)
    sort_id = np.argsort(epc_out.times)
    epc_out = epc_out.duplicate_with_new_data(epc_out.times[sort_id])
    len_ts = len(epc_out.times)
    for key, val in epc_out.__dict__.items():
        if key == 'annotations':
            for key_annot, val_annot in epc_out.annotations.items():
                if (
                        isinstance(val_annot, collections.Iterable) and
                        len(val_annot) == len_ts):
                    epc_out.annotations[key_annot] = val_annot[sort_id]
        elif (
                isinstance(val, collections.Iterable) and
                key not in ['name',
                            'file_origin',
                            'segment',
                            '_dimensionality',
                            'annotations']):
            if len(val) == len_ts:
                epc_out.__dict__[key] = val[sort_id]
    return epc_out


def annotate_units_from_db(units_db, blks):
    '''
    Take dataframe from matlab structure and annotate
    respective units in neo blocks

    Params
    ------
    units_db : pandas.DataFrame
    blks : list of neo.Block

    '''

    for blk in blks:
        units = blk.channel_indexes[0].children

        for unit_i in units:

            unit_i_db = units_db.loc[
                (units_db['animal'] == blk.annotations['animal']) &
                (units_db['date'] == blk.annotations['date']) &
                (units_db['shank_id'] == unit_i.annotations['shank']) &
                (units_db['unit_id'] == unit_i.annotations['cluster'])]
            unit_i.annotations['tagged'] = np.any(
                unit_i_db['class'] == 'tagged_pre')


def multislice(x, starts, stops):
    '''
    Slice array x at multiple sites, defined by starts and stops

    Params
    ------
    x : np.array
    starts : np.array
    stops : np.array

    Returns
    -------
    x_sliced
    '''

    if (isinstance(x, pq.Quantity) or
            isinstance(starts, pq.Quantity) or
            isinstance(stops, pq.Quantity)):
        starts = starts.rescale(x.units)
        stops = stops.rescale(x.units)

    srt_starts = np.searchsorted(x, starts)
    srt_stops = np.searchsorted(x, stops)

    idcs = map(lambda v: np.r_[v[0]:v[1]], zip(srt_starts, srt_stops))
    idcs = np.concatenate(list(idcs))
    x_sliced = x[idcs]

    return x_sliced


def n_events_in_multislice(x, starts, stops):
    '''
    Return number of events in x in events defined by
    starts and stops

    Params
    ------
    x : np.array
    starts : np.array
    stops : np.array

    Returns
    -------
    nevents : np.array, number of events
    '''

    srt_starts = np.searchsorted(x, starts)
    srt_stops = np.searchsorted(x, stops)

    nevents = srt_stops - srt_starts

    return nevents


def determine_if_spktr_is_tagged(spktr,
                                 epoch,
                                 delta_t_before=2*pq.s,
                                 p_sign=10e-10,
                                 direction='exc'):
    '''
    Determine whether a spiketrain is tagged by optostimulation.

    We follow the approach of English et al:

    > To determine whether a unit was light modulated two criteria
    >  were required, 1) statistically significant increase in firing,
    >  and 2) an increase in firing rate > 50% of the spontaneous
    > rate. To test for significant rate changes, the number of spikes
    > emitted during each pulse was tabulated and compared to the number
    > emitted during the same interval within two seconds prior to light
    > delivery. These counts were tested using a Wilcoxon ranksum
    > non-parametric test of means and only units with highly significant
    >  (p < 10e-10) firing rate increases were retained.
    > These stringent methods were used to maximize decoupling of the
    > presynaptic neuron from the back- ground network activity.

    Optostimulation times are given in epoch.
    Result is stored in unit.annotations['tagged']

    Params
    ------
    spktr : neo.SpikeTrain
    epoch : neo.Epoch
    direction : str, 'exc' for excitatory stimulation
                     'inh' for inhibitory stimulation

    Returns
    -------
    tagged : bool
    '''
    # make sure all inputs have the same unit
    spk_ts = spktr.times
    epc_ts = epoch.times.rescale(spk_ts.units)
    if len(epc_ts) <= 2:
        raise Warning('Very low sample size')

    epc_durs = epoch.durations.rescale(spk_ts.units)
    delta_t_before = delta_t_before.rescale(spk_ts.units)
    # count spikes during epochs
    starts_stim = epc_ts
    stops_stim = epc_ts + epc_durs
    n_spks_stim = n_events_in_multislice(spk_ts,
                                         starts_stim,
                                         stops_stim)

    # count spikes before
    starts_bfr = starts_stim - delta_t_before
    stops_bfr = stops_stim - delta_t_before

    # make sure that negative shift, does not by itself
    # fall into a stimulated period
#    assert np.array_equal(src(starts_stim, starts_bfr),
#                          src(stops_stim, starts_bfr))
#    assert np.array_equal(src(starts_stim, stops_bfr),
#                          src(stops_stim, stops_bfr))

    n_spks_bfr = n_events_in_multislice(spk_ts,
                                        starts_bfr,
                                        stops_bfr)
    # condition 1)
    cond1 = False
    if np.mean(n_spks_bfr) != np.mean(n_spks_stim):
        if direction == 'exc':
            _, pval = sc.stats.mannwhitneyu(n_spks_bfr,
                                            n_spks_stim,
                                            alternative='less')
        elif direction == 'inh':
            _, pval = sc.stats.mannwhitneyu(n_spks_bfr,
                                            n_spks_stim,
                                            alternative='greater')
        if pval <= p_sign:
            cond1 = True

    # condition 2)
    cond2 = False
    if direction == 'exc' and np.mean(n_spks_stim) > 1.5*np.mean(n_spks_bfr):
        cond2 = True
    elif direction == 'inh' and np.mean(n_spks_stim) < 0.5*np.mean(n_spks_bfr):
        cond2 = True
    if cond1 and cond2:
        return True
    else:
        return False


def annotate_units_by_stim(blks,
                           delta_t_before=2*pq.s,
                           p_sign=10e-10,
                           direction='exciatory'):
    '''
    For each unit in session, check whether it is
    optogenetically stimualed

    Params
    ------
    blks : list of neo.Block
    direction : str, 'excitatory' or 'inhibitory'

    '''
    for blk in blks:
        seg = blk.segments[0]
        epc = seg.epochs[0]

        units = blk.channel_indexes[0].children

        for unit_i in units:
            spktr = unit_i.spiketrains[0]

            label_tagged = determine_if_spktr_is_tagged(
                spktr,
                epc,
                delta_t_before,
                p_sign)

            unit_i.annotations['tagged'] = label_tagged


def select_blocks_upon_stimtype(blks, stimtype=None, min_n=1,
                                min_intens=None, max_intens=None):
    blks_sel = []

    for blk in blks:
        seg = blk.segments[0]
        epc = seg.epochs[0]
        sel = [np.ones(len(epc.times), dtype=np.bool)]
        if isinstance(min_intens, (int, float)):
            try:
                sel.append(epc.annotations['intensity'] >= min_intens)
            except:
                pdb.set_trace()
        if isinstance(max_intens, (int, float)):
            sel.append(epc.annotations['intensity'] <= max_intens)
        if isinstance(stimtype, str):
            sel.append(epc.labels == stimtype)
        sel = np.logical_and.reduce(sel)
        n_sel = np.sum(sel)
        if n_sel >= min_n:
            blks_sel.append(blk)

    return blks_sel


def calculate_transmission_prob(blks,
                                ccg_time_limit,
                                ccg_binsize,
                                ccg_hollow_fraction,
                                ccg_width,
                                ccg_sig_level_causal,
                                ccg_sig_level_fast,
                                ccg_peak_wndw,
                                condition_annot_pre={'tagged': True},
                                condition_annot_post={'tagged': True},
                                **kwargs):
    """
    Implementation of English, Mckenzie approach to calculate
    spike transmission probability.

    Params
    ------
    require_presyn_tagged: bool, if True, check whether presynaptic is tagged,
                                 otherwise, skip calculation.
    """
    df_cch = pd.DataFrame(
        columns=['animal', 'date', 'shank_pre', 'cluster_pre',
                 'shank_post', 'cluster_post',
                 'pcausal', 'pfast', 'bins', 'cch',
                 'cch_s', 'transprob', 'bool_cnnctd'])
    for blk in blks:
        print(blk.annotations['animal'] + ': ' + blk.annotations['date'])
        units = blk.channel_indexes[0].children
        for unit_pre in units:
            cond_pre_lst = []
            for key in condition_annot_pre.keys():
                cond_pre_lst.append(unit_pre.annotations[key] ==
                                    condition_annot_pre[key])
            cond_pre = np.logical_and.reduce(cond_pre_lst)
            if (cond_pre):
                for unit_post in units:
                    spktr_pre = unit_pre.spiketrains[0]
                    spktr_post = unit_post.spiketrains[0]

                    cond_post_lst = []
                    for key in condition_annot_post.keys():
                        cond_post_lst.append(
                            unit_post.annotations[key] ==
                            condition_annot_post[key])
                    cond_post = np.logical_and.reduce(
                        cond_post_lst)

                    if (unit_pre != unit_post and
                            cond_post and
                            len(spktr_pre) > 0 and
                            len(spktr_post) > 0):
                        pcausal, pfast, bins, cch, cch_s = cch_significance(
                            spktr_pre,
                            spktr_post,
                            limit=ccg_time_limit,
                            binsize=ccg_binsize,
                            hollow_fraction=ccg_hollow_fraction,
                            width=ccg_width, **kwargs)
                        bool_cnnctd = np.any(np.logical_and.reduce(
                            (pcausal < ccg_sig_level_causal,
                             pfast < ccg_sig_level_fast,
                             bins >= ccg_peak_wndw[0],
                             bins <= ccg_peak_wndw[1])))
                        mask = ((bins >= ccg_peak_wndw[0]) &
                                (bins <= ccg_peak_wndw[1]))
                        trans_prob = np.sum(
                            cch[mask] - cch_s[mask]) / len(spktr_pre)

                        df_cch = df_cch.append(
                            {'animal': blk.annotations['animal'],
                             'date': blk.annotations['date'],
                             'shank_pre': unit_pre.annotations['shank'],
                             'cluster_pre': unit_pre.annotations['cluster'],
                             'shank_post': unit_post.annotations['shank'],
                             'cluster_post': unit_post.annotations['cluster'],
                             'pcausal': pcausal,
                             'pfast': pfast,
                             'bins': bins,
                             'cch': cch,
                             'cch_s': cch_s,
                             'bool_cnnctd': np.bool(bool_cnnctd),
                             'transprob': trans_prob},
                            ignore_index=True)
    return df_cch


def calculate_iv(blks,
                 df_sigstim,
                 iv_min_n_stim,
                 iv_window,
                 iv_ltnc,
                 condition_annot_pre={'tagged': True},
                 condition_annot_post={'tagged': False}):
    df_iv = pd.DataFrame(
        columns=['animal', 'date', 'shank_pre', 'cluster_pre',
                 'shank_post', 'cluster_post', 'ivwald',
                 'hitrate_stim',
                 'ys',
                 'Ysr'])

    for blk in blks:
        animal = blk.annotations['animal']
        date = blk.annotations['date']
        print(animal + ': ' + date)
        seg = blk.segments[0]
        epc = seg.epochs[0]
        units = blk.channel_indexes[0].children

        for unit_pre in units:
            # use unit_pre only if it matches criteria
            cond_pre_lst = []
            for key in condition_annot_pre.keys():
                cond_pre_lst.append(unit_pre.annotations[key] ==
                                    condition_annot_pre[key])
            cond_pre = np.logical_and.reduce(cond_pre_lst)
            cluster_pre = unit_pre.annotations['cluster']
            shank_pre = unit_pre.annotations['shank']

            if cond_pre:
                # select stims for respective unit
                df_sel = df_sigstim[
                    (df_sigstim['animal'] == animal) &
                    (df_sigstim['date'] == date) &
                    (df_sigstim['shank_unit'] == shank_pre) &
                    (df_sigstim['cluster'] == cluster_pre)]
                n_stims = df_sel['n_stims'].sum()
                if len(df_sel) > 0 and n_stims >= iv_min_n_stim:
                    cond_lst = []
                    for _, row in df_sel.iterrows():
                        cond_i = {'annotations.intensity': [
                            row['intens_start']-10e-3,
                            row['intens_stop']+10e-3],
                                  'annotations.shank':
                                  row['shank_stim']}
                        cond_lst.append(cond_i)
                    epc_sel = select_epoch_by_multiple_conditions(
                        epc,
                        cond_lst)
                    for unit_post in units:
                        # use unit_post only if it matches criteria
                        cond_post_lst = []
                        for key in condition_annot_post.keys():
                            cond_post_lst.append(
                                unit_post.annotations[key] ==
                                condition_annot_post[key])
                        cond_post = np.logical_and.reduce(
                            cond_post_lst)
                        if (unit_pre != unit_post and cond_post):
                            spktr_pre = unit_pre.spiketrains[0]
                            spktr_post = unit_post.spiketrains[0]
                            iv = IV(
                                spktr_pre.times.rescale(
                                    pq.s).magnitude,
                                spktr_post.times.rescale(
                                    pq.s).magnitude,
                                epc_sel.times.rescale(pq.s).magnitude,
                                iv_window.rescale(pq.s).magnitude,
                                iv_ltnc.rescale(pq.s).magnitude)
                            ys = iv.lams[iv.Stim, 1]
                            ysr = iv.lams[iv.StimRef, 1]
                            df_iv = df_iv.append(
                                {'animal': blk.annotations['animal'],
                                 'date': blk.annotations['date'],
                                 'shank_pre': unit_pre.annotations['shank'],
                                 'cluster_pre':
                                 unit_pre.annotations['cluster'],
                                 'shank_post': unit_post.annotations['shank'],
                                 'cluster_post':
                                 unit_post.annotations['cluster'],
                                 'ivwald': iv.wald,
                                 'hitrate_stim': iv.hit_rate,
                                 'ys': ys,
                                 'ysr': ysr},
                                ignore_index=True)
    return df_iv


def group_stimulations(blks,
                       bins,
                       kernel_width,
                       threshold):
    '''
    Go through stimulation intensities per session
    and group them by proximity.
    Make histogram of intensities, convolve them, detect
    overlapping regions
    '''

    for blk in blks:
        seg = blk.segments[0]
        epc = seg.epochs[0]
        shanks = np.unique(epc.annotations['shank'])
        stim_groups = np.zeros(len(epc.times))

        for shank_i in shanks:
            bool_shank = epc.annotations['shank'] == shank_i
            intens = epc.annotations[
                'intensity'][bool_shank]

            hist, bins = np.histogram(intens, bins)
            kernel = np.array([1]*kernel_width)
            kernel = kernel/np.sum(kernel)
            hist = np.convolve(hist, kernel, mode='same')
            labels = measure.label(hist > threshold)
            labels_unique = np.unique(labels)
            for label_i in labels_unique:
                if label_i >= 1:
                    bins_i = bins[:-1][labels == label_i]
                    intens_start = np.min(bins_i)
                    intens_stop = np.max(bins_i)
                    intens_bool = np.logical_and(
                        intens >= intens_start,
                        intens <= intens_stop)
                    idx_shank = np.where(bool_shank)[0]
                    idx_shankintens = idx_shank[intens_bool]
                    stim_groups[idx_shankintens] = label_i
        epc.annotations['stim_group'] = stim_groups
    return blks


def find_significant_stimulations(blks,
                                  stimccg_binsize,
                                  stimccg_limit,
                                  stimccg_pthres,
                                  condition_annot_unit={},
                                  **kwargs):
    df_cch = pd.DataFrame()

    for blk in blks:
        print(blk.annotations['animal'] + ' ' + blk.annotations['date'])

        seg = blk.segments[0]
        epc = seg.epochs[0]
        units = blk.channel_indexes[0].children
        stim_shanks = np.unique(epc.annotations['shank'])
        for shank in stim_shanks:
            shank_bool = epc.annotations['shank'] == shank
            stim_groups = np.unique(epc.annotations['stim_group'][shank_bool])
            stim_groups = stim_groups[stim_groups != 0]
            print('Shank: ' + str(shank))
            for i, group_i in enumerate(stim_groups):
                print('Group: ' + str(i) + ' of ' + str(len(stim_groups)))
                intens_bool = epc.annotations[
                    'stim_group'][shank_bool] == group_i
                intens_sel = epc.annotations[
                    'intensity'][shank_bool][intens_bool]
                intens_start = np.min(intens_sel) - 1
                intens_stop = np.max(intens_sel) + 1
                intens_mean = np.mean(intens_sel)
                epc_sel = select_times_from_epoch(
                    epc,
                    {'annotations.shank': shank,
                     'annotations.intensity': [intens_start,
                                               intens_stop]})
                for unit_i in tqdm(units):
                    cond_lst = []
                    for key in condition_annot_unit.keys():
                        cond_lst.append(unit_i.annotations[key] ==
                                        condition_annot_unit[key])
                    cond = np.logical_and.reduce(cond_lst)
                    if cond:
                        spktr = unit_i.spiketrains[0]
                        spktr_stim = neo.SpikeTrain(
                            times=epc_sel.times,
                            t_start=spktr.t_start,
                            t_stop=spktr.t_stop)
                        cch, bins = ccg(spktr_stim, spktr,
                                        binsize=stimccg_binsize,
                                        limit=stimccg_limit,
                                        **kwargs)
                        rate_baseline = np.mean(cch[bins < 0.])
                        pfast = np.zeros(cch.shape)
                        for m, val_m in enumerate(cch):
                            pfast[m] = pcc(np.array([val_m]), rate_baseline)
                        bool_sig = pfast < stimccg_pthres
                        first_bin_sig = np.nan
                        if np.sum(bool_sig) > 0:
                            bins_sig = bins[bool_sig]
                            bins_sig = bins_sig[bins_sig > 0]
                            if len(bins_sig) > 0:
                                first_bin_sig = np.min(bins_sig)
                        df_cch = df_cch.append({
                            'animal': blk.annotations['animal'],
                            'date': blk.annotations['date'],
                            'cluster': unit_i.annotations['cluster'],
                            'shank_unit': unit_i.annotations['shank'],
                            'cch': cch,
                            'bins': bins,
                            'pfast': pfast,
                            'intens_mean': intens_mean,
                            'intens_start': intens_start,
                            'intens_stop': intens_stop,
                            'shank_stim': shank,
                            'rate_baseline': rate_baseline,
                            'n_stims': len(epc_sel.times),
                            'first_bin_sig': first_bin_sig},
                                               ignore_index=True)
    return df_cch


def cch2transfprob(ccg, bins,
                   len_spktr_pre,
                   binsize,
                   hollow_fraction,
                   width,
                   peak_wndw,
                   kerntype='gaussian'):
    kernlen = len(ccg) - 1
    kernel = hollow_kernel(kernlen, width, hollow_fraction, kerntype)
    # padd edges
    len_padd = int(kernlen / 2.)
    ccg_padded = np.zeros(len(ccg) + 2 * len_padd)
    # "firstW/2 bins (excluding the very first bin) are duplicated,
    # reversed in time, and prepended to the ccg prior to convolving"
    ccg_padded[0:len_padd] = ccg[1:len_padd+1][::-1]
    ccg_padded[len_padd: - len_padd] = ccg
    # # "Likewise, the lastW/2 bins aresymmetrically appended to the ccg."
    ccg_padded[-len_padd:] = ccg[-len_padd-1:-1][::-1]
    # convolve ccg with kernel
    ccg_smoothed = scs.fftconvolve(ccg_padded, kernel, mode='valid')
    pfast = np.zeros(ccg.shape)
    assert len(ccg) == len(ccg_smoothed)
    for m, (val_m, rate_m) in enumerate(zip(ccg, ccg_smoothed)):
        pfast[m] = pcc(np.array([val_m]), rate_m)
    # pcausal describes the probability of obtaining a peak on one side
    # of the histogram, that is signficantly larger than the largest peak
    # in the anticausal direction. we leave the zero peak empty
    pcausal = np.zeros(ccg.shape)
    ccg_half_len = int(np.floor(len(ccg) / 2.))
    max_pre = np.max(ccg[:ccg_half_len])
    max_post = np.max(ccg[ccg_half_len:])
    for m, val_m in enumerate(ccg):
        if m < ccg_half_len:
            pcausal[m] = pcc(
                np.array([val_m]), max_post)
        if m > ccg_half_len:
            pcausal[m] = pcc(
                np.array([val_m]), max_pre)
    mask = ((bins >= peak_wndw[0]) &
            (bins <= peak_wndw[1]))
    trans_prob = np.sum(
        ccg[mask] - ccg_smoothed[mask]) / len_spktr_pre
    return trans_prob


def bootstrap_cch(
        blks, n,
        btstrp_binsize,
        ccg_binsize,
        ccg_hollow_fraction,
        ccg_peak_wndw,
        ccg_time_limit,
        ccg_width,
        conns_sel=None):

    df = pd.DataFrame(
        columns=['animal', 'date',
                 'shank_pre', 'cluster_pre',
                 'shank_post', 'cluster_post'])
    for conn in tqdm(conns_sel):
        (animal,
         date,
         shank_pre,
         cluster_pre,
         shank_post,
         cluster_post) = conn
        blk = [blk_i for blk_i in blks if
               blk_i.annotations['animal'] == animal and
               blk_i.annotations['date'] == date][0]
        seg = blk.segments[0]
        t_start = seg.t_start
        t_stop = seg.t_stop.rescale(t_start.units)
        btstrp_binsize = btstrp_binsize.rescale(t_start.units)
        bins = np.arange(t_start, t_stop, btstrp_binsize)
        bins = bins * t_start.units
        n_bins = len(bins) - 1
        # get units
        units = blk.channel_indexes[0].children
        unit_pre = [unit_i for unit_i in units if
                    unit_i.annotations['shank'] == shank_pre and
                    unit_i.annotations['cluster'] == cluster_pre][0]
        unit_post = [unit_i for unit_i in units if
                     unit_i.annotations['shank'] == shank_post and
                     unit_i.annotations['cluster'] == cluster_post][0]
        # get spiketrains
        spktr_post = unit_post.spiketrains[0]
        spktr_pre = unit_pre.spiketrains[0]
        len_spktr_pre = len(spktr_pre)

        lst_cch = []
        # cut spike train in small pieces and calculate
        # cch on each piece
        for strt, stp in zip(bins[:-1], bins[1:]):
            pre = spktr_pre.time_slice(strt-ccg_time_limit,
                                       stp+ccg_time_limit)
            post = spktr_post.time_slice(strt-ccg_time_limit,
                                         stp+ccg_time_limit)
            if len(pre.times) > 0 and len(post.times) > 0:
                cch_i, bins_i = correlogram(
                    pre, post,
                    binsize=ccg_binsize,
                    limit=ccg_time_limit)
            else:
                bins_i = np.arange(
                    -ccg_time_limit,
                    ccg_time_limit + ccg_binsize,
                    ccg_binsize)[1:]
                cch_i = np.zeros(len(bins_i))
            lst_cch.append(cch_i)
        # pick without replacement cch pieces,
        # add them together and calculate transmission
        # prob
        for i in range(n):
            idcs_i = np.random.choice(
                np.arange(n_bins), size=n_bins,
                replace=True)
            lst_i = [lst_cch[i] for i in idcs_i]
            cch_sum = np.sum(np.vstack(lst_i), axis=0)
            p = cch2transfprob(
                cch_sum, bins_i,
                len_spktr_pre=len_spktr_pre,
                binsize=ccg_binsize,
                hollow_fraction=ccg_hollow_fraction,
                width=ccg_width,
                peak_wndw=ccg_peak_wndw,
                kerntype='gaussian')
            df = df.append({
                'animal': animal,
                'date': date,
                'cluster_pre': cluster_pre,
                'shank_pre': shank_pre,
                'cluster_post': cluster_post,
                'shank_post': shank_post,
                'trans_prob': p},
                      ignore_index=True)
    return df


def bootstrap_iv(
        blks,
        n,
        df_sigstim,
        iv_min_n_stim,
        iv_window,
        iv_ltnc,
        conns_sel=None):

    df = pd.DataFrame(
        columns=['animal', 'date',
                 'shank_pre', 'cluster_pre',
                 'shank_post', 'cluster_post'])
    for conn in tqdm(conns_sel):
        (animal,
         date,
         shank_pre,
         cluster_pre,
         shank_post,
         cluster_post) = conn
        blk = [blk_i for blk_i in blks if
               blk_i.annotations['animal'] == animal and
               blk_i.annotations['date'] == date][0]
        seg = blk.segments[0]
        epc = seg.epochs[0]

        # get units
        units = blk.channel_indexes[0].children
        unit_pre = [unit_i for unit_i in units if
                    unit_i.annotations['shank'] == shank_pre and
                    unit_i.annotations['cluster'] == cluster_pre][0]
        unit_post = [unit_i for unit_i in units if
                     unit_i.annotations['shank'] == shank_post and
                     unit_i.annotations['cluster'] == cluster_post][0]
        # get spiketrains
        spktr_post = unit_post.spiketrains[0]
        spktr_pre = unit_pre.spiketrains[0]

        # select stimulations
        df_sel = df_sigstim[
            (df_sigstim['animal'] == animal) &
            (df_sigstim['date'] == date) &
            (df_sigstim['shank_unit'] == shank_pre) &
            (df_sigstim['cluster'] == cluster_pre)]
        cond_lst = []
        for _, row in df_sel.iterrows():
            cond_i = {'annotations.intensity': [
                row['intens_start']-10e-3,
                row['intens_stop']+10e-3],
                      'annotations.shank':
                      row['shank_stim']}
            cond_lst.append(cond_i)
        epc_sel = select_epoch_by_multiple_conditions(
            epc,
            cond_lst)

        iv = IV(
            spktr_pre.times.rescale(pq.s).magnitude,
            spktr_post.times.rescale(pq.s).magnitude,
            epc_sel.times.rescale(pq.s).magnitude,
            iv_window.rescale(pq.s).magnitude,
            iv_ltnc.rescale(pq.s).magnitude)
        nstim = len(iv.lams)
        for i in range(n):
            idcs_i = np.random.choice(
                np.arange(nstim), size=nstim,
                replace=True)
            iv_lams_i = iv.lams[idcs_i, :]
            iv_stim_i = iv.Stim[idcs_i]
            iv_stimRef_i = iv.StimRef[idcs_i]
            ys_i = iv_lams_i[iv_stim_i, 1]
            ysr_i = iv_lams_i[iv_stimRef_i, 1]
            wald = ys_i.mean() - ysr_i.mean()

            df = df.append({
                'animal': animal,
                'date': date,
                'cluster_pre': cluster_pre,
                'shank_pre': shank_pre,
                'cluster_post': cluster_post,
                'shank_post': shank_post,
                'ivwald': wald},
                      ignore_index=True)
    return df


def ccg(t1, t2=None, binsize=1*pq.ms,
        limit=100*pq.ms, auto=False,
        **kwargs):
    '''
    Wrapper for cross_correlation_histogram from elephant.

    Note
    ----
    returns bins at right edges
    '''
    import elephant.spike_train_correlation as corr
    import elephant.conversion as conv
    if auto: t2 = t1
    cch, bin_ids = corr.cross_correlation_histogram(
        conv.BinnedSpikeTrain(t1, binsize),
        conv.BinnedSpikeTrain(t2, binsize),
        window=[-limit, limit],
        **kwargs)
    bins = cch.times
    cch = cch.magnitude.flatten()
    if auto:
        # Compensate for the peak at time zero that results in autocorrelations
        # by subtracting the total number of spikes from that bin. Note
        # possible numerical issue here because 0.0 may fall at a bin edge.
        c_temp, bins_temp = np.histogram([0.], bins=bins)
        bin_containing_zero = np.nonzero(c_temp)[0][0]
        cch[bin_containing_zero] = 0
    return cch, bins
