import sys
import os
import numpy as np
import neo
import scipy
import pandas as pd
import quantities as pq
import pdb
import requests
import collections
import scipy as sc
from method import IV
from matplotlib import pyplot as plt
from tools import despine

import statsmodels.api as sm
import copy
from skimage import measure
sys.path.append('../exana/'),
from exana.statistics.tools import ccg_significance, ccg
from exana.statistics.tools import poisson_continuity_correction as pcc


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


def download_files(units_db, link_db,
                   files_ext_general,
                   files_ext_by_shank):
    
    for i, row in units_db.iterrows():
        dir_i = row['animal']
        date_i = row['date']
        shank_i = row['shank_id']
        data_dir_i = row['path']

        if not os.path.exists(data_dir_i):
            os.makedirs(data_dir_i)

        for ext_i in files_ext_by_shank:
            file_name_url = date_i + ext_i + str(shank_i)
            if not os.path.exists(data_dir_i + file_name_url):
                print('Downloading: ' + data_dir_i + file_name_url)
                url = (link_db + '/' + dir_i + '/' + date_i +
                       '/' + file_name_url)
                _download_file(url, data_dir_i + file_name_url)

        for ext_gen_i in files_ext_general:
            file_name_url = date_i + ext_gen_i
            if not os.path.exists(data_dir_i + file_name_url):
                print('Downloading: ' + data_dir_i + file_name_url)
                url = (link_db + '/' + dir_i + '/' + date_i +
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


def get_first_spikes(spktr,
                     epc,
                     unit=pq.s):
    '''
    Get time difference between events in epoch to
    the next spike in spiketrain
    
    Parameters
    ----------
    spktr : neo.Spiketrain
    epc : neo.Epoch with events
    unit : quantities.Quantity (optional),
           allows to specify output unit

    Returns
    -------
    t_delta : np.array
    '''

    epc_ts = epc.times.rescale(unit)
    spk_ts = spktr.times.rescale(unit)

    sorted = np.searchsorted(spk_ts, epc_ts)
    # remove indices that exceed spk_ts
    max_valid = len(spk_ts)-1
    valid_bool = sorted <= max_valid
    sorted = sorted[valid_bool]
    epc_ts = epc_ts[valid_bool]
    t_delta = spk_ts[sorted] - epc_ts

    return t_delta


def get_pseudo_epoch(epc,
                     t_dist_after,
                     t_dist_before,
                     type='one2one',  # 'full'
                     dur='same',
                     ):
    '''
    To interpret the effect of the stimulation,
    we need a baseline.
    For its creation we randomly assign times between
    the stimulation with a certain margin after the previous
    and before the next stimulation.

    Parameters
    ----------
    epc : neo.Epoch, Epoch containing stimulation events
    t_dist_after : margin after previous stimulation where
                   no pseudo event is allowed
    t_dist_before : margin before next stimulation where no
                    pseudo event is allowed
    type : 'one2one', return for each stimulation only a single pseudo
                      stimulation
           'full', return as many pseudo stimulations as possible that
                   fit into the valid window
    dur : 'same', pseudo stimulations have same duration as previous
                  stimulation
          np.float, define length of stimulation manually

    Returns
    -------
    epc_pseudo : neo.Epoch, containing pseudo events
    '''

    assert t_dist_after.units == epc.units
    assert t_dist_before.units == epc.units

    # get real stimulation times
    stim_start = epc.times
    stim_stop = stim_start+epc.durations
    durs = epc.durations
    units = epc.times.units

    # create ranges for pseudo times
    pseudo_start = stim_stop[:-1] + t_dist_after
    pseudo_stop = stim_start[1:] - t_dist_before

    # exclude ranges which are to short
    bool_valid = np.greater(pseudo_stop, pseudo_start)
    pseudo_start = pseudo_start[bool_valid]
    pseudo_stop = pseudo_stop[bool_valid]

    if type == 'one2one':
        ts_pseudo = np.random.uniform(pseudo_start, pseudo_stop)*units
        if dur == 'same':
            durs_pseudo = durs[1:]
        elif isinstance(dur, (int, float)):
            durs_pseudo = np.repeat(dur, len(pseudo_start))

    elif type == 'full':
        ar = [pseudo_start[:, np.newaxis],
              pseudo_stop[:, np.newaxis]]
        if dur == 'same':
            ar.append(durs)
        elif isinstance(dur, (int, float, pq.Quantity)):
            ar.append(np.repeat(dur, len(pseudo_start))[:, np.newaxis])
        else:
            raise ValueError('dur ' + dur + ' not recognized')
        ar = np.hstack(ar)
        ts_pseudo = []
        durs_pseudo = []
        for start_i, stop_i, dur_i in ar:
            ts_i = np.arange(start_i, stop_i, dur_i)
            ts_pseudo.append(ts_i)
            durs_pseudo.append(np.repeat(dur_i, len(ts_i)))
        
        ts_pseudo = np.concatenate(ts_pseudo) * units
        durs_pseudo = np.concatenate(durs_pseudo) * units
    else:
        raise ValueError('Type ' + type + ' not recognized')
    
    epc_pseudo = neo.Epoch(ts_pseudo, durations=durs_pseudo)

    return epc_pseudo


def detect_fast_responding_units(blks, binwidth,
                                 pthresh,
                                 event_sel_dict,
                                 t_dist_before,
                                 t_dist_after,
                                 same_shank=True
):
    '''
    Bin spiketrains of units at stimulation onset
    and during times between and detect whether activity during stimulation
    is significantly larger compared to non-stimulation.
    Stimulations to count responses can be selected, while the reference
    avoids all stimulations
    Results will be noted in unit.annotations['fastresponding']

    Parameters
    ----------
    blks : list of neo.Blocks
    binwidth : pq.Quantity, for binning spiketrain
    pthresh : np.float, Upper limit of test to say whether unit is modulated
    event_sel_dict : dict, conditions to select stimulation events
    t_dist_before, t_dist_after : pq.Quantity, extra distance to stimulation
                                               events when calculating baseline
                                               activity
    same_shank : bool, because of difficulty in parsing it within
                       event_sel_dict we decided here whether we should
                       only look at stimulations at same shank as respective
                       unit
                       
    
    
    Returns
    -------
    out : pd.DataFrame, dataframe summarizing results for each unit
    '''
    df = pd.DataFrame(columns=['animal', 'date', 'cluster', 'shank',
                               'pval', 'fastresponding',
                               'p0', 'n0', 'p1', 'n1'])
    
    src = np.searchsorted
    for blk in blks:
        print(blk.annotations['animal'] + ' - ' + blk.annotations['date'])
        seg = blk.segments[0]
        epc = seg.epochs[0]
        units = blk.channel_indexes[0].children
        shanks = np.unique([unit.annotations['shank'] for unit in units])
        for shank_i in shanks:
            if same_shank:
                event_sel_dict['annotations.shank'] = shank_i
            epc_sel = select_times_from_epoch(
                epc,
                event_sel_dict)

            stimts = epc_sel.times.rescale(pq.ms).magnitude

            units_shank = [unit for unit in units if
                           unit.annotations['shank'] == shank_i]
            epc_pseudo = get_pseudo_epoch(
                epc,
                t_dist_before,
                t_dist_after,
                'full',
                binwidth)
            refts = epc_pseudo.times.rescale(pq.ms).magnitude
            
            for unit in units_shank:
                spktr = unit.spiketrains[0]
                spkts = spktr.times.rescale(pq.ms).magnitude
                wnsz = binwidth.rescale(pq.ms).magnitude
                a = (src(spkts, stimts, 'left') <
                     src(spkts, stimts + wnsz, 'right'))
                b = (src(spkts, refts, 'left') <
                     src(spkts, refts + wnsz, 'right'))
                p0 = np.count_nonzero(a)
                n0 = len(a)
                p1 = np.count_nonzero(b)
                n1 = len(b)
                _, pval = sm.stats.proportions_ztest([p0, p1], [n0, n1],
                                                     alternative='larger')
                fastresponding = False
                if pval < pthresh:
                    fastresponding = True
                unit.annotations['fastresponding'] = fastresponding

                df = df.append({'animal': blk.annotations['animal'],
                                'date': blk.annotations['date'],
                                'cluster': unit.annotations['cluster'],
                                'shank': unit.annotations['shank'],
                                'pval': pval,
                                'fastresponding': fastresponding,
                                'p0': p0, 'n0': n0,
                                'p1': p1, 'n1': n1},
                               ignore_index=True)
    return df


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

        # check whether condition i has to values and describes a range
        if isinstance(val, collections.Iterable) and not isinstance(val, str):
            assert len(val) == 2
            
            start = val[0]
            stop = val[1]
#            if np.any(np.isnan(vals_epc)):
#                pdb.set_trace()
            bool_i = np.logical_and(vals_epc >= start,
                                    vals_epc < stop)
            bools.append(bool_i)

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
                          description=epoch.description,
                          annotations=epoch.annotations)

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
                                 p_sign=10e-10):
    
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
    n_spks_bfr = n_events_in_multislice(spk_ts,
                                        starts_bfr,
                                        stops_bfr)
    # condition 1)
    cond1 = False
    if np.mean(n_spks_bfr) != np.mean(n_spks_stim):
        _, pval = sc.stats.mannwhitneyu(n_spks_bfr,
                                        n_spks_stim,
                                        alternative='less')
        if pval <= p_sign:
            cond1 = True
    
    # condition 2)
    cond2 = False
    if np.mean(n_spks_stim) > 1.5*np.mean(n_spks_bfr):
        cond2 = True
    if cond1 and cond2:
        return True
    else:
        return False


def annotate_units_by_stim(blks,
                           delta_t_before=2*pq.s,
                           p_sign=10e-10):
    '''
    For each unit in session, check whether it is
    optogenetically stimualed

    Params
    ------
    blks : list of neo.Block

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


def plot_ccgs_between_stim_pre_and_post(spktr_stim,
                                        spktr_pre, spktr_post,
                                        binsize_ccg, limit_ccg,
                                        latency, winsize,
                                        plot_density=False,
                                        ylim=None):
    iv = IV(spktr_pre.times.rescale(pq.s).magnitude,
            spktr_post.times.rescale(pq.s).magnitude,
            spktr_stim.times.rescale(pq.s).magnitude,
            winsize.rescale(pq.s).magnitude,
            latency.rescale(pq.s).magnitude)
    pre1 = iv.Stim
    pre0 = iv.StimRef
    # ccg between stim and spktr_pre
    ccg_stimpre, bins = ccg(spktr_stim, spktr_pre, binsize_ccg, limit_ccg)
    # ccg between stim and spktr_pre
    ccg_stimpost_pre1, bins = ccg(spktr_stim[pre1], spktr_post,
                                  binsize_ccg, limit_ccg)
    ccg_stimpost_pre0, bins = ccg(spktr_stim[pre0], spktr_post,
                                  binsize_ccg, limit_ccg)
    if plot_density:
        ccg_stimpre = ccg_stimpre/np.sum(ccg_stimpre)
        ccg_stimpost_pre1 = ccg_stimpost_pre1 / np.sum(ccg_stimpost_pre1)
        ccg_stimpost_pre0 = ccg_stimpost_pre0 / np.sum(ccg_stimpost_pre0)
    
    cntr = int(len(bins)/2)
    fig, ax = plt.subplots(3, 1)
    ax[0].bar(bins[cntr:], ccg_stimpre[cntr:],
              width=binsize_ccg.rescale(bins.units))
    ax[1].bar(bins[cntr:], ccg_stimpost_pre1[cntr:],
              width=binsize_ccg.rescale(bins.units))
    ax[2].bar(bins[cntr:], ccg_stimpost_pre0[cntr:],
              width=binsize_ccg.rescale(bins.units))
    if ylim:
        ax[0].set_ylim(ylim)
        ax[1].set_ylim(ylim)
        ax[2].set_ylim(ylim)
        

def regplot(x, y, data, model, ci=95.,
            scatter_color='b', model_color='k', ax=None,
            scatter_kws={}, regplot_kws={}, cmap=None,
            xlabel=True, ylabel=True, colorbar=True,
            groupby=False,
            **kwargs):
    
    from seaborn import utils
    from seaborn import algorithms as algo
    if ax is None:
        fig, ax = plt.subplots()
    _x = data[x]
    _y = data[y]
    grid = np.linspace(_x.min(), _x.max(), 100)
    
    X = np.c_[np.ones(len(_x)), _x]
    G = np.c_[np.ones(len(grid)), grid]
    
    results = model(_y, X, **kwargs).fit()
    
    def reg_func(xx, yy):
        yhat = model(yy, xx, **kwargs).fit().predict(G)
        return yhat
    yhat = results.predict(G)
    yhat_boots = algo.bootstrap(X, _y, func=reg_func,
                                n_boot=1000, units=None)
    err_bands = utils.ci(yhat_boots, ci, axis=0)
    ax.plot(grid, yhat, color=model_color, **regplot_kws)
    ax.scatter(_x, _y,
               **scatter_kws)
    ax.fill_between(grid, *err_bands, facecolor=model_color, alpha=.15)
    h = plt.Line2D([], [], label='$R^2 = {:.3f}$'.format(results.rsquared),
                   ls='-', color='k')
    plt.legend(handles=[h])
    
    if xlabel:
        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
    if ylabel:
        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(y)
    despine()
    return results


def calculate_transmission_prob(blks,
                                ccg_time_limit,
                                ccg_binsize,
                                ccg_hollow_fraction,
                                ccg_width,
                                ccg_sig_level_causal,
                                ccg_sig_level_fast,
                                ccg_peak_wndw,
                                condition_annot_pre={'tagged': True},
                                condition_annot_post={'tagged': True}):
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
                        pcausal, pfast, bins, cch, cch_s = ccg_significance(
                            spktr_pre,
                            spktr_post,
                            limit=ccg_time_limit,
                            binsize=ccg_binsize,
                            hollow_fraction=ccg_hollow_fraction,
                            width=ccg_width)
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
                 iv_min_n_stim,
                 iv_window,
                 iv_ltnc,
                 condition_annot_pre={'tagged': True},
                 condition_annot_post={'tagged': False}):
    df_iv = pd.DataFrame(
        columns=['animal', 'date', 'shank_pre', 'cluster_pre',
                 'shank_post', 'cluster_post', 'ivwald',
                 'hitrate_stim'])
    for blk in blks:
        animal = blk.annotations['animal']
        date = blk.annotations['date']
        print(animal + ': ' + date)
        seg = blk.segments[0]
        epc = seg.epochs[0]
        units = blk.channel_indexes[0].children
        stims_shank = np.unique(epc.annotations['shank'])

        for stim_shank_i in stims_shank:
            # iterate through all stimulation intensities

            epc_sel = select_times_from_epoch(
                epc,
                {'annotations.shank': stim_shank_i,
                 'labels': 'pulse'})

            for unit_pre in units:
                # use unit_pre only if it matches criteria
                cond_pre_lst = []
                for key in condition_annot_pre.keys():
                    cond_pre_lst.append(unit_pre.annotations[key] ==
                                        condition_annot_pre[key])
                cond_pre = np.logical_and.reduce(cond_pre_lst)
                shank_pre = unit_pre.annotations['shank']
                if (cond_pre and
                    len(epc_sel.times) > iv_min_n_stim and
                        stim_shank_i == shank_pre):
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
                                spktr_pre.times.rescale(pq.s).magnitude,
                                spktr_post.times.rescale(pq.s).magnitude,
                                epc_sel.times.rescale(pq.s).magnitude,
                                iv_window.rescale(pq.s).magnitude,
                                iv_ltnc.rescale(pq.s).magnitude)
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
                                 'hitrate_stim': iv.hit_rate},
                                ignore_index=True)

    return df_iv


def calculate_iv_intensity(blks,
                           iv_df_stim,
                           iv_min_n_stim,
                           iv_window,
                           iv_ltnc,
                           condition_annot_pre={'tagged': True},
                           condition_annot_post={'tagged': False}):
    df_iv = pd.DataFrame(
        columns=['animal', 'date', 'shank_pre', 'cluster_pre',
                 'shank_post', 'cluster_post', 'ivwald',
                 'hitrate_stim',
                 'stimintens_start',
                 'stimintens_stop',
                 'ys',
                 'Ysr'])
    
    for blk in blks:
        animal = blk.annotations['animal']
        date = blk.annotations['date']
        print(animal + ': ' + date)

        # select right stimulation
        df_stim_sel = iv_df_stim.loc[
            (iv_df_stim['animal'] == animal) &
            (iv_df_stim['date'] == date)]
        if len(df_stim_sel) > 0:
            seg = blk.segments[0]
            epc = seg.epochs[0]
            units = blk.channel_indexes[0].children
            stims_shank = np.unique(epc.annotations['shank'])

            for stim_shank_i in stims_shank:
                for _, stim_row in df_stim_sel.iterrows():
                    # iterate through all stimulation intensities
                    iv_stimintens_sel = [np.float(stim_row['intensity_start']),
                                         np.float(stim_row['intensity_stop'])]
                    epc_sel = select_times_from_epoch(
                        epc,
                        {'annotations.shank': stim_shank_i,
                         'labels': 'pulse',
                         'annotations.intensity': iv_stimintens_sel})

                    for unit_pre in units:
                        # use unit_pre only if it matches criteria
                        cond_pre_lst = []
                        for key in condition_annot_pre.keys():
                            cond_pre_lst.append(unit_pre.annotations[key] ==
                                                condition_annot_pre[key])
                        cond_pre = np.logical_and.reduce(cond_pre_lst)
                        shank_pre = unit_pre.annotations['shank']
                        if (cond_pre and
                            len(epc_sel.times) > iv_min_n_stim and
                                stim_shank_i == shank_pre):
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
                                         'stimintens_start': iv_stimintens_sel[0],
                                         'stimintens_stop': iv_stimintens_sel[1],
                                         'ys': ys,
                                         'ysr': ysr},
                                        ignore_index=True)

        else:
            pass
    return df_iv


def keep_spikes_by_stim(blks,
                        keep='stim'):
    '''
    Keep spikes that are either within ('stim') or between
    ('nostim') stimulations
    
    Params
    ------
    blks : list of neo.Blocks
    keep : 'stim', 'nostim'
    '''
    blks_new = []
    for blk in blks:
        blk_new = copy.deepcopy(blk)
        seg = blk_new.segments[0]
        epc = seg.epochs[0]
        stim_start = epc.times
        stim_stop = epc.times + epc.durations
        units = blk_new.channel_indexes[0].children
        for unit in units:
            spktr = unit.spiketrains[0]
            spkts = spktr.times
            spkts_start = spktr.t_start.rescale(spkts.units)
            spkts_stop = spktr.t_stop.rescale(spkts.units)
            stim_start = stim_start.rescale(spkts.units)
            stim_stop = stim_stop.rescale(spkts.units)
            if keep == 'stim':
                new_ts = multislice(spktr.times,
                                    stim_start,
                                    stim_stop)
            elif keep == 'nostim':
                nostim_start = np.concatenate(
                    [[spkts_start], stim_stop]
                    )*spkts.units
                nostim_stop = np.concatenate(
                    [stim_start, [spkts_stop]]
                    )*spkts.units
                new_ts = multislice(
                    spktr.times,
                    nostim_start,
                    nostim_stop
                )
            else:
                raise ValueError('value of variable keep not recognized')
            spktr = spktr.duplicate_with_new_data(
                new_ts,
                t_start=spktr.t_start,
                t_stop=spktr.t_stop)
            unit.spiketrains[0] = spktr
        blks_new.append(blk_new)
    return blks_new


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
        intens = epc.annotations['intensity']
        hist, bins = np.histogram(intens, bins)
        kernel = np.array([1]*kernel_width)
        kernel = kernel/np.sum(kernel)
        hist = np.convolve(hist, kernel, mode='same')
        labels = measure.label(hist > threshold)
        labels_unique = np.unique(labels)
        epc.annotations['stim_group'] = np.zeros(len(epc.times))
        for label_i in labels_unique:
            if label_i >= 1:
                bins_i = bins[:-1][labels == label_i]
                intens_start = np.min(bins_i)
                intens_stop = np.max(bins_i)
                intens_bool = np.logical_and(
                    intens >= intens_start,
                    intens <= intens_stop)

                epc.annotations['stim_group'][intens_bool] = label_i

    return blks


def select_only_first_spike(blks,
                            same_shank_only=False):
    """
    For each spiketrain select only first spike upon a stimulation.
    """
    blks_new = copy.deepcopy(blks)
    for blk in blks_new:
        seg = blk.segments[0]
        epc = seg.epochs[0]
        units = blk.channel_indexes[0].children
        for unit in units:
            shank = unit.annotations['shank']
            if same_shank_only:
                epc_sel = select_times_from_epoch(
                    epc,
                    {'annotations.shank': shank})
            else:
                epc_sel = epc

            stim_ts = epc_sel.times
            spktr = unit.spiketrains[0]
            spk_ts = spktr.times
            pos = np.searchsorted(spk_ts, stim_ts)
            pos = np.unique(pos)
            spk_ts_new = spk_ts[pos]
            spktr_new = spktr.duplicate_with_new_data(
                spk_ts_new,
                t_start=spktr.t_start,
                t_stop=spktr.t_stop)
            unit.spiketrains[0] = spktr_new
    return blks_new
                

def find_significant_stimulations(blks,
                                  stimccg_binsize,
                                  stimccg_limit,
                                  condition_annot_unit={}):
    df_cch = pd.DataFrame()

    for blk in blks:
        seg = blk.segments[0]
        epc = seg.epochs[0]
        units = blk.channel_indexes[0].children
        stim_shanks = np.unique(epc.annotations['shank'])
        stim_groups = np.unique(epc.annotations['stim_group'])
        stim_groups = stim_groups[stim_groups != 0]
        for shank in stim_shanks:
            for group_i in stim_groups:
                intens_bool = epc.annotations['stim_group'] == group_i
                intens_sel = epc.annotations['intensity'][intens_bool]
                intens_start = np.min(intens_sel) - 1
                intens_stop = np.max(intens_sel) + 1
                intens_mean = np.mean(intens_sel)
                epc_sel = select_times_from_epoch(
                    epc,
                    {'annotations.shank': shank,
                     'annotations.intensity': [intens_start,
                                               intens_stop]})
                for unit_i in units:
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
                                        limit=stimccg_limit)
                        rate_baseline = np.mean(cch[bins < 0.])
                        pfast = np.zeros(cch.shape)
                        for m, val_m in enumerate(cch):
                            pfast[m] = pcc(np.array([val_m]), rate_baseline)
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
                            'rate_baseline': rate_baseline},
                                               ignore_index=True)
    return df_cch



    
        
    
