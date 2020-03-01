import numpy as np
import pandas as pd
import sys
import imp
import os.path as op
import nest
from tqdm import tqdm
from pandarallel import pandarallel
import warnings
warnings.filterwarnings("ignore")

# Initialization
pandarallel.initialize(progress_bar=False)

z1 = -2
z2 = 0
x1 = 1
x2 = 3
y1 = 3
y2 = 10

shift = y2

zb1 = z1 - shift
zb2 = z2 - shift
xb1 = x1 - shift
xb2 = x2 - shift
yb1 = y1 - shift
yb2 = y2 - shift

# fnameout = path / 'conditional_means_3-7.feather'
# fnameout = path / 'conditional_means_4-10.feather'
# fnameout = 'conditional_means_z-(3_1)_y(3_7).feather'
# fnameout = 'conditional_means_z-(2_1)_y(35_10).feather'
# fnameout = 'conditional_means_z-(2_1)_y(3_7).feather'
# fnameout = 'conditional_means_z-(2_1)_y(3_10).feather'
fnameout = 'conditional_means_z-(2_0)_y(3_10).feather'

def compute_response(row, stim_times, spikes, a, b):
    senders = spikes.senders
    times = spikes.times
    t = stim_times[row.name]
    idx = np.searchsorted(times, [t + a, t + b], side='right')
    senders_spiked = senders[idx[0]: idx[1]]
    senders_set = senders_spiked[np.isin(senders_spiked, row.index)]
    row.loc[senders_set] = 1
    return row


def compute_conditional_means(row, X, Z, Y, Xb, Zb, Yb):

    x = X.loc[:, int(row.source)]
    z = Z.loc[:, int(row.source)]
    y = Y.loc[:, int(row.target)]
    xb = Xb.loc[:, int(row.source)]
    zb = Zb.loc[:, int(row.source)]
    yb = Yb.loc[:, int(row.target)]

    y_ref = y[(z==1) & (x==0)].mean()
    yb_ref = yb[(zb==1) & (xb==0)].mean()

    y_base = y[x==0].mean()
    yb_base = yb[xb==0].mean()

    y_respons = y[(z==0) & (x==1)].mean()
    yb_respons = yb[(zb==0) & (xb==1)].mean()

    return pd.Series({
        'n_ref': np.sum(z),
        'hit_rate_source': np.mean(x),
        'hit_rate_target': np.mean(y),
        'y_ref': y_ref,
        'yb_ref': yb_ref,
        'y_base': y_base,
        'yb_base': yb_base,
        'y_respons': y_respons,
        'yb_respons': yb_respons
    })


def read_gdf(path):
    spikes_ex, spikes_in = [], []
    for f in path.iterdir():
        if f.suffix == '.gdf':
            df = pd.read_csv(
                f, sep='\t', header=None).rename(columns={0:'senders', 1:'times'}).drop(columns=2)
            if f.stem.startswith('ex'):
                spikes_ex.append(df)

    spikes_ex = pd.concat(spikes_ex)
    return spikes_ex.sort_values(by=['times'])


if __name__ == '__main__':
    import pathlib
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    else:
        raise IOError('Usage: "python analyse_batch.py data_path')
    data_path = pathlib.Path(data_path)
    results = None
    for path in tqdm(sorted(list(data_path.glob('results_*')))):
        if not path.is_dir():
            print('Skipping', path)
            continue

        if (path / fnameout).exists():
            print('Skipping', path)
            continue

        stim_path = path / 'stimulation_data_0.npz'
        if not stim_path.exists():
            print('Skipping', path)
            continue

        spikes = read_gdf(path)

        stim_data = np.load(stim_path, allow_pickle=True)['data'][()]
        stim_times = stim_data['times']
        stim_amps = stim_data['stim_amps']
        stim_nodes = list(stim_amps.keys())
        # only pick excitatory neurons and targets that are not stimulated
        # use the ones from the first dataset
        if results is None:
            conn = pd.read_feather(path / 'connections_0.feather')
            sender_ids_ex = conn.query('weight >= 0').source.sort_values().unique() # excitatory neurons

            results = conn.query(
                'weight >= 0 and target in @sender_ids_ex and target not in @stim_nodes'
            )

            results.loc[:,'source_stimulated'] = results.source.isin(stim_nodes)
            results.loc[:,'target_stimulated'] = results.target.isin(stim_nodes)

            results['stim_amp_source'] = results.parallel_apply(
                lambda x: stim_amps.get(x.source, 0), axis=1)

            sample = results.query('weight > 0.01 and stim_amp_source > 1')
            sample['wr'] = sample.weight.round(3)
            sample = sample.drop_duplicates('wr')

            query = 'weight <= 0.01 and weight >= 0 and stim_amp_source > 1'
            sample_zero = results.query(query)
            sample_zero = sample_zero.sample(2000)
            results = pd.concat([sample, sample_zero])

            include_nodes = np.unique(
                np.concatenate([results.source.values, results.target.values]))
            include_nodes = np.sort(include_nodes)
            n_neurons = len(include_nodes)

        X = pd.DataFrame(
            np.zeros((len(stim_times), n_neurons)),
            columns=include_nodes
        )
        Y, Z, Xb, Yb, Zb = X.copy(), X.copy(), X.copy(), X.copy(), X.copy()
        X = X.parallel_apply(
            compute_response, stim_times=stim_times, spikes=spikes, a=x1, b=x2, axis=1)
        Z = Z.parallel_apply(
            compute_response, stim_times=stim_times, spikes=spikes, a=z1, b=z2, axis=1)
        Y = Y.parallel_apply(
            compute_response, stim_times=stim_times, spikes=spikes, a=y1, b=y2, axis=1)

        Xb = Xb.parallel_apply(
            compute_response, stim_times=stim_times, spikes=spikes, a=xb1, b=xb2, axis=1)
        Zb = Zb.parallel_apply(
            compute_response, stim_times=stim_times, spikes=spikes, a=zb1, b=zb2, axis=1)
        Yb = Yb.parallel_apply(
            compute_response, stim_times=stim_times, spikes=spikes, a=yb1, b=yb2, axis=1)

        results_ = results.join(results.parallel_apply(
            compute_conditional_means, axis=1,
            X=X, Z=Z, Y=Y, Xb=Xb, Zb=Zb, Yb=Yb,
            result_type='expand'
        ))

        results_.reset_index(drop=True).to_feather(path / fnameout)
