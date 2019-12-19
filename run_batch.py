
def make_regressors(stim_data, connections, spikes, z1=-2, z2=0, x1=1, x2=3, y1=3, y2=7, yb1=-4, yb2=0):
    sender_ids = connections.query('weight >= 0').source.sort_values().unique()
    n_neurons = len(sender_ids)
    min_sender_ids = min(sender_ids)
    stim_times = np.array(stim_data['times'])

    X = np.zeros((len(stim_times), n_neurons))
    Y = np.zeros((len(stim_times), n_neurons))
    Z = np.zeros((len(stim_times), n_neurons))
    Yb = np.zeros((len(stim_times), n_neurons))

    senders = spikes.sender
    times = spikes.times

    for i, t in enumerate(stim_times):
        search = [
            t + x1, t + x2,
            t + y1, t + y2,
            t + z1, t + z2,
            t + yb1, t + yb2
        ]
        idx = np.searchsorted(times, search, side='right')
        X[i, senders[idx[0]: idx[1]] - min_sender_ids] = 1
        Y[i, senders[idx[2]: idx[3]] - min_sender_ids] = 1
        Z[i, senders[idx[4]: idx[5]] - min_sender_ids] = 1
        Yb[i, senders[idx[6]: idx[7]] - min_sender_ids] = 1

    return X, Y, Z, Yb


def compute_conditional_means(stim_data, connections, X, Y, Z, Yb):
    source_ids = stim_data['stim_nodes']['ex']
    sender_ids = connections.query('weight >= 0')
    min_sender_ids = min(sender_ids)
    target_ids = sender_ids.query('source not in @stim_ids')
    target_ids = target_ids.source.sort_values().unique()
    y_ref = pd.DataFrame(
        np.zeros((len(source_ids), len(target_ids))),
        index=stim_ids, columns=sender_ids)
    yb_ref = y_ref.copy()
    y_base = y_ref.copy()
    yb_base = y_ref.copy()
    y_respons = y_ref.copy()
    yb_respons = y_ref.copy()

    for source in source_ids:
        for target in target_ids:
            z = Z[:, int(source) - min_sender_ids]
            x = X[:, int(source) - min_sender_ids]
            y = Y[:, int(target) - min_sender_ids]
            yb = Yb[:, int(target) - min_sender_ids]

            y_ref.loc[source, target] = y[z==1].mean()
            yb_ref.loc[source, target] = yb[z==1].mean()

            y_base.loc[source, target] = y[x==0].mean()
            yb_base.loc[source, target] = yb[x==0].mean()

            y_respons.loc[source, target] = y[x==1].mean()
            yb_respons.loc[source, target] = yb[x==1].mean()

    return y_ref, yb_ref, y_base, yb_base, y_respons, yb_respons


if __name__ == '__main__':
    import sys
    import imp
    import os.path as op


    if len(sys.argv) == 5:
        data_path, param_module, connfile, n_runs = sys.argv[1:]
    else:
        raise IOError('Usage: "python simulator.py data_path parameters [connfile]')

    jobname = param_module.replace('.py', '')
    currdir = op.dirname(op.abspath(__file__))
    f, p, d = imp.find_module(jobname, [currdir])
    parameters = imp.load_module(jobname, f, p, d).parameters

    labels = 'y_ref, yb_ref, y_base, yb_base, y_respons, yb_respons'
    for n in range(n_runs):'
        seed = np.random.randint(1000, 9999)
        sim = Simulator(
            parameters,
            data_path=data_path,
            jobname=jobname,
            verbose=False,
            save_to_file=False
        )
        sim.simulate(
            state=False,
            progress_bar=True,
            connfile=connfile
        )
        spikes = pd.DataFrame(nest.GetStatus(sim.spikes_ex, 'events')[0])
        sim.dump_params_to_json(suffix=n)
        X, Y, Z, Yb = make_regressors(
            stim_data=sim.stim_data,
            connections=sim.connections,
            spikes=spikes
        )
        data = compute_conditional_means(
            stim_data=sim.stim_data,
            connections=sim.connections,
            X, Y, Z, Yb
        )
        for df, label in zip(data, labels):
            df.to_feather(
                sim.data_path / '{}_{}.feather'.format(label, n))
