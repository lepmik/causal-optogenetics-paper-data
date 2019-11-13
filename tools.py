import os.path as op
import pandas as pd
import numpy as np

def csv_append_dict(fname, dictionary):
    assert isinstance(dictionary, dict)
    df = pd.DataFrame([dictionary])
    if not fname.endswith('.csv'):
        fname = fname + '.csv'
    if not op.exists(fname):
        df.to_csv(fname, index=False, mode='w', header=True)
    else:
        df.to_csv(fname, index=False, mode='a', header=False)


def create_connection_matrix(p, seed=None):
    import brian2 as br2
    """ create_connection_matrix creates a numpy array that
    holds synaptic weights.

    Params
    ---------
    p: dict, Parameter dictionary

    Returns
    ---------
    m: np.ndarray, connection matrix
    """

    if seed:
        np.random.seed(seed)

    def mu(mean):
        return np.log(mean / np.sqrt(1 + p['p_var'] / mean**2))

    def sigma(mean):
        return np.sqrt(np.log(1 + p['p_var'] / mean**2))

    print('Finding excitatory and inhibitory projections')
    print(
        'J_ex = ', p['J_ex'], 'C_ex = ', p['C_ex'],
        'J_in = ', p['J_in'], 'C_in = ', p['C_in'])

    # weights without units 
    J_ex = p['J_ex']/br2.nS
    J_in = p['J_in']/br2.nS
    J_high = p['J_high']/br2.nS
    J_low = p['J_low']/br2.nS

    # initialize connection matrix
    n_nrns = p['N_ex']+p['N_in']
    m = np.zeros((n_nrns, n_nrns))
    
    # find sources for all targets

    for j in range(n_nrns):
        # connections from excitatory neurons

        # avoid ex autapses
        range_ex = np.arange(0, p['N_ex'])        
        range_ex = range_ex[range_ex != j]

        i = np.random.choice(
            range_ex,
            size=p['C_ex'],
            replace=False)

        # lognormal distribution of synaptic weights for EE syns
        if j < p['N_ex']:
            J_ex_j = np.clip(
                np.random.lognormal(
                    mean=mu(J_ex),
                    sigma=sigma(J_ex),
                    size=p['C_ex']),
                J_low,
                J_high)
            m[j, i] = J_ex_j
        else:
            m[j, i] = J_ex
        
        # connections from inhibitory neurons
        # avoid in autapses        
        range_in = np.arange(p['N_ex'], n_nrns)
        range_in = range_in[range_in != j]

        i=np.random.choice(
            range_in,
            size=p['C_in'],
            replace=False)
        m[j, i] = J_in

    # add nano siemens again
    m = m * br2.nS

    return m
