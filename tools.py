import os.path as op
import pandas as pd


def csv_append_dict(fname, dictionary):
    assert isinstance(dictionary, dict)
    df = pd.DataFrame([dictionary])
    if not fname.endswith('.csv'):
        fname = fname + '.csv'
    if not op.exists(fname):
        df.to_csv(fname, index=False, mode='w', header=True)
    else:
        df.to_csv(fname, index=False, mode='a', header=False)
