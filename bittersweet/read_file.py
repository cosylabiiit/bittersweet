import pandas as pd

def read_smiles_file(f):
    data = pd.read_csv(f, sep=',',
                       encoding='utf-8', header=None)
    return data


