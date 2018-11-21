import pandas as pd

def read_file(LOC, taste, property_type):
    md = pd.read_csv(LOC/('{}-data/model-data-{}.tsv'.format(taste, property_type)), sep='\t')
    gs =pd.read_csv(LOC/('{}-data/gold-standard-{}.tsv'.format(taste, property_type)), sep='\t')
    md.drop(list(set(['orig_idx', 'No.']) & set(md.columns)), axis=1, inplace=True)
    gs.drop(list(set(['orig_idx', 'No.']) & set(gs.columns)), axis=1, inplace=True)
    
    return md, gs
    
            
def read_after_boruta(BORUTA_LOC, LOC, taste, property_type):
    md, gs = read_file(LOC, taste, property_type)
            
    # Get final decisions
    fd = pd.read_csv(BORUTA_LOC/('{}_{}_fd.tsv'.format(property_type, taste)), sep='\t', header=-1)
    confirmed_ftrs = fd.loc[fd[1] == 'Confirmed', 0]

    # Clean up columns
    confirmed_ftrs = confirmed_ftrs.map(lambda s: s.replace('`', '')).tolist()
            
    # Subset md and gs
    md = md.loc[:, list(md.columns)[:7] + confirmed_ftrs]
    gs = gs.loc[:, list(gs.columns)[:8] + confirmed_ftrs]
            
    return md, gs
            