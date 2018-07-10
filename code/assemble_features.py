#requires python 2.7 (of course)
from __future__ import print_function, absolute_import
from WISE_tools import *
from sys import argv
from os.path import exists
import FATS, pandas as pd, numpy as np

assert len(argv) == 4, "This takes three arguments: the data directory with trailing /, the output csv, and the number of cores to use"

data_dir = argv[1]
outfile = argv[2]
ncores = int(argv[3])

assert data_dir[-1] == '/', "The last character of the data directory should be /"

assert exists(data_dir), "Make sure the data directory exists"

names = parse_source_names(data_dir)

def get_features(name):
    """
    A function that is given a name, and then outputs all 68 features (plus the name)
    in a handy pandas dataframe.
    
    Parameter
    ---------
    Name : str
        WISE source name output by WISE_tools.parse_source_names
        
    Returns
    -------
    features : `~pandas.DataFrame`
        Dataframe of all 68 features, plus a column for the name.
    """
    
    df = get_lightcurve(name, data_dir)
    lc = np.array([df['w1mpro'].values,df['mjd'].values,df['w1sigmpro'].values,
                df['w2mpro'].values,df['w1mpro'].values,df['w2mpro'].values,
                df['mjd'].values,df['w1sigmpro'].values,df['w2sigmpro'].values])
    
    a = FATS.FeatureSpace(Data=['magnitude', 'time', 'error', 'magnitude2', 'error2'],
                           excludeList=[])
    a.calculateFeature(lc)
    result = a.result('dict')
    result['Name'] = name
    return pd.DataFrame(data=np.array(result.values()).reshape(1,len(result.values())), 
                        columns=result.keys())
p = Pool(ncores)
dfs = p.map(get_features,names)
out = pd.concat(dfs)
out.to_csv(outfile)