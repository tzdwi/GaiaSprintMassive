import numpy as np, pandas as pd
from glob import glob
from matplotlib import pyplot as plt
from scipy.optimize import minimize

def parse_source_names(directory):
    """
    Gets all of the WISE source names from a directory containing files associated with WISE 
    lightcurves. Lightcurve data should be named in the format 'WISE Jhhmmss.ss[+/-]ddmmss.s*'.
    
    Parameter
    ---------
    directory : str
        Name of the directory to search through for files, in the form path/to/directory/. Can be 
        relative or absolute, but must end with a /.
        
    Returns
    -------
    names : list
        List with source names in the form Jhhmmss.ss[+/-]ddmmss.s
    
    """
    
    files = glob(directory+'*')
    source_names = []
    for file in files:
        if file[len(directory):len(directory)+5] == 'WISE ':
            wise_id = file[len(directory)+5:len(directory)+24]
            source_names.append(wise_id)
    names = np.unique(source_names)
    
    return names

def get_lightcurve(name, directory, clean=True):
    
    """
    From a WISE source name, creates a lightcurve stored in a pandas DataFrame. Lightcurve
    will be cleaned
    
    Parameters
    ----------
    name : str
        WISE source name, in the format Jhhmmss.ss+/-ddmmss.s
    directory : str
        Name of the directory to search through for files, in the form path/to/directory/. Can be 
        relative or absolute, but must end with a /.
    clean : bool
        If True, cleans the data using quality flags in the individual csv files used
        to create the lightcurve. Default True.
        
    Returns
    -------
    df : `~pandas.DataFrame`
    
    """
    
    dfs = []
    #Get relevant data files
    all_csvs = glob(directory+'*{}*.csv'.format(name))
    for csv in all_csvs:
        #read data
        data = pd.read_csv(csv)
        if clean:
            #kinda hacky cleaning
            ok = (
                (data['ph_qual'].str[0] == 'A') &
                (data['nb'] == 1) &
                ((data['cc_flags'].astype(object).str[0:2].astype(str) == '00')|(data['cc_flags'].astype(object) == 0)) &
                (data['w1rchi2'] < 5)
                 )
            if 'qual_frame' in data.columns:
                ok = ok & (data['qual_frame'] > 8)
                
            dfs.append(data[ok])
        else:
            dfs.append(data)
            
    #final LC concatenated and sorted on MJD
    df = pd.concat(dfs)
    df.sort_values('mjd',inplace=True)
    
    return df