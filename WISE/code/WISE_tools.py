import numpy as np, pandas as pd
from glob import glob
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

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
        #return(data)
        if clean:
            #kinda hacky cleaning
            ok = (
                (data['ph_qual'].str[0] == 'A') &
                (data['nb'] == 1) &
                ((data['cc_flags'].astype(object).astype(str).str[0:2] == '00')|(data['cc_flags'].astype(object).astype(str).str[0:2] == '0')) &
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
    
    if clean:
        df.dropna(subset=['w1mpro','w2mpro','w1sigmpro','w2sigmpro'], inplace=True)
    
    df['w1w2'] = df['w1mpro'].values - df['w2mpro'].values
    df['w1w2err'] = np.sqrt(df['w1sigmpro'].values**2.0 + df['w2sigmpro'].values**2.0)
    
    return df

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          normalize_axis=1,
                          title=None,
                          cmap=plt.cm.Blues,
                          classes=None,
                          figsize=(10,10)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    If normalize is True and normalize_axis is 1, returns the confusion matrix. If normalize_axis is 0, 
    returns the efficiency matrix.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    assert normalize_axis in [0,1]

    # Only use the labels that appear in the data
    if classes is None:
        classes = unique_labels(y_true, y_pred)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred,labels=classes)
    if normalize:
        if normalize_axis==1:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        else:
            cm = cm.astype('float') / cm.sum(axis=0)
        np.nan_to_num(cm,copy=False)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylim = ax.get_xlim()[::-1], #trying this to get the ylimits not to cut off...
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    fig.tight_layout()
    fig.figsize = figsize
    return fig,ax

def plot_completeness_contamination(y_true, y_pred,
                              title=None,
                              c_dict=None,
                              m_dict=None,
                              classes=None,
                              figsize=(10,10),
                              **kwargs):
    """
    This function calculates the confusion matrix, normalizes over rows and columns, then plots the diagonals against each other
    If c_dict and m_dict are dictionaries with `classes` as keys, and valid colors or markers as values.
    """

    # Only use the labels that appear in the data
    if classes is None:
        classes = unique_labels(y_true, y_pred)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred,labels=classes)
    conf_mat = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    effi_mat = np.nan_to_num(cm.astype('float') / cm.sum(axis=0))
    
    comp = np.diag(conf_mat)
    cont = 1 - np.diag(effi_mat)

    fig, ax = plt.subplots()
    if c_dict is not None:
        cs = [c_dict[clas] for clas in classes]
    else:
        cs = ['k' for clas in classes]
    if m_dict is not None:
        ms = [m_dict[clas] for clas in classes]
    else:
        ms = ['o' for clas in classes]
            
    for com, con, clas, c, m in zip(comp, cont, classes, cs, ms):
        plt.scatter(com, con, c=c, marker=m, label=clas,**kwargs)
    # We want to show all ticks...
    ax.set(title=title,
           xlabel='Completeness',
           ylabel='Contamination')
    fig.tight_layout()
    fig.figsize = figsize
    return fig,ax