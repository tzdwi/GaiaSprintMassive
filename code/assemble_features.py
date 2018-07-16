#requires python 2.7 (of course)
from __future__ import print_function, absolute_import
from WISE_tools import *
import FATS, pandas as pd, numpy as np
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from astropy.modeling.functional_models import Gaussian2D
from astropy.modeling import fitting
from scipy.odr import *
from sklearn.metrics import r2_score

def KDE_fit(x,y):
    """
    A function that performs a gaussian KDE on x/y data, then fits the result with a sum
    of 2 2D gaussians
    
    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data
        
    Returns
    -------
    KDE_bandwidth : float
        Cross-validated kernel bandwidth
    KD_fit_sqresid : float
        The mean squared residual (fit - KDE)^2
    amp_0 : float
        amplitude of largest gaussian
    xmean_0 : float
        x mean of largest gaussian
    ymean_0 : float
        y mean of largest gaussian
    major_std_0 : float
        Standard deviation along major axis of largest gaussian
    theta_0 : float
        Radians between positive X-axis and the major axis of the largest gaussian
    ecc_0 : float
        Eccentricity of an ellipse, taken at an arbitrary height of the largest gaussian
    amp_1 : float
        amplitude of smallest gaussian
    xmean_1 : float
        x mean of smallest gaussian
    ymean_1 : float
        y mean of smallest gaussian
    major_std_1 : float
        Standard deviation along major axis of smallest gaussian
    theta_1 : float
        Radians between positive X-axis and the major axis of the largest gaussian
    ecc_1 : float
        Eccentricity of an ellipse, taken at an arbitrary height of the smallest gaussian
    """
    
    data = np.vstack([x, y]).T
    #Grid search for best KDE bandwidth
    params = {'bandwidth': np.linspace(np.min(np.diff(y)),np.max(np.diff(y)),100)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(data)
    
    KDE_bandwidth = grid.best_estimator_.bandwidth
    
    kde = grid.best_estimator_
    X, Y = np.meshgrid(np.linspace(np.min(x),np.max(x),100), np.linspace(np.min(y),np.max(y),100))

    xy = np.vstack([X.ravel(), Y.ravel()]).T
    #compute the KDE on a 100x100 grid of points
    Z = np.exp(kde.score_samples(xy)).reshape(X.shape)
    
    #fit KDE estimation with 2 Gaussian model
    g2D_init1 = Gaussian2D(amplitude=np.max(Z), x_mean=X[np.unravel_index(np.argmax(Z),Z.shape)], y_mean=Y[np.unravel_index(np.argmax(Z),Z.shape)], x_stddev=np.std(x), y_stddev=np.std(y), theta=0, bounds={'theta': (0,np.pi),'x_mean': (np.min(x),np.max(x)),'y_mean': (np.min(y),np.max(y)),'x_stddev':(0.001,1),'y_stddev':(0.001,1)})
    g2D_init2 = Gaussian2D(amplitude=np.median(Z), x_mean=np.median(x), y_mean=np.median(y), x_stddev=np.std(x), y_stddev=np.std(y), theta=0, bounds={'theta': (0,np.pi),'x_mean': (np.min(x),np.max(x)),'y_mean': (np.min(y),np.max(y)),'x_stddev':(0.001,1),'y_stddev':(0.001,1)})
    g2D_init = g2D_init1 + g2D_init2

    fitter = fitting.LevMarLSQFitter()
    
    g2D = fitter(g2D_init, X, Y, Z)
    
    KD_fit_sqresid = np.mean(np.power(Z-g2D(X,Y),2.0))
    
    #Sort by largest and smallest amplitude gaussian
    i_large = np.argmax([g2D.amplitude_0,g2D.amplitude_1])
    i_small = np.argmin([g2D.amplitude_0,g2D.amplitude_1])
    g2D_large = g2D[i_large]
    g2D_small = g2D[i_small]
    
    amp_0 = g2D_large.amplitude.value
    amp_1 = g2D_small.amplitude.value
    
    xmean_0 = g2D_large.x_mean.value
    xmean_1 = g2D_small.x_mean.value
    
    ymean_0 = g2D_large.y_mean.value
    ymean_1 = g2D_small.y_mean.value
    
    if g2D_large.x_stddev >= g2D_large.y_stddev:
        
        major_std_0 = g2D_large.x_stddev.value
        theta_0 = g2D_large.theta.value
        ecc_0 = np.sqrt(1.0 - (g2D_large.y_stddev.value/g2D_large.x_stddev.value)**2.0)
    
    else:
        
        major_std_0 = g2D_large.y_stddev.value
        
        if g2D_large.theta <= np.pi/2:
            theta_0 = np.pi/2 + g2D_large.theta.value
            
        elif g2D_large.theta > np.pi/2:
            theta_0 = g2D_large.theta.value - np.pi/2
             
        ecc_0 = np.sqrt(1.0 - (g2D_large.x_stddev.value/g2D_large.y_stddev.value)**2.0)
        
    if g2D_small.x_stddev >= g2D_small.y_stddev:
        
        major_std_1 = g2D_small.x_stddev.value
        theta_1 = g2D_small.theta.value
        ecc_1 = np.sqrt(1.0 - (g2D_small.y_stddev.value/g2D_small.x_stddev.value)**2.0)
    
    else:
        
        major_std_1 = g2D_small.y_stddev.value
        
        if g2D_small.theta <= np.pi/2:
            theta_1 = np.pi/2 + g2D_small.theta.value
            
        elif g2D_small.theta > np.pi/2:
            theta_1 = g2D_small.theta.value - np.pi/2
             
        ecc_1 = np.sqrt(1.0 - (g2D_small.x_stddev.value/g2D_small.y_stddev.value)**2.0)
        
    return (KDE_bandwidth, KD_fit_sqresid, amp_0, xmean_0, ymean_0, major_std_0, theta_0,
            ecc_0, amp_1, xmean_1, ymean_1, major_std_1, theta_1, ecc_1)


def line(p, x):
    m, b = p
    return m*x + b

def linear_CMD_fit(x,y,xerr,yerr):
    """
    Does a linear fit to CMD data where x is color and y is amplitude, returning some fit 
    statistics
    
    Parameters
    ----------
    x : array-like
        color
    y : array-like
        magnitude
    xerr : array-like
        color errors
    yerr : array-like
        magnitude errors
    
    Returns
    -------
    slope : float
        slope of best-fit line
    r_squared : float
        Correlation coefficient (R^2)
    """
    
    data = RealData(x, y, sx=xerr, sy=yerr)
    
    mod = Model(line)
    
    odr = ODR(data, mod, beta0=[-0.1, np.mean(y)])
    out = odr.run()
    
    slope = out.beta[0]
    
    r_squared = r2_score(y, line(out.beta, x))
    
    return slope, r_squared


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
    
    #Now let's add our own features
    #KDE in CMD space
    x = df['w1mpro'].values - df['w2mpro'].values
    xerr = np.sqrt(df['w1sigmpro'].values*df['w1sigmpro'].values + df['w2sigmpro'].values*df['w2sigmpro'].values)
    
    y = df['w1mpro'].values
    yerr = df['w1sigmpro'].values
    
    (KDE_bandwidth, KD_fit_sqresid, amp_0, xmean_0, ymean_0, major_std_0, theta_0,
            ecc_0, amp_1, xmean_1, ymean_1, major_std_1, theta_1, ecc_1) = KDE_fit(x,y)
    
    result['KDE_bandwidth'] = KDE_bandwidth
    result['KD_fit_sqresid'] = KD_fit_sqresid
    result['KD_amp_0'] = amp_0
    result['KD_xmean_0'] = xmean_0
    result['KD_ymean_0'] = ymean_0
    result['KD_major_std_0'] = major_std_0
    result['KD_theta_0'] = theta_0
    result['KD_ecc_0'] = ecc_0
    result['KD_amp_1'] = amp_1
    result['KD_xmean_1'] = xmean_1
    result['KD_ymean_1'] = ymean_1
    result['KD_major_std_1'] = major_std_1
    result['KD_theta_1'] = theta_1
    result['KD_ecc_1'] = ecc_1
    
    slope, r_squared = linear_CMD_fit(x, y, xerr, yerr)
    result['CMD_slope'] = slope
    result['CMD_r_squared'] = r_squared
    
    result['Name'] = name
    
    return pd.DataFrame(data=np.array(result.values()).reshape(1,len(result.values())), 
                        columns=result.keys())

if __name__ == '__main__':
    import warnings
    from sys import argv
    from os.path import exists
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    
        assert len(argv) == 4, "This takes three arguments: the data directory with trailing /, the output csv, and the number of cores to use"

        data_dir = argv[1]
        outfile = argv[2]
        ncores = int(argv[3])

        assert data_dir[-1] == '/', "The last character of the data directory should be /"

        assert exists(data_dir), "Make sure the data directory exists"

        names = parse_source_names(data_dir)

        p = Pool(ncores)
        dfs = p.map(get_features,names)
        out = pd.concat(dfs)
        out.to_csv(outfile)