#requires python 2.7 (of course)
from __future__ import print_function, absolute_import
from WISE_tools import *
import FATS
import pandas as pd, numpy as np
import traceback
import argparse
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from astropy.modeling.functional_models import Gaussian2D
from astropy.modeling import fitting
from scipy.odr import *
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import celerite
from celerite import terms
import emcee as mc

#In celerite-land, this is a DRW Kernel
class DRWTerm(terms.RealTerm):
    parameter_names = ("log_sigma", "log_tau")

    def get_real_coefficients(self, params):
        log_sigma, log_tau = params
        sigma = np.exp(log_sigma)
        tau = np.exp(log_tau)
        return (
            sigma**2.0 , 1/tau,
        )

def mean_per_visit(time,mag,err,dt_tol=100):
    """
    Calculates the mean per-visit point.
    
    Assume some delta time over which something is considered a separate visit.
    """
    visits = []
    visit = np.array([[time[0],mag[0],err[0]]])
    for i in range(1,len(time)):
        dif = time[i] - time[i-1]
        if dif <= dt_tol:
            visit = np.append(visit,[[time[i],mag[i],err[i]]],axis=0)
        else:
            visits.append(visit)
            visit = np.array([[time[i],mag[i],err[i]]])
    visits.append(visit)
    visits = np.array(visits)
    mean_times = []
    mean_mags = []
    mean_errs = []
    for visit in visits:
        mean_times.append(np.mean(visit[:,0]))
        mean_mags.append(np.mean(visit[:,1]))
        mean_errs.append(np.sqrt(np.sum(np.power(visit[:,2],2.0)))/len(visit))
    return np.array(mean_times),np.array(mean_mags),np.array(mean_errs)


def DRW(times,mags,errs):
    """
    Does a quick DRW fit using celerite+emcee to the average points in the visit. Then takes that solution and does an emcee fit to the entire lightcurve.
    
    Parameters
    ----------
    times : array-like
    mags : array-like
    errs : array-like
    
    Returns
    -------
    cDRW_sigma : float
    cDRW_tau : float
    cDRW_mean : float
    DRW_sigma : float
    DRW_tau : float
    DRW_mean : float
    """
    import numpy as np
    
    mt,mm,me = mean_per_visit(times,mags,errs)
    
    #Bounds on sigma: the minimum error -> 10 times the range of mags
    #Bounds on tau: 0.25 times the minimum time difference -> 2 times the time baseline
    
    DRWbounds = dict(log_sigma=(np.log(np.min(me)), np.log(10.0*np.ptp(mm))), 
                 log_tau=(np.log(0.25*np.min(np.diff(mt))), np.log(2.0*np.ptp(mt))))
    
    #First guess on sigma: STD of points
    #First guess on tau: 0.5 * the time baseline
    kern = DRWTerm(log_sigma=np.std(mm), log_tau=np.log(0.5*np.ptp(mt)),bounds=DRWbounds)
    
    #Define and first compute of the DRW
    gp = celerite.GP(kern, mean=np.mean(mm), fit_mean = True)
    gp.compute(mt, me)
    
    #maximize likelihood, which requires autograd's numpy?
    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    def neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)

    def grad_neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.grad_log_likelihood(y)[1]
    
    global np
    autograd = __import__('autograd.numpy', globals(), locals()) 
    np = autograd.numpy
    
    import autograd.numpy as np
    
    soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                method="L-BFGS-B", bounds=bounds, args=(mm, gp))
    
    global np
    numpy = __import__('numpy', globals(), locals()) 
    np = numpy
    
    #Now for the emceee
    def clog_probability(params):
        gp.set_parameter_vector(params)
        lp = gp.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        return gp.log_likelihood(mm) + lp
    
    #Initialize walkers
    initial = np.array(soln.x)
    ndim, nwalkers = len(initial), 32
    csampler = mc.EnsembleSampler(nwalkers, ndim, clog_probability) #coarse
    
    
    #try the coarse
    try:
        #random seed
        np.random.seed(np.random.randint(0,100))
        #Burn in for 1000 steps
        p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
        p0, lp, _ = csampler.run_mcmc(p0, 1000)
    
        #Reset, randomize the seed, run for 5000
        csampler.reset()
        np.random.seed(np.random.randint(0,100))
        csampler.run_mcmc(p0, 5000)
    
        #flatten along step axis
        csamples = csampler.flatchain
        cDRW_sigma = np.exp(np.mean(csamples[:,0]))
        cDRW_tau = np.exp(np.mean(csamples[:,1]))
        cDRW_mean = np.mean(csamples[:,2])
        
    except:
        cDRW_sigma = np.nan
        cDRW_tau = np.nan
        cDRW_mean = np.nan
    
    #Now do the fine time sampling
    gp.compute(times, errs)
    
    def log_probability(params):
        gp.set_parameter_vector(params)
        lp = gp.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        return gp.log_likelihood(mags) + lp
    
    sampler = mc.EnsembleSampler(nwalkers, ndim, log_probability) #fine
    try:
        #random seed
        np.random.seed(np.random.randint(0,100))
        #Burn in for 1000 steps
        p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, 1000)
    
        #Reset, randomize the seed, run for 5000
        sampler.reset()
        np.random.seed(np.random.randint(0,100))
        sampler.run_mcmc(p0, 5000)
    
        #flatten along step axis
        samples = sampler.flatchain
        DRW_sigma = np.exp(np.mean(samples[:,0]))
        DRW_tau = np.exp(np.mean(samples[:,1]))
        DRW_mean = np.mean(samples[:,2])
        
    except:
        DRW_sigma = np.nan
        DRW_tau = np.nan
        DRW_mean = np.nan
    
    return cDRW_sigma,cDRW_tau,cDRW_mean,DRW_sigma,DRW_tau,DRW_mean


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


def get_features(name, data_dir, exclude_list = []):
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
                           excludeList=exclude_list)
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
    
    #Inverse slope, because it should be close to vertical in the CMD
    slope, r_squared = linear_CMD_fit(y, x, yerr, xerr)
    result['CMD_slope'] = slope
    result['CMD_r_squared'] = r_squared
    
    times = df['mjd'].values
    mags = df['w1mpro'].values
    errs = df['w1sigmpro'].values
    try:
        cDRW_sigma, cDRW_tau, cDRW_mean, DRW_sigma, DRW_tau, DRW_mean = DRW(times,mags,errs)
    except:
        cDRW_sigma, cDRW_tau, cDRW_mean, DRW_sigma, DRW_tau, DRW_mean = [np.nan for i in range(6)]
    
    result['cDRW_sigma'] = cDRW_sigma
    result['cDRW_tau'] = cDRW_tau
    result['cDRW_mean'] = cDRW_mean
    result['DRW_sigma'] = DRW_sigma
    result['DRW_tau'] = DRW_tau
    result['DRW_mean'] = DRW_mean
    
    result['Name'] = name
    
    return pd.DataFrame(data=np.array(result.values()).reshape(1,len(result.values())), 
                        columns=result.keys())

if __name__ == '__main__':
    import warnings
    from os.path import exists
    
    feature_names = ['Psi_eta', 'Q31_color', 'PercentAmplitude', 'MaxSlope', 'SmallKurtosis', 'KD_major_std_0', 'KD_theta_0', 'KD_fit_sqresid', 'StetsonJ', 'Eta_color', 'Meanvariance', 'StetsonL', 'Rcs', 'StetsonK', 'KD_xmean_0', 'FluxPercentileRatioMid65', 'Freq3_harmonics_amplitude_0', 'Freq3_harmonics_amplitude_1', 'Freq3_harmonics_amplitude_2', 'Freq3_harmonics_amplitude_3', 'AndersonDarling', 'KD_xmean_1', 'FluxPercentileRatioMid20', 'LinearTrend', 'Freq2_harmonics_rel_phase_3', 'Freq2_harmonics_rel_phase_2', 'Freq2_harmonics_rel_phase_1', 'Freq2_harmonics_rel_phase_0', 'FluxPercentileRatioMid50', 'Eta_e', 'Freq2_harmonics_amplitude_0', 'Freq1_harmonics_amplitude_2', 'Freq1_harmonics_amplitude_3', 'Freq1_harmonics_amplitude_0', 'Freq1_harmonics_amplitude_1', 'SlottedA_length', 'KD_ecc_0', 'Q31', 'CMD_r_squared', 'KD_amp_1', 'Freq2_harmonics_amplitude_2', 'Skew', 'CAR_tau', 'StructureFunction_index_32', 'Std', 'KD_ymean_0', 'MedianBRP', 'KD_ecc_1', 'Mean', 'KD_ymean_1', 'KD_theta_1', 'Beyond1Std', 'Psi_CS', 'KDE_bandwidth', 'Freq3_harmonics_rel_phase_2', 'Freq3_harmonics_rel_phase_3', 'Freq3_harmonics_rel_phase_0', 'Freq3_harmonics_rel_phase_1', 'Amplitude', 'KD_major_std_1', 'Freq2_harmonics_amplitude_1', 'FluxPercentileRatioMid35', 'Freq2_harmonics_amplitude_3', 'Con', 'CMD_slope', 'CAR_mean', 'KD_amp_0', 'PercentDifferenceFluxPercentile', 'Color', 'Period_fit', 'StructureFunction_index_21', 'Freq1_harmonics_rel_phase_0', 'Freq1_harmonics_rel_phase_1', 'Freq1_harmonics_rel_phase_2', 'Freq1_harmonics_rel_phase_3', 'PairSlopeTrend', 'CAR_sigma', 'Autocor_length', 'StructureFunction_index_31', 'MedianAbsDev', 'Gskew', 'FluxPercentileRatioMid80', 'PeriodLS', 'StetsonK_AC','cDRW_tau','cDRW_sigma','cDRW_mean','DRW_tau','DRW_sigma','DRW_mean']

    fail_d = {col:np.nan for col in feature_names}
    fail_d['Name'] = None
    fail_out = pd.DataFrame(data=np.array(fail_d.values()).reshape(1,len(fail_d.values())),             columns=fail_d.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir')
    parser.add_argument('--outfile')
    parser.add_argument('--ncores')
    parser.add_argument('--timeout')
    parser.add_argument('--exclude')
    args = parser.parse_args()
    
    data_dir = args.datadir
    outfile = args.outfile
    ncores = int(args.ncores)
    timeout = float(args.timeout)
    exclude_list = []
    if args.exclude is not None:
        exclude_list = args.exclude.split()
        fail_out.drop(exclude_list,axis=1)
        
    def safe_features(name):
        print(name)
        try:
            out = get_features(name,data_dir,exclude_list=exclude_list)
            return out
        except Exception as e:
            fail_out['Name'] = [name]
            print('Caught exception for {0}'.format(name))
            return fail_out
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        assert data_dir[-1] == '/', "The last character of the data directory should be /"

        assert exists(data_dir), "Make sure the data directory exists"

        names = parse_source_names(data_dir)
        
        results = []

        with ProcessPool(max_workers=ncores) as pool:
            future = pool.map(safe_features, names, timeout=timeout)

            iterator = future.result()

            # iterate over all results, if a computation timed out
            # print it and continue to the next result
            while True:
                try:
                    result = next(iterator)
                    results.append(result)
                except StopIteration:
                    break  
                except TimeoutError as error:
                    print("function took longer than {0} seconds".format(error.args[1]))


        out = pd.concat(results)
        out.to_csv(outfile)
