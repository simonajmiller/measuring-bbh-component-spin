import numpy as np
from scipy.stats import gaussian_kde
import math
import sys 
sys.path.append('/home/simona.miller/measuring-bbh-component-spin/Code/PopulationInference')
from posterior_helper_functions import *

def reflected_kde_1d(x, A, B, npoints=1000): 
    grid = np.linspace(A,B,npoints)
    kde_on_grid = gaussian_kde(x)(grid) + gaussian_kde(2*A-x)(grid) + gaussian_kde(2*B-x)(grid)
    return grid, kde_on_grid

def calculate_chiEff(chi1, chi2, cost1, cost2, q): 
    chieff = (chi1*cost1 + q*chi2*cost2)/(1+q)
    return chieff

def calculate_chiP(chi1, chi2, sint1, sint2, q): 
    term1 = chi1*sint1
    term2 = (2 + 4*q)/(4 + 3*q)*q*chi2*sint2
    chip = np.maximum(term1,term2)
    return chip 


def get_KDE_dict_spins(pop, npoints=500): 
    
    # spin magnitude
    chi1 = np.sqrt(pop.s1x**2 + pop.s1y**2 + pop.s1z**2)
    chi2 = np.sqrt(pop.s2x**2 + pop.s2y**2 + pop.s2z**2)
    chi = np.concatenate((chi1,chi2))
    chi_kde = reflected_kde_1d(chi, 0, 1, npoints=npoints)[1]
    
    # tilt angle
    cost1 = pop.s1z/chi1
    cost2 = pop.s2z/chi2
    cost = np.concatenate((cost1,cost2))
    cost_kde = reflected_kde_1d(cost, -1, 1, npoints=npoints)[1]
    
    # chi effective
    q = pop.m2/pop.m1
    chieff = calculate_chiEff(chi1, chi2, cost1, cost2, q)
    chieff_kde = reflected_kde_1d(chieff, -1, 1, npoints=npoints)[1]
    
    # chi p
    sint1 = np.sin(np.arccos(cost1))
    sint2 = np.sin(np.arccos(cost2))
    chip = calculate_chiP(chi1, chi2, sint1, sint2, q)
    chip_kde = reflected_kde_1d(chip, 0, 1, npoints=npoints)[1]
    
    # put KDEs into dict
    kde_dict = {
        'chi':chi_kde, 
        'cost':cost_kde, 
        'chieff':chieff_kde, 
        'chip':chip_kde
    }
    
    return kde_dict 


def draw_chiEffs_and_chiPs_betaDoubleGauss(mu_chi, sigma_chi, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, Bq,  mCut=None, n=1):
    
    # transform from mu and sigma to a and b for beta distribution
    a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2)
    
    # draw uniform component spins + masses
    nRandomDraws = 10000
    samp_idxs = np.arange(nRandomDraws)
    chi1s = np.random.rand(nRandomDraws)
    chi2s = np.random.rand(nRandomDraws)
    cost1s = np.random.rand(nRandomDraws)*2 - 1
    cost2s = np.random.rand(nRandomDraws)*2 - 1
    mAs = np.random.rand(nRandomDraws)*100
    mBs = np.random.rand(nRandomDraws)*100
    m1s = np.maximum(mAs, mBs)
    m2s = np.minimum(mAs, mBs)
    
    # calculate p(spins,masses) for these uniform samples, using functions from posterior_helper_functions.py
    p_chi1 = calculate_betaDistribution(chi1s, a, b)
    p_chi2 = calculate_betaDistribution(chi2s, a, b)
    p_cost1 = calculate_Double_Gaussian(cost1s, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1)
    p_cost2 = calculate_Double_Gaussian(cost2s, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1)
    p_masses = p_astro_masses(m1s, m2s, mCut=mCut, bq=Bq)
    
    weights = p_chi1*p_chi2*p_cost1*p_cost2*p_masses
    weights_normed = weights/np.sum(weights)
    weights_normed[np.where(weights_normed<0)] = 0 # get rid of tiny division errors
    
    # select a subset of the samples subject to the weights calculated from p(spins,masses)
    idxs = np.random.choice(samp_idxs, p=weights_normed, size=n)  
    
    # calculate chi-eff for these samples
    q = m2s[idxs]/m1s[idxs]
    chi_eff = calculate_chiEff(chi1s[idxs], chi2s[idxs], cost1s[idxs], cost2s[idxs], q)
    
    # and chi-p
    sint1s = np.sin(np.arccos(cost1s))
    sint2s = np.sin(np.arccos(cost2s))
    chip = calculate_chiP(chi1s[idxs], chi2s[idxs], sint1s[idxs], sint2s[idxs], q)
        
    return chi_eff, chip



def draw_chiEffs_and_chiPs_betaGauss(mu_chi, sigma_chi, mu_cost, sigma_cost, Bq, mCut=None, n=1):
    
    # transform from mu and sigma to a and b for beta distribution
    a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2)
    
    # draw uniform component spins + masses
    nRandomDraws = 10000
    samp_idxs = np.arange(nRandomDraws)
    chi1s = np.random.rand(nRandomDraws)
    chi2s = np.random.rand(nRandomDraws)
    cost1s = np.random.rand(nRandomDraws)*2 - 1
    cost2s = np.random.rand(nRandomDraws)*2 - 1
    mAs = np.random.rand(nRandomDraws)*100
    mBs = np.random.rand(nRandomDraws)*100
    m1s = np.maximum(mAs, mBs)
    m2s = np.minimum(mAs, mBs)
    
    # calculate p(spins,masses) for these uniform samples, using functions from posterior_helper_functions.py
    p_chi1 = calculate_betaDistribution(chi1s, a, b)
    p_chi2 = calculate_betaDistribution(chi2s, a, b)
    p_cost1 = calculate_Gaussian_1D(cost1s, mu_cost, sigma_cost, -1, 1)     
    p_cost2 = calculate_Gaussian_1D(cost2s, mu_cost, sigma_cost, -1, 1)     
    p_masses = p_astro_masses(m1s, m2s, mCut=mCut, bq=Bq)
    
    weights = p_chi1*p_chi2*p_cost1*p_cost2*p_masses
    weights_normed = weights/np.sum(weights)
    weights_normed[np.where(weights_normed<0)] = 0 # get rid of tiny division errors
    
    # select a subset of the samples subject to the weights calculated from p(spins,masses)
    idxs = np.random.choice(samp_idxs, p=weights_normed, size=n)  
    
    # calculate chi-eff for these samples
    q = m2s[idxs]/m1s[idxs]
    chi_eff = calculate_chiEff(chi1s[idxs], chi2s[idxs], cost1s[idxs], cost2s[idxs], q)
    
    # and chi-p
    sint1s = np.sin(np.arccos(cost1s))
    sint2s = np.sin(np.arccos(cost2s))
    chip = calculate_chiP(chi1s[idxs], chi2s[idxs], sint1s[idxs], sint2s[idxs], q)
        
    return chi_eff, chip


def findSlope(x_values, y_values): 
    
    X = np.asarray([[x, 1] for x in x_values])
    X_T = np.transpose(X)
        
    try:
        X_matrix = np.matmul(np.linalg.inv(np.matmul(X_T, X)), X_T)
    except: 
        return np.inf
    
    slope, intercept = np.matmul(X_matrix, y_values)
    
    return slope


def get_fraction_underpredicted(d, params=['chi1', 'cost1', 'chieff', 'chip'], nTrials=10): 
    
    ## Get slopes of pp plot
    slopes_dict = {}
            
    # Cycle through parameters
    for param_key in params:

        # Array in which to store slopes and corresponding 'x' values
        nCatalogs, nEvents = np.asarray(d['predicted'][param_key]).shape

        nCut = math.ceil(nEvents/70 * 4) # amount to cut off from ends for linear regression

        if param_key=='chi1' or param_key=='cost1': # combine chi1 + chi2, and cost1 +cost2
            nXVals = 2*nEvents - nCut
        else: 
            nXVals = nEvents - nCut
        slopes = np.zeros((nCatalogs, nXVals))
        x_vals = np.zeros((nCatalogs, nXVals))

        # Cycle through catalogs
        for i in range(nCatalogs): 

            # Sort spins
            if param_key=='chi1' or param_key=='cost1': 
                # combine chi1 + chi2, and cost1 +cost2
                pred_spins_arr = np.sort(np.concatenate((d['predicted'][param_key][i], 
                                                         d['predicted'][param_key[:-1]+'2'][i])))
                obs_spins_arr = np.sort(np.concatenate((d['observed'][param_key][i], 
                                                        d['observed'][param_key[:-1]+'2'][i])))
            else:
                pred_spins_arr = np.sort(d['predicted'][param_key][i])
                obs_spins_arr = np.sort(d['observed'][param_key][i])

            # Calculate slopes
            x_vals[i,:] = pred_spins_arr[int(nCut/2):-int(nCut/2)]
            slopes[i,:] = [findSlope(pred_spins_arr[j:j+nCut], 
                                     obs_spins_arr[j:j+nCut]) for j in range(nXVals)]

        # shape into proper form
        slopes_dict[param_key] = {
            'slopes':np.reshape(slopes, (nTrials, int(nCatalogs/nTrials), nXVals)),
            'x_vals':np.reshape(x_vals, (nTrials, int(nCatalogs/nTrials), nXVals)) 
        }
        
    ## From slopes, get fraction under predicted
    percentages_dict = {}
    
    # Cycle through parameters
    for param_key in params: 

        # Gets slopes and x vals for this population and parameters
        slopes = slopes_dict[param_key]['slopes']
        all_x_vals = slopes_dict[param_key]['x_vals']

        # Binning
        all_x = np.concatenate(np.concatenate(all_x_vals))
        xmin = np.min(all_x)
        xmax = np.max(all_x)
        xbins = np.linspace(xmin, xmax, 50)

        # Cycle through trials, each has it's own set of y values
        percentages_all = []

        for j in range(nTrials): 

            all_y = np.concatenate(slopes[j])
            all_x_this = np.concatenate(all_x_vals[j])

            # Calculate fraction in each bin above the line x=1
            percentages = []
            midpoints = []
            for i in np.arange(len(xbins)-1):

                lower_bound = xbins[i]
                upper_bound = xbins[i+1]
                mask = (all_x_this >= lower_bound) & (all_x_this <= upper_bound)
                x_vals = all_x_this[mask]
                y_vals = all_y[mask]

                # percentage of values below y=x, ignoring the points with only a few traces (< 5 % of catalogs)
                if len(x_vals) >= int(0.05*nCatalogs): 
                    percentages += [sum(y_vals < 1) / len(x_vals)]
                else: 
                    percentages += [np.inf] # will ignore when averaging

                # midpoint of bin
                midpoints += [0.5*(lower_bound+upper_bound)]

            percentages_all.append(percentages)     

        # get mean and std dev across trials
        per_transpose = np.transpose(percentages_all)
        means = np.zeros(len(midpoints))
        stds = np.zeros(len(midpoints))
        for i in np.arange(len(midpoints)): 
            mask = (per_transpose[i] != np.inf)
            per = per_transpose[i][mask]
            means[i] = np.mean(per)
            stds[i] = np.std(per)

        # make dict
        percentages_dict[param_key] = {'percentages':means, 'error':stds, 'x_vals':np.asarray(midpoints)}
        
    return slopes_dict, percentages_dict