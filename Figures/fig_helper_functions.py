import numpy as np
import sys 

sys.path.append('/home/simona.miller/measuring-bbh-component-spin/Code/PopulationInference')
from posterior_helper_functions import *

def calculate_chiEff(chi1, chi2, cost1, cost2, q): 
    chieff = (chi1*cost1 + q*chi2*cost2)/(1+q)
    return chieff

def calculate_chiP(chi1, chi2, sint1, sint2, q): 
    term1 = chi1*sint1
    term2 = (2 + 4*q)/(4 + 3*q)*q*chi2*sint2
    chip = np.maximum(term1,term2)
    return chip 


def draw_chiEffs_and_chiPs_betaDoubleGauss(mu_chi, sigma_chi, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, Bq, n=1):
    
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
    p_masses = p_astro_masses(m1s, m2s, bq=Bq)
    
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



def draw_chiEffs_and_chiPs_betaGauss(mu_chi, sigma_chi, mu_cost, sigma_cost, Bq, n=1):
    
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
    p_masses = p_astro_masses(m1s, m2s, bq=Bq)
    
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