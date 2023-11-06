# Import necessary packages & set-up plotting aesthetics
import numpy as np 
import pylab
import matplotlib
import matplotlib.pyplot as plt
import json
import pandas as pd
import scipy
from scipy.interpolate import LinearNDInterpolator
from astropy.cosmology import Planck13 as cosmo 
H0 = cosmo.H(0) 
from astropy.constants import c

# set up interpolants
reference_zs = np.linspace(0.001,2,5000)
reference_Dcs = cosmo.comoving_distance(reference_zs)
reference_dDc_dzs = c / (H0 * cosmo.efunc(reference_zs))

import sys
sys.path.append('/home/simona.miller/.conda/envs/emcee/lib/python3.6/site-packages')
import kalepy as kale

sys.path.append('/home/simona.miller/measuring-bbh-component-spin/Code/PopulationInference/')
from posterior_helper_functions import *

"""
This script is janky, please ignore!
"""

def calculate_chiEff(chi1, chi2, cost1, cost2, q): 
    chieff = (chi1*cost1 + q*chi2*cost2)/(1+q)
    return chieff

def calculate_chiP(chi1, chi2, cost1, cost2, q): 
    sint1 = np.sin(np.arccos(cost1))
    sint2 = np.sin(np.arccos(cost1))
    term1 = chi1*sint1
    term2 = (2 + 4*q)/(4 + 3*q)*q*chi2*sint2
    chip = np.maximum(term1,term2)
    return chip 

def p_astro_z(z, dV_dz, kappa=2.7):
    p_z = dV_dz*np.power(1.+z,kappa-1.)
    return p_z

def dDL_dz(z): 
    # dD_L/dz = D_c + (1+z) dD_c/dz  = D_c + (1+z) c /( H0 * E(z) )  
    D_c = np.interp(z, reference_zs, reference_Dcs)
    dDc_dz = np.interp(z, reference_zs, reference_dDc_dzs)
    dDLdz =  D_c + (1+z)*dDc_dz
    return dDLdz


def betaDoubleGaussian(spin_samps, spinModelArgs):
    
    # individual event samples
    chi_samps, cost_samps = spin_samps
    
    # hyper pe sample
    a, b, mu1, sigma1, mu2, sigma2, f = spinModelArgs
    
    # calculate distribution
    p_chi = calculate_betaDistribution(chi_samps, a, b)
    p_cost = calculate_Double_Gaussian(chi_samps, mu1, sigma1, mu2, sigma2, f, -1, 1)
    return p_chi*p_cost



def plot_pp_error_bars(ax, N, number_x_values=1001):
    """
    param: N , int, number of independent draw from prior runs 
    """
    
    confidence_interval = [0.68, 0.95, 0.997]
    x_values = np.linspace(0, 1, number_x_values)
    confidence_interval_alpha = 0.1
    if isinstance(confidence_interval_alpha, float):
        confidence_interval_alpha = [confidence_interval_alpha] * len(confidence_interval)

    # stolen from https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/result.py#L2102
    # make_pp_plot
    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1. - ci) / 2.
        # each bin ((1-ci)/2, (1+ci)/2) describes probability of falling in that bound 
        # not sure exactly what is happening here...
        lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 at 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color='k')
        
        
        
def pop_reweight(sampleDict, samplesFromtargetDistribution, nTarget=1000): 

    # dict in which to put reweighted individual event samples
    sampleDict_rw = {}
    
    # unpack target spin distribution 
    target_chis = np.concatenate([samplesFromtargetDistribution['a1'], samplesFromtargetDistribution['a2']])    
    target_costs = np.concatenate([samplesFromtargetDistribution['cost1'], samplesFromtargetDistribution['cost2']])
    
    # make KD
    target_spins_kde = kale.KDE([target_chis, target_costs], reflect=[[0,1], [-1,1]])
    
    # cycle through events
    events = [event for event in sampleDict.keys()]
    for k, event in enumerate(events): 
        
        print(f"reweighting event {k+1} of {len(sampleDict)}", end='\r')
        
        # Unpack posterior samples for this event
        chi1_samples = np.asarray(sampleDict[event]['a1'])
        chi2_samples =  np.asarray(sampleDict[event]['a2'])
        cost1_samples = np.asarray(sampleDict[event]['cost1'])
        cost2_samples = np.asarray(sampleDict[event]['cost2'])
        m1_samples = np.asarray(sampleDict[event]['m1'])
        m2_samples = np.asarray(sampleDict[event]['m2'])
        z_samples = np.asarray(sampleDict[event]['z'])
        dVdz_samples = np.asarray(sampleDict[event]['dVc_dz'])
        
        # Number of posterior samples for this event
        nSamples = len(chi1_samples)
        
        '''
        New priors = population distribution
        '''

        # spins
        _, new_spin1_prior = target_spins_kde.density([chi1_samples,cost1_samples], probability=True)
        _, new_spin2_prior = target_spins_kde.density([chi2_samples,cost2_samples], probability=True)

        # masses
        new_m1_m2_prior = p_astro_masses(m1_samples, m2_samples, mCut=8)
        
        # redshifts
        new_z_prior = p_astro_z(z_samples, dV_dz=dVdz_samples)        
        
        # multiply them all together
        new_prior = new_spin1_prior*new_spin2_prior*new_m1_m2_prior*new_z_prior

        '''
        Do reweighting
        '''

        # calculate weights = ratio of priors
        old_prior = np.asarray(sampleDict[event]['bilby_prior'])
        weights = new_prior/old_prior
        weights = weights/np.sum(weights)
        
        nSamples = len(weights)

        # select random subset of samples from the event posterior subject to these weights
        idxs = np.random.choice(nSamples, size=min(nTarget, nSamples), p=weights)
        
        # Add into reweighted sampleDict
        sampleDict_rw[event] = {
            'chi1':chi1_samples[idxs].tolist(),
            'chi2':chi2_samples[idxs].tolist(),
            'cost1':cost1_samples[idxs].tolist(),
            'cost2':cost2_samples[idxs].tolist(),
            'm1':m1_samples[idxs].tolist(),
            'm2':m2_samples[idxs].tolist(),
            'z':z_samples[idxs].tolist(),
            'injected_params':sampleDict[event]['injected_params']
        }

    return sampleDict_rw


def pop_reweight2(sampleDict, spinModel, modelArgs, nTarget=1000): 

    # dict in which to put reweighted individual event samples
    sampleDict_rw = {}
    
    # cycle through events
    events = [event for event in sampleDict.keys()]
    for k, event in enumerate(events): 
        
        print(f"reweighting event {k+1} of {len(sampleDict)}", end='\r')
        
        # Unpack posterior samples for this event
        chi1_samples = np.asarray(sampleDict[event]['a1'])
        chi2_samples =  np.asarray(sampleDict[event]['a2'])
        cost1_samples = np.asarray(sampleDict[event]['cost1'])
        cost2_samples = np.asarray(sampleDict[event]['cost2'])
        m1_samples = np.asarray(sampleDict[event]['m1'])
        m2_samples = np.asarray(sampleDict[event]['m2'])
        z_samples = np.asarray(sampleDict[event]['z'])
        dVdz_samples = np.asarray(sampleDict[event]['dVc_dz'])
        
        # Number of posterior samples for this event
        nSamples = len(chi1_samples)
        
        '''
        New priors = population distribution
        '''

        # spins
        spinModelArgs = modelArgs[:-1]
        new_spin1_prior = spinModel([chi1_samples, cost1_samples], spinModelArgs)
        new_spin2_prior = spinModel([chi2_samples, cost2_samples], spinModelArgs)

        # masses
        new_m1_m2_prior = p_astro_masses(m1_samples, m2_samples, mCut=8, bq=modelArgs[-1])
        
        # redshifts
        new_z_prior = p_astro_z(z_samples, dV_dz=dVdz_samples)        
        
        # multiply them all together
        new_prior = new_spin1_prior*new_spin2_prior*new_m1_m2_prior*new_z_prior

        '''
        Do reweighting
        '''

        # calculate weights = ratio of priors
        old_prior = np.asarray(sampleDict[event]['bilby_prior'])
        weights = new_prior/old_prior
        weights = weights/np.sum(weights)
        
        nSamples = len(weights)

        # select random subset of samples from the event posterior subject to these weights
        idxs = np.random.choice(nSamples, size=min(nTarget, nSamples), p=weights)
        
        # Add into reweighted sampleDict
        sampleDict_rw[event] = {
            'chi1':chi1_samples[idxs].tolist(),
            'chi2':chi2_samples[idxs].tolist(),
            'cost1':cost1_samples[idxs].tolist(),
            'cost2':cost2_samples[idxs].tolist(),
            'm1':m1_samples[idxs].tolist(),
            'm2':m2_samples[idxs].tolist(),
            'z':z_samples[idxs].tolist(),
            'injected_params':sampleDict[event]['injected_params']
        }

    return sampleDict_rw


if __name__ == "__main__":
    
#     # True populations
#     f_root = '/home/simona.miller/measuring-bbh-component-spin/Data/InjectedPopulationParameters/' 
#     true_pops = {}
    
#     true_pops['1'] = pd.read_json(f_root+'underlying_population1_highSpinPrecessing_fullmassrange.json')
#     true_pops['2'] = pd.read_json(f_root+'underlying_population2_mediumSpin_fullmassrange.json')
#     true_pops['3'] = pd.read_json(f_root+'underlying_population3_lowSpinAligned_fullmassrange.json')
    
#     # Format true pop spin parameters correctly
#     pop_numbers = ['1', '2', '3']
#     true_pops_transformed = {}
#     for pop in pop_numbers: 

#         true_pop = true_pops[pop]

#         m1 = true_pop.m1
#         m2 = true_pop.m2
#         a1 = np.sqrt(true_pop.s1x**2 + true_pop.s1y**2 + true_pop.s1z**2)
#         a2 = np.sqrt(true_pop.s2x**2 + true_pop.s2y**2 + true_pop.s2z**2)
#         cost1 = true_pop.s1z/a1
#         cost2 = true_pop.s2z/a2

#         # mass cutoff 
#         mask = (m1 > 8) & (m2 > 8)

#         true_pops_transformed[pop] = {
#             'a1':a1[mask],
#             'a2':a2[mask],
#             'cost1':cost1[mask],
#             'cost2':cost2[mask]
#         }

    
#     # PE samples individual events
#     f_root = '/home/simona.miller/measuring-bbh-component-spin/Data/PopulationInferenceInput/' 
#     sampleDicts = {}
#     with open(f_root+'sampleDict_population1_highSpinPrecessing.json', 'r') as f:
#         sampleDicts['1'] = json.load(f)
#     with open(f_root+'sampleDict_population2_mediumSpin.json', 'r') as f:
#         sampleDicts['2'] = json.load(f)
#     with open(f_root+'sampleDict_population3_lowSpinAligned.json', 'r') as f:
#         sampleDicts['3'] = json.load(f)
        
    
#     # Reweighting
#     sampleDicts_rw = {}
#     for pop in ['1', '2', '3']:
#         print('\npopulation '+pop)

#         # get the correct individual event posterior samples
#         sampleDict_temp = sampleDicts[pop]
#         sampleDict = {}
#         for event in sampleDict_temp.keys():
#             d1 = {p:sampleDict_temp[event][p] for p in ['m1', 'm2', 'z', 'dVc_dz', 'injected_params', 'bilby_prior']}
#             d2 = {p:sampleDict_temp[event][p]['bilby_posterior'] for p in ['a1', 'a2', 'cost1', 'cost2']}
#             sampleDict[event] = {**d1, **d2}

#         # get the target distribution = full detected distribution 
#         target_pop_samples = true_pops_transformed[pop]
        
#         # reweight and add to over-all dict
#         sampleDicts_rw[pop] =  pop_reweight(sampleDict, target_pop_samples)    

    
#         # save as we go 
#         savename = 'for_pp_plots/102723_reweighted_to_truth.json'
#         with open(savename, "w") as outfile:
#             json.dump(sampleDicts_rw, outfile)
            

    
    # PE samples individual events
    f_root = '/home/simona.miller/measuring-bbh-component-spin/Data/PopulationInferenceInput/' 
    sampleDicts = {}
    with open(f_root+'sampleDict_population1_highSpinPrecessing.json', 'r') as f:
        sampleDicts['1'] = json.load(f)
    with open(f_root+'sampleDict_population2_mediumSpin.json', 'r') as f:
        sampleDicts['2'] = json.load(f)
    with open(f_root+'sampleDict_population3_lowSpinAligned.json', 'r') as f:
        sampleDicts['3'] = json.load(f)
        
    # Hyperparams to test against    
    hyperparams = { 
        '1':[1.46, 1.2, 0.13, 0.25, 0.66, 0.74, 0.7, 0.96],  
        '2':[2.1, 5.66, -0.22, 0.6, 0.75, 0.27, 0.44, 0.96],
        '3':[3.42, 14.58, -0.99, 0.51, 0.99, 0.29, 0.28, 0.96],
    }

    # Reweighting
    sampleDicts_rw = {}
    for pop in ['1', '2', '3']:
        print('\npopulation '+pop)

        # get the correct individual event posterior samples
        sampleDict_temp = sampleDicts[pop]
        sampleDict = {}
        for event in sampleDict_temp.keys():
            d1 = {p:sampleDict_temp[event][p] for p in ['m1', 'm2', 'z', 'dVc_dz', 'injected_params', 'bilby_prior']}
            d2 = {p:sampleDict_temp[event][p]['gaussian_sigma_0.1'] for p in ['a1', 'a2', 'cost1', 'cost2']}
            sampleDict[event] = {**d1, **d2}
        
        # reweight and add to over-all dict
        sampleDicts_rw[pop] =  pop_reweight2(sampleDict, betaDoubleGaussian, hyperparams[pop])    

    
        # save as we go 
        savename = 'for_pp_plots/110123_sigma_0.1_reweighted_to_measured.json'
        with open(savename, "w") as outfile:
            json.dump(sampleDicts_rw, outfile)
            
    print(f'Done -- saved at {savename}')