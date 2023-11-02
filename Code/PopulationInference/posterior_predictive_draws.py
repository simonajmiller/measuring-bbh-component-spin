import numpy as np 
import pylab

import json
import random
import sys 
import os

sys.path.append('../../Figures/')
from fig_helper_functions import *

'''
Load data
'''

# PopulationInferenceOutput data directory
f_root = '../../Data/PopulationInferenceOutput/'

# For loading emcee samples
date = '101923'
#model = 'betaPlusDoubleGaussian'
model = 'betaPlusGaussian'
pops = ['1', '2', '3']
nevents = ['70', '300']
posterior_keys = {
    '70': ['bilby_posterior', 'gaussian_sigma_0.1', 'gaussian_sigma_0.3', 'gaussian_sigma_0.5'], 
    '300': ['bilby_posterior', 'gaussian_sigma_0.1', 'gaussian_sigma_0.3', 'gaussian_sigma_0.5'], 
    
}

# Make dicts
all_emcee_data = {}
all_sampleDicts_rw = {}

# Cycle through the different populations, number of events, and runs
for pop in pops: 
    
    emcee_data_pop = {}
    rw_data_pop = {}
    
    for nevent in nevents:
        
        emcee_data_event = {}
        rw_event_data = {}
        
        for posterior_key in posterior_keys[nevent]:
                
            if posterior_key!='bilby_posterior':
                name = f"{date}_{model}_pop{pop}_{nevent}events_{posterior_key}"
            else: 
                name = f"{date}_{model}_pop{pop}_{nevent}events"
                
            # emcee output
            with open(f_root+f"{model}/{name}.json", 'r') as f:
                emcee_data_event[posterior_key] =  json.load(f)

            # weighted sample dicts corresponding to the above emcee samples
            with open(f_root+f"for_pp_plots/{name}_reweighted_sampleDict.json", 'r') as f:
                rw_event_data[posterior_key] = json.load(f)
        
        emcee_data_pop[nevent+'events'] = emcee_data_event 
        rw_data_pop[nevent+'events'] = rw_event_data 
    
    all_emcee_data['pop'+pop] = emcee_data_pop
    all_sampleDicts_rw['pop'+pop] = rw_data_pop
    
# Load injectionDict
with open(f_root+"../PopulationInferenceInput/injectionDict.json", 'r') as f: 
    injectionDict = json.load(f)

# Unpack injections
chi1s = np.asarray(injectionDict['a1'])
chi2s = np.asarray(injectionDict['a2'])
cost1s = np.asarray(injectionDict['cost1'])
cost2s = np.asarray(injectionDict['cost2'])
m1s = np.asarray(injectionDict['m1'])
m2s = np.asarray(injectionDict['m2'])
zs = np.asarray(injectionDict['z'])
dVdzs = np.asarray(injectionDict['dVdz'])

# Draw probability for component spins, masses, + redshift
p_draw = np.asarray(injectionDict['p_draw_a1a2cost1cost2'])*np.asarray(injectionDict['p_draw_m1m2z'])

# Number of items in injectonDict
ninjections = len(chi1s)


'''
Generate predicted versus observed draws from populations
'''

# Number of catalogs to create
nCatalogs = 1000

# Create dict for predicted versus observed "catalogs" 
pred_vs_obs_spins = {}

# Cycle through populations
pop_keys = ['pop'+p for p in pops]
nevents_keys = [int(n) for n in nevents]

for pop_key in pop_keys:

    # Print progress
    print(pop_key)
    pred_vs_obs_spins_pop = {}
    
    for nevents in nevents_keys:
        
        pred_vs_obs_spins_pop_run = {}
                        
        for posterior_key in posterior_keys[str(nevents)]:

            # Print progress
            events_key = str(nevents)+'events'
            print(events_key, posterior_key)

            # Fetch emcee data for this population
            emcee_data = all_emcee_data[pop_key][events_key][posterior_key]

            # Create arrays in which to store the predicted spin parameters
            chi1_pred = np.zeros((nCatalogs, nevents))
            chi2_pred = np.zeros((nCatalogs, nevents))
            cost1_pred = np.zeros((nCatalogs, nevents))
            cost2_pred = np.zeros((nCatalogs, nevents))
            chieff_pred = np.zeros((nCatalogs, nevents))
            chip_pred = np.zeros((nCatalogs, nevents))

            # And the observed spin parameters 
            chi1_obs = np.zeros((nCatalogs, nevents))
            chi2_obs = np.zeros((nCatalogs, nevents))
            cost1_obs = np.zeros((nCatalogs, nevents))
            cost2_obs = np.zeros((nCatalogs, nevents))
            chieff_obs = np.zeros((nCatalogs, nevents))
            chip_obs = np.zeros((nCatalogs, nevents))

            # Generate nCatalog instantiations of "catalogs"
            for run in np.arange(nCatalogs):
                
                print(run, end='\r')

                '''
                "Predicted" spins
                '''

                # Choose numinjections hyperparameters and draw a sample from each corresponding distribution
                nhyperparams = len(emcee_data['mu_chi']['processed'])
                random_indices = np.random.choice(nhyperparams, size=nevents, replace=False)

                for i,ind in enumerate(random_indices):
                    
                    if model=='betaPlusGaussian':

                        # Fetch parameters 
                        mu_chi = emcee_data['mu_chi']['processed'][ind]
                        sigma_chi = emcee_data['sigma_chi']['processed'][ind]
                        mu_cost = emcee_data['mu_cost']['processed'][ind]
                        sigma_cost = emcee_data['sigma_cost']['processed'][ind]
                        Bq = emcee_data['Bq']['processed'][ind]

                        # transform from mu and sigma to a and b for beta distribution
                        a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2)

                        # Calculate weights on values from injectionDict
                        p_chi1 = calculate_betaDistribution(chi1s, a, b)
                        p_chi2 = calculate_betaDistribution(chi2s, a, b)
                        p_cost1 = calculate_Gaussian_1D(cost1s, mu_cost, sigma_cost, -1, 1)
                        p_cost2 = calculate_Gaussian_1D(cost2s, mu_cost, sigma_cost, -1, 1)
                        
                    elif model=='betaPlusDoubleGaussian': 
                        
                        # Fetch parameters 
                        mu_chi = emcee_data['mu_chi']['processed'][ind]
                        sigma_chi = emcee_data['sigma_chi']['processed'][ind]
                        mu_cost1 = emcee_data['mu1_cost']['processed'][ind]
                        sigma_cost1 = emcee_data['sigma1_cost']['processed'][ind]
                        mu_cost2 = emcee_data['mu2_cost']['processed'][ind]
                        sigma_cost2 = emcee_data['sigma2_cost']['processed'][ind]
                        frac = emcee_data['MF_cost']['processed'][ind]
                        Bq = emcee_data['Bq']['processed'][ind]

                        # transform from mu and sigma to a and b for beta distribution
                        a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2)

                        # Calculate weights on values from injectionDict
                        p_chi1 = calculate_betaDistribution(chi1s, a, b)
                        p_chi2 = calculate_betaDistribution(chi2s, a, b)
                        p_cost1 = calculate_Double_Gaussian(cost1s, mu_cost1, sigma_cost1, mu_cost2, sigma_cost2, frac,  -1, 1)
                        p_cost2 = calculate_Double_Gaussian(cost2s, mu_cost1, sigma_cost1, mu_cost2, sigma_cost2, frac,  -1, 1)
                        
                    p_masses = p_astro_masses(m1s, m2s, bq=Bq, mCut=8)
                    p_z = p_astro_z(zs, dV_dz=dVdzs)

                    weights = p_chi1*p_chi2*p_cost1*p_cost2*p_masses*p_z/p_draw
                    weights_normed = weights/np.sum(weights)
                    weights_normed[np.where(weights_normed<0)] = 0 # get rid of tiny division errors

                    # Select a sample with probabilities given by the normalized weights
                    idx = np.random.choice(ninjections, p=weights_normed, size=1) 
                    chi1 = chi1s[idx]
                    chi2 = chi2s[idx]
                    cost1 = cost1s[idx]
                    cost2 = cost2s[idx]

                    # Calculate chi_eff 
                    q = m2s[idx]/m1s[idx]
                    chi_eff = calculate_chiEff(chi1, chi2, cost1, cost2, q)

                    # Calculate chi_p
                    sint1 = np.sin(np.arccos(cost1))
                    sint2 = np.sin(np.arccos(cost2))
                    chi_p = calculate_chiP(chi1, chi2, sint1, sint2, q)

                    # Add to arrays
                    chi1_pred[run,i] = chi1
                    chi2_pred[run,i] = chi2
                    cost1_pred[run,i] = cost1
                    cost2_pred[run,i] = cost2
                    chieff_pred[run,i] = chi_eff
                    chip_pred[run,i] = chi_p


                '''
                Observed spins
                '''

                # Load reweighted individual event samples
                sampleDict_rw = all_sampleDicts_rw[pop_key][events_key][posterior_key]

                # Cycle through events
                for j, name in enumerate(sampleDict_rw.keys()):

                    d = sampleDict_rw[name]['reweighted']

                    # Select a random sample
                    idx = random.choice(np.arange(len(d['chi1'])))
                    chi1 = d['chi1'][idx]
                    chi2 = d['chi2'][idx]
                    cost1 = d['cost1'][idx]
                    cost2 = d['cost2'][idx]

                    # Calculate chi_eff
                    q = d['m2'][idx]/d['m1'][idx]
                    chi_eff = calculate_chiEff(chi1, chi2, cost1, cost2, q)

                    # Calculate chi_p
                    sint1 = np.sin(np.arccos(cost1))
                    sint2 = np.sin(np.arccos(cost2))
                    chi_p = calculate_chiP(chi1, chi2, sint1, sint2, q)

                    # Add to arrays
                    chi1_obs[run,j] = chi1
                    chi2_obs[run,j] = chi2
                    cost1_obs[run,j] = cost1
                    cost2_obs[run,j] = cost2
                    chieff_obs[run,j] = chi_eff
                    chip_obs[run,j] = chi_p

            # Add results to dict 
            pred_vs_obs_spins_pop_run[posterior_key] = {
                'predicted':{
                    'chi1':chi1_pred.tolist(),
                    'chi2':chi2_pred.tolist(),
                    'cost1':cost1_pred.tolist(),
                    'cost2':cost2_pred.tolist(),
                    'chieff':chieff_pred.tolist(), 
                    'chip':chip_pred.tolist()
                },
                'observed':{
                    'chi1':chi1_obs.tolist(),
                    'chi2':chi2_obs.tolist(),
                    'cost1':cost1_obs.tolist(),
                    'cost2':cost2_obs.tolist(),
                    'chieff':chieff_obs.tolist(), 
                    'chip':chip_obs.tolist()
                }
            }
                                
        pred_vs_obs_spins_pop[events_key] = pred_vs_obs_spins_pop_run

    pred_vs_obs_spins[pop_key] = pred_vs_obs_spins_pop

# Save
with open(f_root+f'for_pp_plots/{date}_{model}_pred_vs_obs_spins_dict_pp.json', "w") as f:
    json.dump(pred_vs_obs_spins, f)