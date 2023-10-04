import numpy as np
import sys
from scipy.special import erf
from scipy.special import beta
import json 

import astropy.cosmology as cosmo
import astropy.units as u
from astropy.cosmology import Planck15

from posterior_helper_functions import * 

"""
Function to do posterior reweighting 

    sampleDict = dictionary containing individual event samples
    hyperPEDict = dictionary containing hyperparemeter samples: from json file that emcee outputs
    
"""

def pop_reweight(sampleDict, hyperPEDict, model): 
    
    # Number of hyperparameter samples
    nHyperPESamps = len(hyperPEDict['mu_chi']['processed'])
    
    # dict in which to put reweighted individual event samples
    sampleDict_rw = {}
    
    # cycle through events
    for k, event in enumerate(sampleDict): 
        
        print(f"reweighting event {k+1} of {len(sampleDict)}: {event}", end='\r')
        
        # Unpack posterior samples for this event
        chi1_samples = np.asarray(sampleDict[event]['a1'])
        chi2_samples =  np.asarray(sampleDict[event]['a2'])
        cost1_samples = np.asarray(sampleDict[event]['cost1'])
        cost2_samples = np.asarray(sampleDict[event]['cost2'])
        m1_samples = np.asarray(sampleDict[event]['m1'])
        m2_samples = np.asarray(sampleDict[event]['m2'])
        z_samples = np.asarray(sampleDict[event]['z'])
        dVdz_samples = np.asarray(sampleDict[event]['dVc_dz'])

        # indices corresponding to each sample for these events (will be used below in the for loop)
        nSamples = len(chi1_samples)
        indices = np.arange(nSamples)
        
        # arrays in which to store reweighted samples for this event
        new_chi1_samps = np.zeros(nHyperPESamps)
        new_chi2_samps = np.zeros(nHyperPESamps)
        new_cost1_samps = np.zeros(nHyperPESamps)
        new_cost2_samps = np.zeros(nHyperPESamps)
        new_mass1_samps = np.zeros(nHyperPESamps)
        new_mass2_samps = np.zeros(nHyperPESamps)
        
        # cycle through hyper PE samps
        for i in range(nHyperPESamps):
            
            '''iterating through each curve of the trace plot'''
            
            if model == 'betaPlusGaussian':
                
                # Fetch i^th hyper PE sample: 
                mu_chi = hyperPEDict['mu_chi']['processed'][i]
                sigma_chi = hyperPEDict['sigma_chi']['processed'][i]
                mu_cost = hyperPEDict['mu_cost']['processed'][i]
                sigma_cost = hyperPEDict['sigma_cost']['processed'][i]
                Bq = hyperPEDict['Bq']['processed'][i]

                # Translate mu_chi and sigma_chi to beta function parameters a and b 
                # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
                a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)

                # Evaluate model at the locations of samples for this event
                p_chi1 = calculate_betaDistribution(chi1_samples, a, b)
                p_chi2 = calculate_betaDistribution(chi2_samples, a, b)
                p_cost1 = calculate_Gaussian_1D(cost1_samples, mu_cost, sigma_cost, -1, 1)
                p_cost2 = calculate_Gaussian_1D(cost2_samples, mu_cost, sigma_cost, -1, 1)
                
            elif model == 'betaPlusDoubleGaussian':
                
                # Fetch i^th hyper PE sample: 
                mu_chi = hyperPEDict['mu_chi']['processed'][i]
                sigma_chi = hyperPEDict['sigma_chi']['processed'][i]
                mu1_cost = hyperPEDict['mu1_cost']['processed'][i]
                sigma1_cost = hyperPEDict['sigma1_cost']['processed'][i]
                mu2_cost = hyperPEDict['mu2_cost']['processed'][i]
                sigma2_cost = hyperPEDict['sigma2_cost']['processed'][i]
                MF_cost = hyperPEDict['MF_cost']['processed'][i]
                Bq = hyperPEDict['Bq']['processed'][i]

                # Translate mu_chi and sigma_chi to beta function parameters a and b 
                # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
                a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)

                # Evaluate model at the locations of samples for this event
                p_chi1 = calculate_betaDistribution(chi1_samples, a, b)
                p_chi2 = calculate_betaDistribution(chi2_samples, a, b)
                p_cost1 = calculate_Double_Gaussian(cost1_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1)
                p_cost2 = calculate_Double_Gaussian(cost2_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1)
            
        
            # Pop dist for all four params combined is a product of each four individual dists
            pSpins = p_chi1*p_chi2*p_cost1*p_cost2
            
            # PE priors for chi_i and cost_i are all uniform, so we set them to unity here
            nSamples = pSpins.size
            spin_PE_prior = np.ones(nSamples)
            
            # Need to reweight by astrophysical priors on m1, m2, z ...
            # - p(m1)*p(m2)
            p_astro_m1_m2 = p_astro_masses(m1_samples, m2_samples, bq=Bq, mCut=8)
            old_m1_m2_prior = np.power(1.+z_samples, 2) # PE prior on masses is uniform in DETECTOR FRAME component masses
            # - p(z)
            p_astro_redshift = p_astro_z(z_samples, dV_dz=dVdz_samples)
            old_z_prior = p_astro_z(z_samples, dV_dz=dVdz_samples, kappa=0) # see: bilby.gw.prior.UniformSourceFrame prior 
            # - For full m1, m2, z prior reweighting: 
            m1_m2_z_prior_ratio = (p_astro_m1_m2/old_m1_m2_prior)*(p_astro_redshift/old_z_prior) 
            
            # calculate weights for this hyper parameter
            weights = pSpins*m1_m2_z_prior_ratio/spin_PE_prior
            weights = weights/np.sum(weights)
            
            # select a random sample from the event posterior subject to these weights
            j = np.random.choice(indices, p=weights)
            
            # populate the new sample arrays with this random sample
            new_chi1_samps[i] = chi1_samples[j]
            new_chi2_samps[i] = chi2_samples[j]
            new_cost1_samps[i] = cost1_samples[j]
            new_cost2_samps[i] = cost2_samples[j]
            new_mass1_samps[i] = m1_samples[j]
            new_mass2_samps[i] = m2_samples[j]
        
        # Add into reweighted sampleDict
        downsample_idxs = np.random.choice(indices, size=nHyperPESamps)
        sampleDict_rw[event] = {
            'original':{
                'chi1':np.asarray(sampleDict[event]['a1'])[downsample_idxs].tolist(),
                'chi2':np.asarray(sampleDict[event]['a2'])[downsample_idxs].tolist(),
                'cost1':np.asarray(sampleDict[event]['cost1'])[downsample_idxs].tolist(), 
                'cost2':np.asarray(sampleDict[event]['cost2'])[downsample_idxs].tolist(),
                'm1':np.asarray(sampleDict[event]['m1'])[downsample_idxs].tolist(),
                'm2':np.asarray(sampleDict[event]['m2'])[downsample_idxs].tolist()
            },
            'reweighted':{
                'chi1':new_chi1_samps.tolist(),
                'chi2':new_chi2_samps.tolist(),
                'cost1':new_cost1_samps.tolist(),
                'cost2':new_cost2_samps.tolist(),
                'm1':new_mass1_samps.tolist(),
                'm2':new_mass2_samps.tolist()
            }
        }

    return sampleDict_rw


"""
Actually loading and running pop reweighting
"""

if __name__=="__main__":
    
    # To translate between pop # and full name used in saving
    pop_names_dict = {
        '1':'population1_highSpinPrecessing', 
        '2':'population2_mediumSpin', 
        '3':'population3_lowSpinAligned'
    }
    
    # Directory home 
    froot = '../../Data/'
    
    # Run settings we want to reweight
    date = '092823'
    models = ['betaPlusGaussian']
    pops = ['3']
    nevents = ['70', '300']
    posterior_keys = {
        '70': ['bilby_posterior', 'gaussian_sigma_0.1', 'gaussian_sigma_0.2', 'gaussian_sigma_0.3', 
                      'gaussian_sigma_0.4', 'gaussian_sigma_0.5', 'gaussian_sigma_1'], 
        '300': ['bilby_posterior']

    }
    
    print('\n') 
        
    # Cycle through them all 
    for model in models:
        for pop in pops: 
            for nevent in nevents:
                for posterior_key in posterior_keys[nevent]:

                    print(f'{date}, {model}, pop {pop}, {nevent} events, {posterior_key} samples')

                    # Cycle through runs we want to reweight
                    if posterior_key != 'bilby_posterior':
                        filename = f'{date}_{model}_pop{pop}_{nevent}events_{posterior_key}'
                    else: 
                        filename = f'{date}_{model}_pop{pop}_{nevent}events'

                    # Load dict with individual event PE samples (Load sampleDict):
                    pop_name = pop_names_dict[pop]
                    with open(froot+f'PopulationInferenceInput/sampleDict_{pop_name}.json') as f:
                        sampleDict_full = json.load(f) 

                    # Load population parameter PE samps
                    with open(froot+f'PopulationInferenceOutput/{model}/'+filename+'.json', 'r') as f:
                        hyperPEDict = json.load(f)

                    # Select only events from sampleDict used in this emcee run
                    events = hyperPEDict['events_used']
                    sampleDict = {}
                    for event in events:
                        # for masses and redshifts always use bilby posteriors
                        d1 = {p:sampleDict_full[event][p] for p in ['m1', 'm2', 'z', 'dVc_dz']}
                        # for spin magnitude and tilts, option to use bilby or gaussian posteriors
                        d2 = {p:sampleDict_full[event][p][posterior_key] for p in ['a1', 'a2', 'cost1', 'cost2']}
                        # combine into final sampleDict
                        sampleDict[event] = {**d1, **d2}

                    # Run reweighting 
                    sampleDict_rw = pop_reweight(sampleDict, hyperPEDict, model)   

                    # Save results
                    with open(froot+'PopulationInferenceOutput/for_pp_plots/'+filename+'_reweighted_sampleDict.json', "w") as f:        
                        json.dump(sampleDict_rw,f)

                    print('\n') 