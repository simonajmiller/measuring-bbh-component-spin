import numpy as np
from posterior_helper_functions import *

def betaPlusDoubleGaussian(c,sampleDict,injectionDict,priorDict,only_prior=False): 
    
    """
    Implementation of the beta+doubleGaussian model: a component spin distribution with spin 
    magnitude as beta distribution and the cosine of the tilt angle as a double gaussian, for 
    inference within `emcee`.
    
    Model used in Section III of **PAPER TITLE**
    
    Parameters
    ----------
    c : `numpy.array`
        array containing hyper-parameter samples in the order: 
            [ mu_chi, sigma_chi, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, Bq ] 
        where 
        
        - mu_chi = mean of spin magnitude beta distribution
        
        - sigma_chi = std. dev. of spin magnitude distribution
        
        - mu1_cost = mean of gaussian 1 for cosine tilt distribution
        
        - sigma1_cost = std. dev. gaussian 1 for cosine tilt distribution
        
        - mu2_cost = mean of gaussian 2 for cosine tilt distribution
        
        - sigma2_cost = std. dev. of gaussian 2 for cosine tilt distribution
        
        - MF_cost = fraction in gaussian 1 for cosine tilt distribution
                
        - Bq = power law slope of the mass ratio distribution    
        
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    priorDict : dict
        Precomputed dictionary containing bounds for the priors on each hyper-parameter
    
    Returns
    -------
    logP : float
        log posterior for the input sample 'c'
    logL : float
        log likelihood for the input sample 'c'
    Neff : float
        effective sample number for injDict for the input sample 'c'
    minNsamp : float
        minimum effective sample number over all events in sampleDict for the input sample 'c'
    """
    
    # Make sure hyper-sample is the right length
    assert len(c)==8, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    mu_chi = c[0]
    sigma_chi = c[1]
    mu1_cost = c[2] 
    sigma1_cost = c[3]
    mu2_cost = c[4] 
    sigma2_cost = c[5]
    MF_cost = c[6]
    Bq = c[7]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    if mu_chi < priorDict['mu_chi'][0] or mu_chi > priorDict['mu_chi'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma_chi < priorDict['sigma_chi'][0] or sigma_chi > priorDict['sigma_chi'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    if mu1_cost < priorDict['mu_cost'][0] or mu1_cost > priorDict['mu_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma1_cost < priorDict['sigma_cost'][0] or sigma1_cost > priorDict['sigma_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    if mu2_cost < priorDict['mu_cost'][0] or mu2_cost > priorDict['mu_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma2_cost < priorDict['sigma_cost'][0] or sigma2_cost > priorDict['sigma_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif MF_cost < priorDict['MF_cost'][0] or MF_cost > priorDict['MF_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    
    # Additionally, to break the degeneracy between the two gaussians in the tilt distribution, 
    # impose that mu1_cost <= mu2_cost
    elif mu1_cost > mu2_cost:
        return -np.inf, -np.inf, -np.inf, -np.inf
    
    # If the sample falls inside our prior range, continue
    else:
        
        # Initialize log-likelihood
        logL = 0.
        
        # Translate mu_chi and sigma_chi to beta function parameters a and b 
        # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
        a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)
        
        # Impose cut on a and b: must be greater then or equal to 1 in order
        # for distribution to go to 0 at chi=0 and chi=1 (aka nonsingular)
        if a<=1. or b<=1.: 
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # Prior on Bq - gaussian centered at 0 with sigma=3
        logPrior = -(Bq**2)/18. 
        
        # --- Selection effects --- 
    
        # Unpack injections
        chi1_det = np.asarray(injectionDict['a1'])
        chi2_det = np.asarray(injectionDict['a2'])
        cost1_det = np.asarray(injectionDict['cost1'])
        cost2_det = np.asarray(injectionDict['cost2'])
        m1_det = np.asarray(injectionDict['m1'])
        m2_det = np.asarray(injectionDict['m2'])
        z_det = np.asarray(injectionDict['z'])
        dVdz_det = np.asarray(injectionDict['dVdz'])
        
        # Draw probability for component spins, masses, + redshift
        p_draw = np.asarray(injectionDict['p_draw_a1a2cost1cost2'])*np.asarray(injectionDict['p_draw_m1m2z'])
        
        # Detected spins
        p_chi1_det = calculate_betaDistribution(chi1_det, a, b)
        p_chi2_det = calculate_betaDistribution(chi2_det, a, b)
        p_cost1_det = calculate_Double_Gaussian(cost1_det, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
        p_cost2_det = calculate_Double_Gaussian(cost2_det, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
        pdet_spins = p_chi1_det*p_chi2_det*p_cost1_det*p_cost2_det
        
        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, bq=Bq, mCut=8)
        pdet_z = p_astro_z(z_det, dV_dz=dVdz_det)
        
        # Construct full weighting factors
        p_det = pdet_spins*pdet_masses*pdet_z
        det_weights = p_det/p_draw
        
        if np.max(det_weights)==0:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Neff = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Neff<=4*nEvents:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # This is where we return the prior instead of the posterior if only_prior==True
        elif only_prior: 
            return logPrior, -np.inf, Neff, -np.inf
        
        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logL += log_detEff
        
        # --- Loop across BBH events ---
        Nsamps = np.zeros(len(sampleDict)) 
        for i,event in enumerate(sampleDict):

            # Unpack posterior samples for this event
            chi1_samples = np.asarray(sampleDict[event]['a1'])
            chi2_samples =  np.asarray(sampleDict[event]['a2'])
            cost1_samples = np.asarray(sampleDict[event]['cost1'])
            cost2_samples = np.asarray(sampleDict[event]['cost2'])
            m1_samples = np.asarray(sampleDict[event]['m1'])
            m2_samples = np.asarray(sampleDict[event]['m2'])
            z_samples = np.asarray(sampleDict[event]['z'])
            dVdz_samples = np.asarray(sampleDict[event]['dVc_dz'])
            bilby_prior_samples = np.asarray(sampleDict[event]['bilby_prior'])
            
            # Evaluate model at the locations of samples for this event
            p_chi1 = calculate_betaDistribution(chi1_samples, a, b)
            p_chi2 = calculate_betaDistribution(chi2_samples, a, b)
            p_cost1 = calculate_Double_Gaussian(cost1_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
            p_cost2 = calculate_Double_Gaussian(cost2_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
            
            # Pop dist for all four params combined is a product of each four individual dists
            p_spins = p_chi1*p_chi2*p_cost1*p_cost2
                        
            # Need to reweight by astrophysical priors on m1, m2, z ...
            p_astro_m1_m2 = p_astro_masses(m1_samples, m2_samples, bq=Bq, mCut=8)
            p_astro_redshift = p_astro_z(z_samples, dV_dz=dVdz_samples)
            
            # Sum over probabilities to get the marginalized likelihood for this event
            det_weights_event = p_spins*p_astro_m1_m2*p_astro_redshift/bilby_prior_samples
            nSamples = det_weights_event.size
            pEvidence = (1.0/nSamples)*np.sum(det_weights_event)
            
            # Calculate effective sample number
            Nsamp_event = np.sum(det_weights_event)**2/np.sum(det_weights_event**2)
                        
            # Cut in effective samples for each event
            # if Nsamp_event <= 10: 
            #     return -np.inf, -np.inf, -np.inf, -np.inf
            
            Nsamps[i] = Nsamp_event

            # Add to our running total
            logL += np.log(pEvidence)
            
        # Get minimum effective sample number over events
        minNsamp = np.min(Nsamps)
        
        # Combine likelihood and prior to get posteriors
        logP = logL + logPrior

        if logP!=logP:
            return -np.inf, -np.inf, -np.inf, -np.inf

        else:
            return logP, logL, Neff, minNsamp
        
        
def betaPlusGaussian(c,sampleDict,injectionDict,priorDict,only_prior=False): 
    
    """
    Implementation of the Beta+Gaussian model: a component spin distribution with spin 
    magnitude as beta distribution and the cosine of the tilt angle as another 1D gaussian, for 
    inference within `emcee`. We do not include variance between the two gaussians.
    
    Model used in Section IV of **PAPER TITLE**
    
    Parameters
    ----------
    c : `numpy.array`
        array containing hyper-parameter samples in the order: 
            [ mu_chi, sigma_chi, mu_cost, sigma_cost, Bq ] 
        where 
        
        - mu_chi = mean of spin magnitude beta
        
        - sigma_chi = std. dev. of spin magnitude beta
        
        - mu_cost = mean of cosine tilt gaussian
        
        - sigma_cost = std. dev. of cosine tilt gaussian
                
        - Bq = power law slope of the mass ratio distribution    
        
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    priorDict : dict
        Precomputed dictionary containing bounds for the priors on each hyper-parameter
    
    Returns
    -------
    logP : float
        log posterior for the input sample 'c'
    logL : float
        log likelihood for the input sample 'c'
    Neff : float
        effective sample number for injDict for the input sample 'c'
    minNsamp : float
        minimum effective sample number over all events in sampleDict for the input sample 'c'
    """
    
    # Make sure hyper-sample is the right length
    assert len(c)==5, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    mu_chi = c[0]
    sigma_chi = c[1]
    mu_cost = c[2] 
    sigma_cost = c[3]
    Bq = c[4]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    if mu_chi < priorDict['mu_chi'][0] or mu_chi > priorDict['mu_chi'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma_chi < priorDict['sigma_chi'][0] or sigma_chi > priorDict['sigma_chi'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    if mu_cost < priorDict['mu_cost'][0] or mu_cost > priorDict['mu_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma_cost < priorDict['sigma_cost'][0] or sigma_cost > priorDict['sigma_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    
    # If the sample falls inside our prior range, continue
    else:
        
        # Initialize log-likelihood
        logL = 0.
        
        # Translate mu_chi and sigma_chi to beta function parameters a and b 
        # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
        a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)
        
        # Impose cut on a and b: must be greater then or equal to 1 in order
        # for distribution to go to 0 at chi=0 and chi=1 (aka nonsingular)
        if a<=1. or b<=1.: 
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # Prior on Bq - gaussian centered at 0 with sigma=3
        logPrior = -(Bq**2)/18. 
        
        # --- Selection effects --- 
    
        # Unpack injections
        chi1_det = np.asarray(injectionDict['a1'])
        chi2_det = np.asarray(injectionDict['a2'])
        cost1_det = np.asarray(injectionDict['cost1'])
        cost2_det = np.asarray(injectionDict['cost2'])
        m1_det = np.asarray(injectionDict['m1'])
        m2_det = np.asarray(injectionDict['m2'])
        z_det = np.asarray(injectionDict['z'])
        dVdz_det = np.asarray(injectionDict['dVdz'])
        
        # Draw probability for component spins, masses, + redshift
        p_draw = np.asarray(injectionDict['p_draw_a1a2cost1cost2'])*np.asarray(injectionDict['p_draw_m1m2z'])
        
        # Detected spins
        p_chi1_det = calculate_betaDistribution(chi1_det, a, b)
        p_chi2_det = calculate_betaDistribution(chi2_det, a, b)
        p_cost1_det = calculate_Gaussian_1D(cost1_det, mu_cost, sigma_cost, -1, 1.)
        p_cost2_det = calculate_Gaussian_1D(cost2_det, mu_cost, sigma_cost, -1, 1.)
        pdet_spins = p_chi1_det*p_chi2_det*p_cost1_det*p_cost2_det
        
        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, bq=Bq, mCut=8)
        pdet_z = p_astro_z(z_det, dV_dz=dVdz_det)
        
        # Construct full weighting factors
        p_det = pdet_spins*pdet_masses*pdet_z
        det_weights = p_det/p_draw
        
        if np.max(det_weights)==0:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Neff = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Neff<=4*nEvents:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # This is where we return the prior instead of the posterior if only_prior==True
        elif only_prior: 
            return logPrior, -np.inf, Neff, -np.inf
        
        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logL += log_detEff
        
        # --- Loop across BBH events ---
        Nsamps = np.zeros(len(sampleDict)) 
        for i,event in enumerate(sampleDict):

            # Unpack posterior samples for this event
            chi1_samples = np.asarray(sampleDict[event]['a1'])
            chi2_samples =  np.asarray(sampleDict[event]['a2'])
            cost1_samples = np.asarray(sampleDict[event]['cost1'])
            cost2_samples = np.asarray(sampleDict[event]['cost2'])
            m1_samples = np.asarray(sampleDict[event]['m1'])
            m2_samples = np.asarray(sampleDict[event]['m2'])
            z_samples = np.asarray(sampleDict[event]['z'])
            dVdz_samples = np.asarray(sampleDict[event]['dVc_dz'])
            bilby_prior_samples = np.asarray(sampleDict[event]['bilby_prior'])
            
            # Evaluate model at the locations of samples for this event
            p_chi1 = calculate_betaDistribution(chi1_samples, a, b)
            p_chi2 = calculate_betaDistribution(chi2_samples, a, b)
            p_cost1 = calculate_Gaussian_1D(cost1_samples, mu_cost, sigma_cost, -1, 1.)
            p_cost2 = calculate_Gaussian_1D(cost2_samples, mu_cost, sigma_cost, -1, 1.)
            
            # Pop dist for all four params combined is a product of each four individual dists
            p_spins = p_chi1*p_chi2*p_cost1*p_cost2
                        
            # Need to reweight by astrophysical priors on m1, m2, z ...
            p_astro_m1_m2 = p_astro_masses(m1_samples, m2_samples, bq=Bq, mCut=8)
            p_astro_redshift = p_astro_z(z_samples, dV_dz=dVdz_samples)
            
            # Sum over probabilities to get the marginalized likelihood for this event
            det_weights_event = p_spins*p_astro_m1_m2*p_astro_redshift/bilby_prior_samples
            nSamples = det_weights_event.size
            pEvidence = (1.0/nSamples)*np.sum(det_weights_event)
            
            # Calculate effective sample number
            Nsamps[i] = np.sum(det_weights_event)**2/np.sum(det_weights_event**2)
            
            # Cut in effective samples for each event
            # if Nsamp_event <= 10: 
            #     return -np.inf, -np.inf, -np.inf, -np.inf

            # Add to our running total
            logL += np.log(pEvidence)
            
        # Get minimum effective sample number over events
        minNsamp = np.min(Nsamps)
        
        # Combine likelihood and prior to get posteriors
        logP = logL + logPrior

        if logP!=logP:
            return -np.inf, -np.inf, -np.inf, -np.inf

        else:
            return logP, logL, Neff, minNsamp
        
        
def betaPlusGaussianAndIsotropic(c,sampleDict,injectionDict,priorDict,only_prior=False): 
    
    # Make sure hyper-sample is the right length
    assert len(c)==5, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    mu_chi = c[0]
    sigma_chi = c[1]
    sigma_cost = c[2] 
    MF_cost = c[3]
    Bq = c[4]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    if mu_chi < priorDict['mu_chi'][0] or mu_chi > priorDict['mu_chi'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma_chi < priorDict['sigma_chi'][0] or sigma_chi > priorDict['sigma_chi'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    if sigma_cost < priorDict['sigma_cost'][0] or sigma_cost > priorDict['sigma_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif MF_cost < priorDict['MF_cost'][0] or MF_cost > priorDict['MF_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    
    # If the sample falls inside our prior range, continue
    else:
        
        # Initialize log-likelihood
        logL = 0.
        
        # Translate mu_chi and sigma_chi to beta function parameters a and b 
        # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
        a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)
        
        # Impose cut on a and b: must be greater then or equal to 1 in order
        # for distribution to go to 0 at chi=0 and chi=1 (aka nonsingular)
        if a<=1. or b<=1.: 
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # Prior on Bq - gaussian centered at 0 with sigma=3
        logPrior = -(Bq**2)/18. 
        
        # --- Selection effects --- 
    
        # Unpack injections
        chi1_det = np.asarray(injectionDict['a1'])
        chi2_det = np.asarray(injectionDict['a2'])
        cost1_det = np.asarray(injectionDict['cost1'])
        cost2_det = np.asarray(injectionDict['cost2'])
        m1_det = np.asarray(injectionDict['m1'])
        m2_det = np.asarray(injectionDict['m2'])
        z_det = np.asarray(injectionDict['z'])
        dVdz_det = np.asarray(injectionDict['dVdz'])
        
        # Draw probability for component spins, masses, + redshift
        p_draw = np.asarray(injectionDict['p_draw_a1a2cost1cost2'])*np.asarray(injectionDict['p_draw_m1m2z'])
        
        # Detected spins
        p_chi1_det = calculate_betaDistribution(chi1_det, a, b)
        p_chi2_det = calculate_betaDistribution(chi2_det, a, b)
        p_cost1_det = 0.5*(1-MF_cost)*np.ones(len(cost1_det)) + MF_cost*calculate_Gaussian_1D(cost1_det, 1, sigma_cost, -1, 1.)
        p_cost2_det = 0.5*(1-MF_cost)*np.ones(len(cost2_det)) + MF_cost*calculate_Gaussian_1D(cost2_det, 1, sigma_cost, -1, 1.)
        pdet_spins = p_chi1_det*p_chi2_det*p_cost1_det*p_cost2_det
        
        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, bq=Bq, mCut=8)
        pdet_z = p_astro_z(z_det, dV_dz=dVdz_det)
        
        # Construct full weighting factors
        p_det = pdet_spins*pdet_masses*pdet_z
        det_weights = p_det/p_draw
        
        if np.max(det_weights)==0:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Neff = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Neff<=4*nEvents:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # This is where we return the prior instead of the posterior if only_prior==True
        elif only_prior: 
            return logPrior, -np.inf, Neff, -np.inf
        
        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logL += log_detEff
        
        # --- Loop across BBH events ---
        Nsamps = np.zeros(len(sampleDict)) 
        for i,event in enumerate(sampleDict):

            # Unpack posterior samples for this event
            chi1_samples = np.asarray(sampleDict[event]['a1'])
            chi2_samples =  np.asarray(sampleDict[event]['a2'])
            cost1_samples = np.asarray(sampleDict[event]['cost1'])
            cost2_samples = np.asarray(sampleDict[event]['cost2'])
            m1_samples = np.asarray(sampleDict[event]['m1'])
            m2_samples = np.asarray(sampleDict[event]['m2'])
            z_samples = np.asarray(sampleDict[event]['z'])
            dVdz_samples = np.asarray(sampleDict[event]['dVc_dz'])
            bilby_prior_samples = np.asarray(sampleDict[event]['bilby_prior'])
            
            # Evaluate model at the locations of samples for this event
            p_chi1 = calculate_betaDistribution(chi1_samples, a, b)
            p_chi2 = calculate_betaDistribution(chi2_samples, a, b)
            p_cost1 = 0.5*(1-MF_cost)*np.ones(len(cost1_samples)) + MF_cost*calculate_Gaussian_1D(cost1_samples, 1, sigma_cost, -1, 1.)
            p_cost2 = 0.5*(1-MF_cost)*np.ones(len(cost2_samples)) + MF_cost*calculate_Gaussian_1D(cost2_samples, 1, sigma_cost, -1, 1.)
            
            # Pop dist for all four params combined is a product of each four individual dists
            p_spins = p_chi1*p_chi2*p_cost1*p_cost2
                        
            # Need to reweight by astrophysical priors on m1, m2, z ...
            p_astro_m1_m2 = p_astro_masses(m1_samples, m2_samples, bq=Bq, mCut=8)
            p_astro_redshift = p_astro_z(z_samples, dV_dz=dVdz_samples)
            
            # Sum over probabilities to get the marginalized likelihood for this event
            det_weights_event = p_spins*p_astro_m1_m2*p_astro_redshift/bilby_prior_samples
            nSamples = det_weights_event.size
            pEvidence = (1.0/nSamples)*np.sum(det_weights_event)
            
            # Calculate effective sample number
            Nsamps[i] = np.sum(det_weights_event)**2/np.sum(det_weights_event**2)

            # Add to our running total
            logL += np.log(pEvidence)
            
        # Get minimum effective sample number over events
        minNsamp = np.min(Nsamps)
        
        # Combine likelihood and prior to get posteriors
        logP = logL + logPrior

        if logP!=logP:
            return -np.inf, -np.inf, -np.inf, -np.inf

        else:
            return logP, logL, Neff, minNsamp
        
        
def betaPlusDoubleGaussianJustChi(c,sampleDict,injectionDict,priorDict,true_cost_params,only_prior=False): 
    
    # Make sure hyper-sample is the right length
    assert len(c)==2, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    mu_chi = c[0]
    sigma_chi = c[1]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    if mu_chi < priorDict['mu_chi'][0] or mu_chi > priorDict['mu_chi'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma_chi < priorDict['sigma_chi'][0] or sigma_chi > priorDict['sigma_chi'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    
    # If the sample falls inside our prior range, continue
    else:
        
        # Initialize log-likelihood
        logL = 0.
        
        # Translate mu_chi and sigma_chi to beta function parameters a and b 
        # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
        a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)
        
        # Impose cut on a and b: must be greater then or equal to 1 in order
        # for distribution to go to 0 at chi=0 and chi=1 (aka nonsingular)
        if a<=1. or b<=1.: 
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        logPrior = 0
        
        # --- Selection effects --- 
    
        # Unpack injections
        chi1_det = np.asarray(injectionDict['a1'])
        chi2_det = np.asarray(injectionDict['a2'])
        cost1_det = np.asarray(injectionDict['cost1'])
        cost2_det = np.asarray(injectionDict['cost2'])
        m1_det = np.asarray(injectionDict['m1'])
        m2_det = np.asarray(injectionDict['m2'])
        z_det = np.asarray(injectionDict['z'])
        dVdz_det = np.asarray(injectionDict['dVdz'])
        
        # Draw probability for component spins, masses, + redshift
        p_draw = np.asarray(injectionDict['p_draw_a1a2cost1cost2'])*np.asarray(injectionDict['p_draw_m1m2z'])
        
        # Detected spins
        p_chi1_det = calculate_betaDistribution(chi1_det, a, b)
        p_chi2_det = calculate_betaDistribution(chi2_det, a, b)
        p_cost1_det = calculate_Double_Gaussian(cost1_det, *true_cost_params, -1, 1.)
        p_cost2_det = calculate_Double_Gaussian(cost2_det, *true_cost_params, -1, 1.)
        pdet_spins = p_chi1_det*p_chi2_det*p_cost1_det*p_cost2_det
        
        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, mCut=8)
        pdet_z = p_astro_z(z_det, dV_dz=dVdz_det)
        
        # Construct full weighting factors
        p_det = pdet_spins*pdet_masses*pdet_z
        det_weights = p_det/p_draw
        
        if np.max(det_weights)==0:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Neff = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Neff<=4*nEvents:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # This is where we return the prior instead of the posterior if only_prior==True
        elif only_prior: 
            return logPrior, -np.inf, Neff, -np.inf
        
        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logL += log_detEff
        
        # --- Loop across BBH events ---
        Nsamps = np.zeros(len(sampleDict)) 
        for i,event in enumerate(sampleDict):

            # Unpack posterior samples for this event
            chi1_samples = np.asarray(sampleDict[event]['a1'])
            chi2_samples =  np.asarray(sampleDict[event]['a2'])
            cost1_samples = np.asarray(sampleDict[event]['cost1'])
            cost2_samples = np.asarray(sampleDict[event]['cost2'])
            m1_samples = np.asarray(sampleDict[event]['m1'])
            m2_samples = np.asarray(sampleDict[event]['m2'])
            z_samples = np.asarray(sampleDict[event]['z'])
            dVdz_samples = np.asarray(sampleDict[event]['dVc_dz'])
            bilby_prior_samples = np.asarray(sampleDict[event]['bilby_prior'])
            
            # Evaluate model at the locations of samples for this event
            p_chi1 = calculate_betaDistribution(chi1_samples, a, b)
            p_chi2 = calculate_betaDistribution(chi2_samples, a, b)
            p_cost1 = calculate_Double_Gaussian(cost1_samples, *true_cost_params, -1, 1.)
            p_cost2 = calculate_Double_Gaussian(cost2_samples, *true_cost_params, -1, 1.)
            
            # Pop dist for all four params combined is a product of each four individual dists
            p_spins = p_chi1*p_chi2*p_cost1*p_cost2
                        
            # Need to reweight by astrophysical priors on m1, m2, z ...
            p_astro_m1_m2 = p_astro_masses(m1_samples, m2_samples, mCut=8)
            p_astro_redshift = p_astro_z(z_samples, dV_dz=dVdz_samples)
            
            # Sum over probabilities to get the marginalized likelihood for this event
            det_weights_event = p_spins*p_astro_m1_m2*p_astro_redshift/bilby_prior_samples
            nSamples = det_weights_event.size
            pEvidence = (1.0/nSamples)*np.sum(det_weights_event)
            
            # Calculate effective sample number
            Nsamp_event = np.sum(det_weights_event)**2/np.sum(det_weights_event**2) 
            Nsamps[i] = Nsamp_event

            # Add to our running total
            logL += np.log(pEvidence)
            
        # Get minimum effective sample number over events
        minNsamp = np.min(Nsamps)
        
        # Combine likelihood and prior to get posteriors
        logP = logL + logPrior

        if logP!=logP:
            return -np.inf, -np.inf, -np.inf, -np.inf

        else:
            return logP, logL, Neff, minNsamp
        
        
        
def betaPlusDoubleGaussianJustCosTheta(c,sampleDict,injectionDict,priorDict,true_chi_params,only_prior=False): 
    
    # Make sure hyper-sample is the right length
    assert len(c)==5, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    mu1_cost = c[0] 
    sigma1_cost = c[1]
    mu2_cost = c[2] 
    sigma2_cost = c[3]
    MF_cost = c[4]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    if mu1_cost < priorDict['mu_cost'][0] or mu1_cost > priorDict['mu_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma1_cost < priorDict['sigma_cost'][0] or sigma1_cost > priorDict['sigma_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    if mu2_cost < priorDict['mu_cost'][0] or mu2_cost > priorDict['mu_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma2_cost < priorDict['sigma_cost'][0] or sigma2_cost > priorDict['sigma_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif MF_cost < priorDict['MF_cost'][0] or MF_cost > priorDict['MF_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    
    # Additionally, to break the degeneracy between the two gaussians in the tilt distribution, 
    # impose that mu1_cost <= mu2_cost
    elif mu1_cost > mu2_cost:
        return -np.inf, -np.inf, -np.inf, -np.inf
    
    # If the sample falls inside our prior range, continue
    else:
        
        # Initialize log-likelihood
        logL = 0.
        
        # Translate mu_chi and sigma_chi to beta function parameters a and b 
        # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
        mu_chi, sigma_chi = true_chi_params
        a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)
        
        # Impose cut on a and b: must be greater then or equal to 1 in order
        # for distribution to go to 0 at chi=0 and chi=1 (aka nonsingular)
        if a<=1. or b<=1.: 
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        logPrior = 0
        
        # --- Selection effects --- 
    
        # Unpack injections
        chi1_det = np.asarray(injectionDict['a1'])
        chi2_det = np.asarray(injectionDict['a2'])
        cost1_det = np.asarray(injectionDict['cost1'])
        cost2_det = np.asarray(injectionDict['cost2'])
        m1_det = np.asarray(injectionDict['m1'])
        m2_det = np.asarray(injectionDict['m2'])
        z_det = np.asarray(injectionDict['z'])
        dVdz_det = np.asarray(injectionDict['dVdz'])
        
        # Draw probability for component spins, masses, + redshift
        p_draw = np.asarray(injectionDict['p_draw_a1a2cost1cost2'])*np.asarray(injectionDict['p_draw_m1m2z'])
        
        # Detected spins
        p_chi1_det = calculate_betaDistribution(chi1_det, a, b)
        p_chi2_det = calculate_betaDistribution(chi2_det, a, b)
        p_cost1_det = calculate_Double_Gaussian(cost1_det, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
        p_cost2_det = calculate_Double_Gaussian(cost2_det, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
        pdet_spins = p_chi1_det*p_chi2_det*p_cost1_det*p_cost2_det
        
        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, mCut=8)
        pdet_z = p_astro_z(z_det, dV_dz=dVdz_det)
        
        # Construct full weighting factors
        p_det = pdet_spins*pdet_masses*pdet_z
        det_weights = p_det/p_draw
        
        if np.max(det_weights)==0:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Neff = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Neff<=4*nEvents:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # This is where we return the prior instead of the posterior if only_prior==True
        elif only_prior: 
            return logPrior, -np.inf, Neff, -np.inf
        
        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logL += log_detEff
        
        # --- Loop across BBH events ---
        Nsamps = np.zeros(len(sampleDict)) 
        for i,event in enumerate(sampleDict):

            # Unpack posterior samples for this event
            chi1_samples = np.asarray(sampleDict[event]['a1'])
            chi2_samples =  np.asarray(sampleDict[event]['a2'])
            cost1_samples = np.asarray(sampleDict[event]['cost1'])
            cost2_samples = np.asarray(sampleDict[event]['cost2'])
            m1_samples = np.asarray(sampleDict[event]['m1'])
            m2_samples = np.asarray(sampleDict[event]['m2'])
            z_samples = np.asarray(sampleDict[event]['z'])
            dVdz_samples = np.asarray(sampleDict[event]['dVc_dz'])
            bilby_prior_samples = np.asarray(sampleDict[event]['bilby_prior'])
            
            # Evaluate model at the locations of samples for this event
            p_chi1 = calculate_betaDistribution(chi1_samples, a, b)
            p_chi2 = calculate_betaDistribution(chi2_samples, a, b)
            p_cost1 = calculate_Double_Gaussian(cost1_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
            p_cost2 = calculate_Double_Gaussian(cost2_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
            
            # Pop dist for all four params combined is a product of each four individual dists
            p_spins = p_chi1*p_chi2*p_cost1*p_cost2
                        
            # Need to reweight by astrophysical priors on m1, m2, z ...
            p_astro_m1_m2 = p_astro_masses(m1_samples, m2_samples, mCut=8)
            p_astro_redshift = p_astro_z(z_samples, dV_dz=dVdz_samples)
            
            # Sum over probabilities to get the marginalized likelihood for this event
            det_weights_event = p_spins*p_astro_m1_m2*p_astro_redshift/bilby_prior_samples
            nSamples = det_weights_event.size
            pEvidence = (1.0/nSamples)*np.sum(det_weights_event)
            
            # Calculate effective sample number
            Nsamp_event = np.sum(det_weights_event)**2/np.sum(det_weights_event**2) 
            Nsamps[i] = Nsamp_event

            # Add to our running total
            logL += np.log(pEvidence)
            
        # Get minimum effective sample number over events
        minNsamp = np.min(Nsamps)
        
        # Combine likelihood and prior to get posteriors
        logP = logL + logPrior

        if logP!=logP:
            return -np.inf, -np.inf, -np.inf, -np.inf

        else:
            return logP, logL, Neff, minNsamp
        
        
def betaPlusDoubleGaussian2(c,sampleDict,injectionDict,priorDict,only_prior=False): 
    
    # sort by mixing fraction
    
    # Make sure hyper-sample is the right length
    assert len(c)==8, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    mu_chi = c[0]
    sigma_chi = c[1]
    mu1_cost = c[2] 
    sigma1_cost = c[3]
    mu2_cost = c[4] 
    sigma2_cost = c[5]
    MF_cost = c[6]
    Bq = c[7]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    if mu_chi < priorDict['mu_chi'][0] or mu_chi > priorDict['mu_chi'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma_chi < priorDict['sigma_chi'][0] or sigma_chi > priorDict['sigma_chi'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    if mu1_cost < priorDict['mu_cost'][0] or mu1_cost > priorDict['mu_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma1_cost < priorDict['sigma_cost'][0] or sigma1_cost > priorDict['sigma_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    if mu2_cost < priorDict['mu_cost'][0] or mu2_cost > priorDict['mu_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma2_cost < priorDict['sigma_cost'][0] or sigma2_cost > priorDict['sigma_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif MF_cost < priorDict['MF_cost'][0] or MF_cost > 0.5*priorDict['MF_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    
    # If the sample falls inside our prior range, continue
    else:
        
        # Initialize log-likelihood
        logL = 0.
        
        # Translate mu_chi and sigma_chi to beta function parameters a and b 
        # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
        a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)
        
        # Impose cut on a and b: must be greater then or equal to 1 in order
        # for distribution to go to 0 at chi=0 and chi=1 (aka nonsingular)
        if a<=1. or b<=1.: 
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # Prior on Bq - gaussian centered at 0 with sigma=3
        logPrior = -(Bq**2)/18. 
        
        # --- Selection effects --- 
    
        # Unpack injections
        chi1_det = np.asarray(injectionDict['a1'])
        chi2_det = np.asarray(injectionDict['a2'])
        cost1_det = np.asarray(injectionDict['cost1'])
        cost2_det = np.asarray(injectionDict['cost2'])
        m1_det = np.asarray(injectionDict['m1'])
        m2_det = np.asarray(injectionDict['m2'])
        z_det = np.asarray(injectionDict['z'])
        dVdz_det = np.asarray(injectionDict['dVdz'])
        
        # Draw probability for component spins, masses, + redshift
        p_draw = np.asarray(injectionDict['p_draw_a1a2cost1cost2'])*np.asarray(injectionDict['p_draw_m1m2z'])
        
        # Detected spins
        p_chi1_det = calculate_betaDistribution(chi1_det, a, b)
        p_chi2_det = calculate_betaDistribution(chi2_det, a, b)
        p_cost1_det = calculate_Double_Gaussian(cost1_det, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
        p_cost2_det = calculate_Double_Gaussian(cost2_det, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
        pdet_spins = p_chi1_det*p_chi2_det*p_cost1_det*p_cost2_det
        
        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, bq=Bq, mCut=8)
        pdet_z = p_astro_z(z_det, dV_dz=dVdz_det)
        
        # Construct full weighting factors
        p_det = pdet_spins*pdet_masses*pdet_z
        det_weights = p_det/p_draw
        
        if np.max(det_weights)==0:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Neff = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Neff<=4*nEvents:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # This is where we return the prior instead of the posterior if only_prior==True
        elif only_prior: 
            return logPrior, -np.inf, Neff, -np.inf
        
        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logL += log_detEff
        
        # --- Loop across BBH events ---
        Nsamps = np.zeros(len(sampleDict)) 
        for i,event in enumerate(sampleDict):

            # Unpack posterior samples for this event
            chi1_samples = np.asarray(sampleDict[event]['a1'])
            chi2_samples =  np.asarray(sampleDict[event]['a2'])
            cost1_samples = np.asarray(sampleDict[event]['cost1'])
            cost2_samples = np.asarray(sampleDict[event]['cost2'])
            m1_samples = np.asarray(sampleDict[event]['m1'])
            m2_samples = np.asarray(sampleDict[event]['m2'])
            z_samples = np.asarray(sampleDict[event]['z'])
            dVdz_samples = np.asarray(sampleDict[event]['dVc_dz'])
            bilby_prior_samples = np.asarray(sampleDict[event]['bilby_prior'])
            
            # Evaluate model at the locations of samples for this event
            p_chi1 = calculate_betaDistribution(chi1_samples, a, b)
            p_chi2 = calculate_betaDistribution(chi2_samples, a, b)
            p_cost1 = calculate_Double_Gaussian(cost1_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
            p_cost2 = calculate_Double_Gaussian(cost2_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
            
            # Pop dist for all four params combined is a product of each four individual dists
            p_spins = p_chi1*p_chi2*p_cost1*p_cost2
                        
            # Need to reweight by astrophysical priors on m1, m2, z ...
            p_astro_m1_m2 = p_astro_masses(m1_samples, m2_samples, bq=Bq, mCut=8)
            p_astro_redshift = p_astro_z(z_samples, dV_dz=dVdz_samples)
            
            # Sum over probabilities to get the marginalized likelihood for this event
            det_weights_event = p_spins*p_astro_m1_m2*p_astro_redshift/bilby_prior_samples
            nSamples = det_weights_event.size
            pEvidence = (1.0/nSamples)*np.sum(det_weights_event)
            
            # Calculate effective sample number
            Nsamps[i] = np.sum(det_weights_event)**2/np.sum(det_weights_event**2)

            # Add to our running total
            logL += np.log(pEvidence)
            
        # Get minimum effective sample number over events
        minNsamp = np.min(Nsamps)
        
        # Combine likelihood and prior to get posteriors
        logP = logL + logPrior

        if logP!=logP:
            return -np.inf, -np.inf, -np.inf, -np.inf

        else:
            return logP, logL, Neff, minNsamp
        
        
def betaPlusDoubleGaussian2_noSelectionSpins(c,sampleDict,injectionDict,priorDict,only_prior=False): 
    
    # sort by mixing fraction
    
    # Make sure hyper-sample is the right length
    assert len(c)==8, 'Input sample has wrong length'
    
    # Number of events 
    nEvents = len(sampleDict)
    
    # Unpack hyper-parameters
    mu_chi = c[0]
    sigma_chi = c[1]
    mu1_cost = c[2] 
    sigma1_cost = c[3]
    mu2_cost = c[4] 
    sigma2_cost = c[5]
    MF_cost = c[6]
    Bq = c[7]
    
    # Reject samples outside of our prior bounds for those with uniform priors
    if mu_chi < priorDict['mu_chi'][0] or mu_chi > priorDict['mu_chi'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma_chi < priorDict['sigma_chi'][0] or sigma_chi > priorDict['sigma_chi'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    if mu1_cost < priorDict['mu_cost'][0] or mu1_cost > priorDict['mu_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma1_cost < priorDict['sigma_cost'][0] or sigma1_cost > priorDict['sigma_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    if mu2_cost < priorDict['mu_cost'][0] or mu2_cost > priorDict['mu_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif sigma2_cost < priorDict['sigma_cost'][0] or sigma2_cost > priorDict['sigma_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    elif MF_cost < priorDict['MF_cost'][0] or MF_cost > 0.5*priorDict['MF_cost'][1]:
        return -np.inf, -np.inf, -np.inf, -np.inf
    
    # If the sample falls inside our prior range, continue
    else:
        
        # Initialize log-likelihood
        logL = 0.
        
        # Translate mu_chi and sigma_chi to beta function parameters a and b 
        # See: https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance
        a, b = mu_sigma2_to_a_b(mu_chi, sigma_chi**2.)
        
        # Impose cut on a and b: must be greater then or equal to 1 in order
        # for distribution to go to 0 at chi=0 and chi=1 (aka nonsingular)
        if a<=1. or b<=1.: 
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # Prior on Bq - gaussian centered at 0 with sigma=3
        logPrior = -(Bq**2)/18. 
        
        # --- Selection effects --- 
    
        # Unpack injections for mass + redshift
        m1_det = np.asarray(injectionDict['m1'])
        m2_det = np.asarray(injectionDict['m2'])
        z_det = np.asarray(injectionDict['z'])
        dVdz_det = np.asarray(injectionDict['dVdz'])
        
        # Draw probability for component masses, + redshift
        p_draw = np.asarray(injectionDict['p_draw_m1m2z'])
        
        # Detected masses and redshifts
        pdet_masses = p_astro_masses(m1_det, m2_det, bq=Bq, mCut=8)
        pdet_z = p_astro_z(z_det, dV_dz=dVdz_det)
        
        # Construct full weighting factors
        p_det = pdet_masses*pdet_z
        det_weights = p_det/p_draw
        
        if np.max(det_weights)==0:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # Check for sufficient sampling size
        # Specifically require 4*Ndet effective detections, according to https://arxiv.org/abs/1904.10879
        Neff = np.sum(det_weights)**2/np.sum(det_weights**2)
        if Neff<=4*nEvents:
            return -np.inf, -np.inf, -np.inf, -np.inf
        
        # This is where we return the prior instead of the posterior if only_prior==True
        elif only_prior: 
            return logPrior, -np.inf, Neff, -np.inf
        
        # Calculate detection efficiency and add to log posterior
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logL += log_detEff
        
        # --- Loop across BBH events ---
        Nsamps = np.zeros(len(sampleDict)) 
        for i,event in enumerate(sampleDict):

            # Unpack posterior samples for this event
            chi1_samples = np.asarray(sampleDict[event]['a1'])
            chi2_samples =  np.asarray(sampleDict[event]['a2'])
            cost1_samples = np.asarray(sampleDict[event]['cost1'])
            cost2_samples = np.asarray(sampleDict[event]['cost2'])
            m1_samples = np.asarray(sampleDict[event]['m1'])
            m2_samples = np.asarray(sampleDict[event]['m2'])
            z_samples = np.asarray(sampleDict[event]['z'])
            dVdz_samples = np.asarray(sampleDict[event]['dVc_dz'])
            bilby_prior_samples = np.asarray(sampleDict[event]['bilby_prior'])
            
            # Evaluate model at the locations of samples for this event
            p_chi1 = calculate_betaDistribution(chi1_samples, a, b)
            p_chi2 = calculate_betaDistribution(chi2_samples, a, b)
            p_cost1 = calculate_Double_Gaussian(cost1_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
            p_cost2 = calculate_Double_Gaussian(cost2_samples, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, MF_cost, -1, 1.)
            
            # Pop dist for all four params combined is a product of each four individual dists
            p_spins = p_chi1*p_chi2*p_cost1*p_cost2
                        
            # Need to reweight by astrophysical priors on m1, m2, z ...
            p_astro_m1_m2 = p_astro_masses(m1_samples, m2_samples, bq=Bq, mCut=8)
            p_astro_redshift = p_astro_z(z_samples, dV_dz=dVdz_samples)
            
            # Sum over probabilities to get the marginalized likelihood for this event
            det_weights_event = p_spins*p_astro_m1_m2*p_astro_redshift/bilby_prior_samples
            nSamples = det_weights_event.size
            pEvidence = (1.0/nSamples)*np.sum(det_weights_event)
            
            # Calculate effective sample number
            Nsamps[i] = np.sum(det_weights_event)**2/np.sum(det_weights_event**2)

            # Add to our running total
            logL += np.log(pEvidence)
            
        # Get minimum effective sample number over events
        minNsamp = np.min(Nsamps)
        
        # Combine likelihood and prior to get posteriors
        logP = logL + logPrior

        if logP!=logP:
            return -np.inf, -np.inf, -np.inf, -np.inf

        else:
            return logP, logL, Neff, minNsamp