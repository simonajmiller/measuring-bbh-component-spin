import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from jax.scipy.special import erf
from jax import vmap,lax
import numpy as np
from utilities import *

def baseline(sampleDict,injectionDict,fixedMassDistribution=True):

    """
    Implementation of a Gaussian effective spin distribution for inference within `numpyro`

    Parameters
    ----------
    sampleDict : dict
        Precomputed dictionary containing posterior samples for each event in our catalog
    injectionDict : dict
        Precomputed dictionary containing successfully recovered injections
    """
    
    # Sample our hyperparameters
    # alpha: Power-law index on primary mass distribution
    # mu_m1: Location of gaussian peak in primary mass distribution
    # sig_m1: Width of gaussian peak
    # f_peak: Fraction of events comprising gaussian peak
    # mMax: Location at which BBH mass distribution tapers off
    # mMin: Lower boundary at which BBH mass distribution tapers off
    # dmMax: Taper width above maximum mass
    # dmMin: Taper width below minimum mass
    # bq: Power-law index on the conditional secondary mass distribution p(m2|m1)
    # mu: Mean of the chi-effective distribution
    # logsig_chi: Log10 of the chi-effective distribution's standard deviation

    logR30 = numpyro.sample("logR30",dist.Uniform(-4,2))
    R30 = numpyro.deterministic("R30",10.**logR30)
    bq = numpyro.sample("bq",dist.Normal(0,3))
    kappa = numpyro.sample("kappa",dist.Normal(0,5))

    if fixedMassDistribution==True:

        alpha = numpyro.deterministic("alpha",-3.51)
        mu_m1 = numpyro.deterministic("mu_m1",33.61)
        mMin = numpyro.deterministic("mMin",10.)
        sig_m1 = numpyro.deterministic("sig_m1",4.72)
        log_f_peak = numpyro.deterministic("log_f_peak",jnp.log10(0.003))
        mMax = numpyro.deterministic("mMax",88.21)
        log_dmMin = numpyro.deterministic("log_dmMin",0.)
        log_dmMax = numpyro.deterministic("log_dmMax",0.5)

    else:

        alpha = numpyro.sample("alpha",dist.Normal(-2,3))
        mu_m1 = numpyro.sample("mu_m1",dist.Uniform(20,50))
        mMin = numpyro.sample("mMin",dist.Uniform(5,15))

        logit_sig_m1 = numpyro.sample("logit_sig_m1",dist.Normal(0,logit_std))
        logit_log_f_peak = numpyro.sample("logit_log_f_peak",dist.Normal(0,logit_std))
        logit_mMax = numpyro.sample("logit_mMax",dist.Normal(0,logit_std))
        logit_log_dmMin = numpyro.sample("logit_log_dmMin",dist.Normal(0,logit_std))
        logit_log_dmMax = numpyro.sample("logit_log_dmMax",dist.Normal(0,logit_std))

        sig_m1,jac_sig_m1 = get_value_from_logit(logit_sig_m1,2.,15.)
        log_f_peak,jac_log_f_peak = get_value_from_logit(logit_log_f_peak,-4,0.)
        mMax,jac_mMax = get_value_from_logit(logit_mMax,50.,100.)
        log_dmMin,jac_log_dmMin = get_value_from_logit(logit_log_dmMin,-1,1)
        log_dmMax,jac_log_dmMax = get_value_from_logit(logit_log_dmMax,0.,1.5)

        numpyro.deterministic("sig_m1",sig_m1)
        numpyro.deterministic("log_f_peak",log_f_peak)
        numpyro.deterministic("mMax",mMax)
        numpyro.deterministic("log_dmMin",log_dmMin)
        numpyro.deterministic("log_dmMax",log_dmMax)

        numpyro.factor("p_sig_m1",logit_sig_m1**2/(2.*logit_std**2)-jnp.log(jac_sig_m1))
        numpyro.factor("p_log_f_peak",logit_log_f_peak**2/(2.*logit_std**2)-jnp.log(jac_log_f_peak))
        numpyro.factor("p_mMax",logit_mMax**2/(2.*logit_std**2)-jnp.log(jac_mMax))
        numpyro.factor("p_log_dmMin",logit_log_dmMin**2/(2.*logit_std**2)-jnp.log(jac_log_dmMin))
        numpyro.factor("p_log_dmMax",logit_log_dmMax**2/(2.*logit_std**2)-jnp.log(jac_log_dmMax))

    logit_mu_chi = numpyro.sample("logit_mu_chi",dist.Normal(0,logit_std))
    logit_sig_chi = numpyro.sample("logit_sig_chi",dist.Normal(0,logit_std))
    logit_sig_cost_high = numpyro.sample("logit_sig_cost_high",dist.Normal(0,logit_std))
    logit_sig_cost_low = numpyro.sample("logit_sig_cost_low",dist.Normal(0,logit_std))
    logit_mu_cost_low = numpyro.sample("logit_mu_cost_low",dist.Normal(0,logit_std))
    logit_mu_cost_high = numpyro.sample("logit_mu_cost_high",dist.Normal(0,logit_std))
    logit_f_big = numpyro.sample("logit_f_big",dist.Normal(0,logit_std))

    mu_chi,jac_mu_chi = get_value_from_logit(logit_mu_chi,0.,1.)
    sig_chi,jac_sig_chi = get_value_from_logit(logit_sig_chi,0.1,1.)
    sig_cost_high,jac_sig_cost_high = get_value_from_logit(logit_sig_cost_high,0.1,1.5)
    sig_cost_low,jac_sig_cost_low = get_value_from_logit(logit_sig_cost_low,0.1,1.5)
    mu_cost_low,jac_mu_cost_low = get_value_from_logit(logit_mu_cost_low,-1,1.)
    mu_cost_high,jac_mu_cost_high = get_value_from_logit(logit_mu_cost_high,-1,1.)
    f_big,jac_f_big = get_value_from_logit(logit_f_big,0.5,1.)

    numpyro.deterministic("mu_chi",mu_chi)
    numpyro.deterministic("sig_chi",sig_chi)
    numpyro.deterministic("sig_cost_high",sig_cost_high)
    numpyro.deterministic("sig_cost_low",sig_cost_low)
    numpyro.deterministic("mu_cost_low",mu_cost_low)
    numpyro.deterministic("mu_cost_high",mu_cost_high)
    numpyro.deterministic("f_big",f_big)

    numpyro.factor("p_mu_chi",logit_mu_chi**2/(2.*logit_std**2)-jnp.log(jac_mu_chi))
    numpyro.factor("p_sig_chi",logit_sig_chi**2/(2.*logit_std**2)-jnp.log(jac_sig_chi))
    numpyro.factor("p_sig_cost_high",logit_sig_cost_high**2/(2.*logit_std**2)-jnp.log(jac_sig_cost_high))
    numpyro.factor("p_sig_cost_low",logit_sig_cost_low**2/(2.*logit_std**2)-jnp.log(jac_sig_cost_low))
    numpyro.factor("p_mu_cost_low",logit_mu_cost_low**2/(2.*logit_std**2)-jnp.log(jac_mu_cost_low))
    numpyro.factor("p_mu_cost_high",logit_mu_cost_high**2/(2.*logit_std**2)-jnp.log(jac_mu_cost_high))
    numpyro.factor("p_f_big",logit_f_big**2/(2.*logit_std**2)-jnp.log(jac_f_big))

    # Normalization
    p_m1_norm = massModel(30.,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)
    p_z_norm = (1.+0.2)**kappa

    # Read out found injections
    # Note that `pop_reweight` is the inverse of the draw weights for each event
    a1_det = injectionDict['a1']
    a2_det = injectionDict['a2']
    cost1_det = injectionDict['cost1']
    cost2_det = injectionDict['cost2']
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    p_draw = injectionDict['p_draw_m1m2z']*injectionDict['p_draw_a1a2cost1cost2']

    # Compute proposed population weights
    p_m1_det = massModel(m1_det,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/p_m1_norm
    p_m2_det = (1.+bq)*m2_det**bq/(m1_det**(1.+bq)-tmp_min**(1.+bq))
    p_m2_det = jnp.where(m2_det>tmp_min,p_m2_det,0)
    p_a1_det = truncatedNormal(a1_det,mu_chi,sig_chi,0,1)
    p_a2_det = truncatedNormal(a2_det,mu_chi,sig_chi,0,1)
    p_cost1_det = (1.-f_big)*truncatedNormal(cost1_det,mu_cost_low,sig_cost_low,-1,1) + f_big*truncatedNormal(cost1_det,mu_cost_high,sig_cost_high,-1,1)
    p_cost2_det = (1.-f_big)*truncatedNormal(cost2_det,mu_cost_low,sig_cost_low,-1,1) + f_big*truncatedNormal(cost2_det,mu_cost_high,sig_cost_high,-1,1)
    p_z_det = dVdz_det*(1.+z_det)**(kappa-1.)/p_z_norm 
    R_pop_det = R30*p_m1_det*p_m2_det*p_z_det*p_a1_det*p_a2_det*p_cost1_det*p_cost2_det

    # Form ratio of proposed weights over draw weights
    inj_weights = R_pop_det/(p_draw/1.)
    
    # As a fit diagnostic, compute effective number of injections
    nEff_inj = jnp.sum(inj_weights)**2/jnp.sum(inj_weights**2)
    nObs = 1.0*len(sampleDict)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/nObs)

    # Compute net detection efficiency and add to log-likelihood
    Nexp = jnp.sum(inj_weights)/len(inj_weights)
    numpyro.factor("rate",-Nexp)
    
    # This function defines the per-event log-likelihood
    # m1_sample: Primary mass posterior samples
    # m2_sample: Secondary mass posterior samples
    # z_sample: Redshift posterior samples
    # dVdz_sample: Differential comoving volume at each sample location
    # Xeff_sample: Effective spin posterior samples
    # priors: PE priors on each sample
    def logp(m1_sample,m2_sample,z_sample,dVdz_sample,a1_sample,a2_sample,cost1_sample,cost2_sample,priors):

        # Compute proposed population weights
        p_m1 = massModel(m1_sample,alpha,mu_m1,sig_m1,10.**log_f_peak,mMax,mMin,10.**log_dmMax,10.**log_dmMin)/p_m1_norm
        p_m2 = (1.+bq)*m2_sample**bq/(m1_sample**(1.+bq)-tmp_min**(1.+bq))
        p_m2 = jnp.where(m2_sample>tmp_min,p_m2,0)
        p_a1 = truncatedNormal(a1_sample,mu_chi,sig_chi,0,1)
        p_a2 = truncatedNormal(a2_sample,mu_chi,sig_chi,0,1)
        p_cost1 = (1.-f_big)*truncatedNormal(cost1_sample,mu_cost_low,sig_cost_low,-1,1) + f_big*truncatedNormal(cost1_sample,mu_cost_high,sig_cost_high,-1,1)
        p_cost2 = (1.-f_big)*truncatedNormal(cost2_sample,mu_cost_low,sig_cost_low,-1,1) + f_big*truncatedNormal(cost2_sample,mu_cost_high,sig_cost_high,-1,1)
        p_z = dVdz_sample*(1.+z_sample)**(kappa-1.)/p_z_norm
        R_pop = R30*p_m1*p_m2*p_z*p_a1*p_a2*p_cost1*p_cost2

        mc_weights = R_pop/priors
        
        # Compute effective number of samples and return log-likelihood
        n_eff = jnp.sum(mc_weights)**2/jnp.sum(mc_weights**2)     
        return jnp.log(jnp.mean(mc_weights)),n_eff
    
    # Map the log-likelihood function over each event in our catalog
    log_ps,n_effs = vmap(logp)(
                        jnp.array([sampleDict[k]['m1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['m2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['dVc_dz'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['bilby_prior'] for k in sampleDict]))
        
    # As a diagnostic, save minimum number of effective samples across all events
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp",jnp.sum(log_ps))

