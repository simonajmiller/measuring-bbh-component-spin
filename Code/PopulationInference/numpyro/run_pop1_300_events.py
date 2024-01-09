import numpyro
import json
nChains = 1
numpyro.set_host_device_count(nChains)
from numpyro.infer import NUTS,MCMC
from jax import random
from jax import config
config.update("jax_enable_x64", True)
import arviz as az
import numpy as np
np.random.seed(117)
from baseline_population import baseline
import sys

# File path root for where to store data 
froot = "/home/simona.miller/measuring-bbh-component-spin/Data/"

# Run over several chains to check convergence

# Get dictionaries holding injections and posterior samples
with open(froot+"PopulationInferenceInput/injectionDict.json","r") as jf:
    injectionDict = json.load(jf)

for key in injectionDict:
    injectionDict[key] = np.array(injectionDict[key])

injectionDict['nTrials'] = len(injectionDict['m1'])

with open(froot+"PopulationInferenceInput/sampleDict_population1_highSpinPrecessing.json","r") as jf:
    sampleDict = json.load(jf)

with open(froot+"PopulationInferenceOutput/betaPlusDoubleGaussian2/110623_betaPlusDoubleGaussian2_pop1_300events.json","r") as jf:
    simonaOutput = json.load(jf)

events_to_be_used = simonaOutput['events_used']
eventLabels = list(sampleDict.keys())
events_to_be_popped = list(set(eventLabels) - set(events_to_be_used))

for event in events_to_be_popped:
    sampleDict.pop(event)

sample_limit = 4000
for event in sampleDict:

    sampleDict[event]['a1'] = sampleDict[event]['a1']['bilby_posterior']
    sampleDict[event]['a2'] = sampleDict[event]['a2']['bilby_posterior']
    sampleDict[event]['cost1'] = sampleDict[event]['cost1']['bilby_posterior']
    sampleDict[event]['cost2'] = sampleDict[event]['cost2']['bilby_posterior']

    p_draw = np.ones(len(sampleDict[event]['m1']))
    p_draw[np.array(sampleDict[event]['m1'])>88.] = 0.
    p_draw[np.array(sampleDict[event]['m2'])<8.] = 0.
    p_draw /= np.sum(p_draw)

    # Randomly downselect to the desired number of samples       
    inds_to_keep = np.random.choice(np.arange(len(sampleDict[event]['m1'])),size=sample_limit,replace=True,p=p_draw)
    for key in sampleDict[event].keys():
        if key!='injected_params':
            sampleDict[event][key] = np.array(sampleDict[event][key])[inds_to_keep]

# Set up NUTS sampler over our likelihood
kernel = NUTS(baseline,target_accept_prob=0.9,dense_mass=[("logit_mu_cost_low","logit_mu_cost_high","logit_f_big","logit_sig_cost_high","logit_sig_cost_low")])
mcmc = MCMC(kernel,num_warmup=1500,num_samples=1500,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(119)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict,True)
mcmc.print_summary()

# Save out data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"./output/pop1_300_events.cdf")

