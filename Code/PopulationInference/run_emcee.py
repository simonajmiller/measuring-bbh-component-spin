import numpy as np
import argparse
import glob
import emcee as mc
import json
import sys
from posterior_helper_functions import draw_initial_walkers_uniform
from posteriors import *
from postprocessing import processEmceeChain 

"""
Definitions and loading data
"""

# Parse commandline arguments
p = argparse.ArgumentParser()
p.add_argument('--model')
p.add_argument('--date')
p.add_argument('--pop')
p.add_argument('--nevents', type=int)
p.add_argument('--nsteps', type=int, default=20000)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--posterior-key', default='bilby_posterior')
p.add_argument('--only-prior', action='store_true')
args = p.parse_args()

# Pass population and number of events via commandline
pop = args.pop
nevents = args.nevents

# Bilby samples or mock gaussian samples
posterior_key = args.posterior_key

# Option to run on only the prior
only_prior = args.only_prior

# set seed for reproducibility
seed = args.seed if args.seed!=0 else 2345 # arbitrary
np.random.seed(seed)

# Model
model = args.model 
allowed_models = ["betaPlusDoubleGaussian", "betaPlusGaussian", "betaPlusGaussianAndIsotropic",
                  "betaPlusDoubleGaussianJustChi", "betaPlusDoubleGaussianJustCosTheta", 
                  "betaPlusDoubleGaussian2", "betaPlusDoubleGaussian2_noSelectionSpins"]
assert model in allowed_models, f"Given model ('{model}') not in allowed models ({allowed_models})"

# How to save
if only_prior: 
    model_savename = f"{args.date}_{model}_prior_{nevents}events"
else: 
    model_savename = f"{args.date}_{model}_pop{pop}_{nevents}events"

# Add flags to savename if not default
if posterior_key != 'bilby_posterior': 
    model_savename += f"_{posterior_key}"
if args.seed != 0:
    model_savename += f"_seed{seed}" 

print(f'Running {model_savename} ...')

# File path root for where to store data 
froot = "/home/simona.miller/measuring-bbh-component-spin/Data/"

# Names of the different populations
pop_names = {
    '1':'population1_highSpinPrecessing', 
    '2':'population2_mediumSpin',
    '3':'population3_lowSpinAligned'
}

# Load sampleDict
with open(froot+f"PopulationInferenceInput/sampleDict_{pop_names[pop]}.json", 'r') as f: 
    sampleDict_full = json.load(f)
    
## Condition sampleDict: 
keys_to_ignore = ['2418', '3339']
# 1. Choose subset of sampleDict if necessary
if int(nevents)<len(sampleDict_full.keys()): 
    keys = [key for key in sampleDict_full.keys() if key not in keys_to_ignore]
    events = np.random.choice(keys, size=int(nevents), replace=False)
    sampleDict_temp1 = {event:sampleDict_full[event] for event in events}
else: 
    sampleDict_temp1 = sampleDict_full
    
    if int(nevents)>len(sampleDict_temp1.keys()): 
        print('Too many events requested. Changing nevents to max # possible.')  
        new_nevents = len(sampleDict_temp1.keys())
        model_savename = model_savename.replace(str(nevents), str(new_nevents))
        print('Savename updated: ', model_savename)
        nevents = new_nevents

print(f'After nevents cut, {len(sampleDict_temp1)} remaining.')
    
# 2. Choose the correct set of posterior samples from the sampleDict
sampleDict = {}
for event in sampleDict_temp1.keys():
    # for masses and redshifts always use bilby posteriors
    d1 = {p:sampleDict_temp1[event][p] for p in ['m1', 'm2', 'z', 'dVc_dz', 'bilby_prior']}
    # for spin magnitude and tilts, option to use bilby or gaussian posteriors
    d2 = {p:sampleDict_temp1[event][p][posterior_key] for p in ['a1', 'a2', 'cost1', 'cost2']}
    # combine into final sampleDict
    sampleDict[event] = {**d1, **d2}
    
# Load injectionDict
with open(froot+"PopulationInferenceInput/injectionDict.json", 'r') as f: 
    injectionDict = json.load(f)
    
# Will save emcee chains temporarily in the .tmp folder in this directory
output_folder_tmp = froot+"PopulationInferenceOutput/.tmp/"
output_tmp = output_folder_tmp+model_savename


"""
Initializing emcee walkers or picking up where an old chain left off.
"""

# Define emcee parameters
nWalkers = 20       # number of walkers 
nSteps = args.nsteps    # number of steps for chain

# dimension of parameter space (number hyper params):
if model in ['betaPlusDoubleGaussian', 'betaPlusDoubleGaussian2', 'betaPlusDoubleGaussian2_noSelectionSpins']:
    dim = 8             
elif model in ['betaPlusGaussian', 'betaPlusDoubleGaussianJustCosTheta', "betaPlusGaussianAndIsotropic"]: 
    dim = 5
elif model=='betaPlusDoubleGaussianJustChi': 
    dim = 2

# Set prior bounds 
priorDict = {
    'mu_chi':(0., 1.),
    'sigma_chi':(0.07, 0.5),
    'mu_cost':(-1., 1.),
    'sigma_cost':(0.16, 0.8),
    'MF_cost':(0., 1.)
}

# True mu_chi, sigma_chi, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, f, Bq:
hyperparams = { 
    '1':[0.55, 0.26, 0.19, 0.18, 0.42, 0.75, 0.55, 0.96],  
    '2':[0.32, 0.16, 0.33, 0.64, 0.59, 0.40, 0.36, 0.96],
    '3':[0.19, 0.12, -0.98, 0.44, 0.98, 0.31, 0.26, 0.96]
}

# Search for existing chains
old_chains = np.sort(glob.glob("{0}_r??.npy".format(output_tmp)))

# If no chain already exists, begin a new one
if len(old_chains)==0:
    
    print('\nNo old chains found, generating initial walkers ... ')

    run_version = 0

    # Initialize walkers 
    
    if 'betaPlusDoubleGaussian' in model:
                
        # Draw initial walkers close to true value
        if model in ['betaPlusDoubleGaussian', 'betaPlusDoubleGaussianJustChi', 
                     'betaPlusDoubleGaussian2', 'betaPlusDoubleGaussian2_noSelectionSpins']:
            initial_walkers = np.transpose(
                [draw_initial_walkers_uniform(nWalkers, (hyperparams[pop][i]-0.02, hyperparams[pop][i]+0.02)) for i in range(dim)]
            )
        elif model=='betaPlusDoubleGaussianJustCosTheta':
            hp = hyperparams[pop][2:-1]
            initial_walkers = np.transpose(
                [draw_initial_walkers_uniform(nWalkers, (h-0.02, h+0.02)) for h in hp]
            )
        
        if '2' in model: 
            # replace mixing fraction walkers
            initial_walkers[:,-2] = draw_initial_walkers_uniform(nWalkers, (0, 0.5))
    
    elif model=='betaPlusGaussian' or model=='betaPlusGaussianAndIsotropic': 
        
        # Manually draw initial walkers from a small region
        initial_mu_chis = draw_initial_walkers_uniform(nWalkers, (0.2,0.4)) # TODO: change these?
        initial_sigma_chis = draw_initial_walkers_uniform(nWalkers, (0.17,0.25))
        initial_mu_costs = draw_initial_walkers_uniform(nWalkers, (0.2,0.4))
        initial_sigma_costs = draw_initial_walkers_uniform(nWalkers, (0.17,0.25))
        initial_Bqs = np.random.normal(loc=0, scale=3, size=nWalkers)

        # Put together all initial walkers into a single array
        initial_walkers = np.transpose(
            [initial_mu_chis, initial_sigma_chis, initial_mu_costs, initial_sigma_costs, initial_Bqs]
        )   
            
# Otherwise resume existing chain
else:
    
    print('\nOld chains found, loading and picking up where they left off ... ' )
    
    # Load existing file and iterate run version
    old_chain = np.concatenate([np.load(chain) for chain in old_chains], axis=1)
    run_version = int(old_chains[-1][-6:-4])+1

    # Strip off any trailing zeros due to incomplete run
    goodInds = np.where(old_chain[0,:,0]!=0.0)[0]
    old_chain = old_chain[:,goodInds,:]

    # Initialize new walker locations to final locations from old chain
    initial_walkers = old_chain[:,-1,:]
    
    # Figure out how many more steps we need to take 
    nSteps = nSteps - old_chain.shape[1]
    
        
print('Initial walkers:')
print(initial_walkers)


"""
Launching emcee
"""

if nSteps>0: # if the run hasn't already finished

    assert dim==initial_walkers.shape[1], "'dim' = wrong number of dimensions for 'initial_walkers'"

    print(f'\nLaunching emcee with {dim} hyper-parameters, {nSteps} steps, and {nWalkers} walkers ...')
    
    # for metadata: logL, Neff, and min(Nsamps)   
    dtype_for_blobs = [ ("logL", float), ("Neff", float), ("minNsamps", float)]
    
    # Args to pass to posterior function: 
    args = [sampleDict,injectionDict,priorDict]
    if model=='betaPlusDoubleGaussianJustChi':
        args.append(hyperparams[pop][2:-1])
    elif model=='betaPlusDoubleGaussianJustCosTheta':
        args.append(hyperparams[pop][:2])

    # Specify which posterior model we're using 
    POSTERIOR_MODEL = globals()[model]
    
    # make sampler object
    sampler = mc.EnsembleSampler(
        nWalkers,
        dim,
        POSTERIOR_MODEL,
        args=args,
        kwargs={'only_prior':only_prior},
        threads=16,
        blobs_dtype=dtype_for_blobs 
    )

    print('\nRunning emcee ... ')

    for i,result in enumerate(sampler.sample(initial_walkers,iterations=nSteps)):

        # Save every 10 iterations
        if i%10==0:
            np.save("{0}_r{1:02d}.npy".format(output_tmp,run_version),sampler.chain)
            np.save("{0}_r{1:02d}_Neffs.npy".format(output_tmp,run_version),sampler.get_blobs()['Neff'])
            np.save("{0}_r{1:02d}_minNsamps.npy".format(output_tmp,run_version),sampler.get_blobs()['minNsamps'])
            np.save("{0}_r{1:02d}_logL.npy".format(output_tmp,run_version),sampler.get_blobs()['logL'])

        # Print progress every 100 iterations
        if i%100==0:
            print(f'On step {i} of {nSteps}', end='\r')

    # Save raw output chains
    np.save("{0}_r{1:02d}.npy".format(output_tmp,run_version),sampler.chain)
    np.save("{0}_r{1:02d}_Neffs.npy".format(output_tmp,run_version),sampler.get_blobs()['Neff'])
    np.save("{0}_r{1:02d}_minNsamps.npy".format(output_tmp,run_version),sampler.get_blobs()['minNsamps'])
    np.save("{0}_r{1:02d}_logL.npy".format(output_tmp,run_version),sampler.get_blobs()['logL'])


"""
Running post processing and saving results
"""

print('\nDoing post processing ...')

# Load in data in correct format
if nSteps==0:
    run_version=run_version-1

if run_version==0:
    chainRaw = np.load("{0}_r00.npy".format(output_tmp))
    NeffsRaw = np.load("{0}_r00_Neffs.npy".format(output_tmp))
    minNsampsRaw = np.load("{0}_r00_minNsamps.npy".format(output_tmp))
    logLRaw = np.load("{0}_r00_logL.npy".format(output_tmp))
else:
    chainRaw = np.concatenate([np.load(chain) for chain in np.sort(glob.glob("{0}_r??.npy".format(output_tmp)))], axis=1)
    NeffsRaw = np.concatenate([np.load(chain) for chain in np.sort(glob.glob("{0}_r??_Neffs.npy".format(output_tmp)))])
    minNsampsRaw = np.concatenate([np.load(chain) for chain in np.sort(glob.glob("{0}_r??_minNsamps.npy".format(output_tmp)))])
    logLRaw = np.concatenate([np.load(chain) for chain in np.sort(glob.glob("{0}_r??_logL.npy".format(output_tmp)))])

blobsRaw = {
    'Neff':NeffsRaw,
    'minNsamps':minNsampsRaw, 
    'logL':logLRaw
}

# Run post-processing 
chainDownsampled, blobsDownsampled = processEmceeChain(chainRaw, blobs=blobsRaw)

# Parameter names of our sampled hyper-parameters
if model in ['betaPlusDoubleGaussian', 'betaPlusDoubleGaussian2', 'betaPlusDoubleGaussian2_noSelectionSpins']:
    params_list = ['mu_chi', 'sigma_chi', 'mu1_cost','sigma1_cost', 'mu2_cost', 'sigma2_cost', 'MF_cost', 'Bq']
elif model=='betaPlusGaussian': 
    params_list = ['mu_chi', 'sigma_chi', 'mu_cost','sigma_cost', 'Bq']
elif model=='betaPlusGaussianAndIsotropic': 
    params_list = ['mu_chi', 'sigma_chi', 'sigma_cost', 'MF_cost', 'Bq']
elif model=='betaPlusDoubleGaussianJustChi': 
    params_list = ['mu_chi', 'sigma_chi']
elif model=='betaPlusDoubleGaussianJustCosTheta': 
    params_list = ['mu1_cost','sigma1_cost', 'mu2_cost', 'sigma2_cost', 'MF_cost']
    
# Format output into an easily readable format 
results = {
    **{
        p:{'unprocessed':chainRaw[:,:,i].tolist(), 'processed':chainDownsampled[:,i].tolist()} for i,p in enumerate(params_list)
    },
    **{
        'Neff':blobsDownsampled['Neff'].tolist(),
        'minNsamps':blobsDownsampled['minNsamps'].tolist(), 
        'logL':blobsDownsampled['logL'].tolist(), 
        'events_used':[str(k) for k in sampleDict.keys()]
    }
}
    
# Save
savename = froot+f"PopulationInferenceOutput/{model}/{model_savename}.json"
with open(savename, "w") as outfile:
    json.dump(results, outfile)
print(f'Done! Run saved at {savename}')
