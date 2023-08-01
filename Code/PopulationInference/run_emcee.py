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

# For calculate cuts in chirpmass
def chirpmass(m1, m2): 
    q = m2/m1
    Mc = (q/(1.+q)**2)**(3./5.)*(m1+m2)
    return Mc

mMin = 6
minPossibleMChirp = chirpmass(mMin,mMin)

# Parse commandline arguments
p = argparse.ArgumentParser()
p.add_argument('--model')
p.add_argument('--date')
p.add_argument('--pop')
p.add_argument('--nevents', type=int)
p.add_argument('--nsteps', type=int)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--posterior-key', default='bilby_posterior')
p.add_argument('--lower-chirpmass-cut', type=int, default=minPossibleMChirp)
args = p.parse_args()

# Pass population and number of events via commandline
pop = args.pop
nevents = args.nevents

# Bilby samples or mock gaussian samples
posterior_key = args.posterior_key

# Chirpmass cut on data?
chirpmass_cut = args.lower_chirpmass_cut

# Model
model = args.model 
allowed_models = ["betaPlusDoubleGaussian", "betaPlusGaussian"]
assert model in allowed_models, f"Given model ('{model}') not in allowed models ({allowed_models})"

# How to save
model_savename = f"{args.date}_{model}_pop{pop}_{nevents}events"

# Add flags to savename if not default
if posterior_key != 'bilby_posterior': 
    model_savename += f"_{posterior_key}"
if args.seed != 0:
    model_savename += f"_{seed}" 
if chirpmass_cut > minPossibleMChirp: 
    model_savename += f"_chirpmasscut{chirpmass_cut}" 

# set seed for reproducibility
seed = args.seed if args.seed!=0 else 2345 # arbitrary
np.random.seed(seed)

print(f'Running {model_savename} ...')

# File path root for where to store data 
froot = "/home/simona.miller/measuring-bbh-component-spin/Data/"

# Define emcee parameters
nWalkers = 20       # number of walkers 
dim = 8             # dimension of parameter space (number hyper params)
nSteps = args.nsteps    # number of steps for chain

# Names of the different populations
pop_names = {
    '1':'population1_highSpinPrecessing', 
    '2':'population2_mediumSpin',
    '3':'population3_lowSpinAligned'
}

# Load sampleDict
with open(froot+f"PopulationInferenceInput/sampleDict_{pop_names[pop]}_full_mass_range.json", 'r') as f: 
    sampleDict_full = json.load(f)
    
## Condition sampleDict: 
    
# 1. Impose chirpmass cut if necessary 
if chirpmass_cut > minPossibleMChirp: 
    sampleDict_temp1 = {}
    for event in sampleDict_full.keys(): 
        # fetch event's injected parameters and calculate the chirp mass
        inj_m1 = sampleDict_full[event]['injected_params']['m1']
        inj_m2 = sampleDict_full[event]['injected_params']['m2']
        inj_mc = chirpmass(inj_m1, inj_m2)
        # only keep events with injected chirpmass greater than the given cut
        if inj_mc >= chirpmass_cut: 
            sampleDict_temp1[event] = sampleDict_full[event]
else: 
    sampleDict_temp1 = sampleDict_full
    
print(f'\nLoaded sampleDict had {len(sampleDict_full)} events; after chirpmass cut, {len(sampleDict_temp1)} remaining.')

# 2. Choose subset of sampleDict if necessary
if int(nevents)<len(sampleDict_temp1.keys()): 
    keys = [key for key in sampleDict_temp1.keys()]
    events = np.random.choice(keys, size=int(nevents), replace=False)
    sampleDict_temp2 = {event:sampleDict_temp1[event] for event in events}
else: 
    sampleDict_temp2 = sampleDict_temp1
    
    if int(nevents)>len(sampleDict_temp2.keys()): 
        print('Too many events requested. Changing nevents to max # possible.')  
        new_nevents = len(sampleDict_temp2.keys())
        model_savename = model_savename.replace(str(nevents), str(new_nevents))
        print('Savename updated: ', model_savename)
        nevents = new_nevents

print(f'After nevents cut, {len(sampleDict_temp2)} remaining.')
    
# 3. Choose the correct set of posterior samples from the sampleDict
sampleDict = {}
for event in sampleDict_temp2.keys():
    # for masses and redshifts always use bilby posteriors
    d1 = {p:sampleDict_temp2[event][p] for p in ['m1', 'm2', 'z', 'dVc_dz']}
    # for spin magnitude and tilts, option to use bilby or gaussian posteriors
    d2 = {p:sampleDict_temp2[event][p][posterior_key] for p in ['a1', 'a2', 'cost1', 'cost2']}
    # combine into final sampleDict
    sampleDict[event] = {**d1, **d2}
    
# Load injectionDict
with open(froot+"PopulationInferenceInput/injectionDict_full_mass_range.json", 'r') as f: 
    injectionDict_full = json.load(f)
    
# Chirpmass cut on injectionDict if necessary: 
if chirpmass_cut > minPossibleMChirp:
    injectionDict = {}
    inj_mc = chirpmass(np.asarray(injectionDict_full['m1']), np.asarray(injectionDict_full['m2']))
    mask = inj_mc >= chirpmass_cut
    for param in injectionDict_full.keys(): 
        array_with_cut = np.asarray(injectionDict_full[param])[mask]
        injectionDict[param] = array_with_cut.tolist()
else: 
    injectionDict = injectionDict_full
    
print(f"Loaded injectionDict had {len(injectionDict_full['m1'])} events; after chirpmass cut, {len(injectionDict['m1'])} remaining.\n")
    
# Will save emcee chains temporarily in the .tmp folder in this directory
output_folder_tmp = froot+"PopulationInferenceOutput/.tmp/"
output_tmp = output_folder_tmp+model_savename


"""
Initializing emcee walkers or picking up where an old chain left off.
"""

# Set prior bounds 
priorDict = {
    'mu_chi':(0., 1.),
    'sigma_chi':(0.07, 0.5),
    'mu_cost':(-1., 1.),
    'sigma_cost':(0.16, 0.8),
    'MF_cost':(0., 1.)
}

# Search for existing chains
old_chains = np.sort(glob.glob("{0}_r??.npy".format(output_tmp)))

# If no chain already exists, begin a new one
if len(old_chains)==0:
    
    print('\nNo old chains found, generating initial walkers ... ')

    run_version = 0

    # Initialize walkers 
    
    if model=='betaPlusDoubleGaussian':
        
        # True mu_chi, sigma_chi, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, f, Bq:
        hyperparams = { 
            '1':[0.55, 0.26, 0.19, 0.18, 0.42, 0.75, 0.55, 0.96],  
            '2':[0.32, 0.16, 0.33, 0.64, 0.59, 0.40, 0.36, 0.96],
            '3':[0.19, 0.12, -0.98, 0.44, 0.98, 0.31, 0.26, 0.96]
        }
        # Draw initial walkers close to true value
        initial_walkers = np.transpose([draw_initial_walkers_uniform(nWalkers, (hyperparams[pop][i]-0.02, hyperparams[pop][i]+0.02)) for i in range(dim)])
    
    elif model=='betaPlusGaussian': 
        
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
    
    # for metadata 
    dtype_for_blobs = [ ("logL", float), ("Neff", float), ("minNsamps", float)]

    # specify which hyper-posterior model we're using 
    if model=='betaPlusDoubleGaussian':
        POSTERIOR_MODEL = betaPlusDoubleGaussian
    elif model=='betaPlusGaussian':
        POSTERIOR_MODEL = betaPlusGaussian
    
    # make sampler object
    sampler = mc.EnsembleSampler(
        nWalkers,
        dim,
        POSTERIOR_MODEL, # model in posteriors.py
        args=[sampleDict,injectionDict,priorDict], # arguments passed to model in posteriors.py
        threads=16,
        blobs_dtype=dtype_for_blobs #for metadata: logL, Neff, and min(Nsamps)    
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

# Format output into an easily readable format 
if model=='betaPlusDoubleGaussian':
    results = {
        'mu_chi':{'unprocessed':chainRaw[:,:,0].tolist(), 'processed':chainDownsampled[:,0].tolist()},
        'sigma_chi':{'unprocessed':chainRaw[:,:,1].tolist(), 'processed':chainDownsampled[:,1].tolist()},
        'mu1_cost':{'unprocessed':chainRaw[:,:,2].tolist(), 'processed':chainDownsampled[:,2].tolist()},
        'sigma1_cost':{'unprocessed':chainRaw[:,:,3].tolist(), 'processed':chainDownsampled[:,3].tolist()},
        'mu2_cost':{'unprocessed':chainRaw[:,:,4].tolist(), 'processed':chainDownsampled[:,4].tolist()},
        'sigma2_cost':{'unprocessed':chainRaw[:,:,5].tolist(), 'processed':chainDownsampled[:,5].tolist()},
        'MF_cost':{'unprocessed':chainRaw[:,:,6].tolist(), 'processed':chainDownsampled[:,6].tolist()},
        'Bq':{'unprocessed':chainRaw[:,:,7].tolist(), 'processed':chainDownsampled[:,7].tolist()}, 
        'Neff':blobsDownsampled['Neff'].tolist(),
        'minNsamps':blobsDownsampled['minNsamps'].tolist(), 
        'logL':blobsDownsampled['logL'].tolist(), 
        'events_used':[str(k) for k in sampleDict.keys()]
    } 
elif model=='betaPlusGaussian': 
    results = {
        'mu_chi':{'unprocessed':chainRaw[:,:,0].tolist(), 'processed':chainDownsampled[:,0].tolist()},
        'sigma_chi':{'unprocessed':chainRaw[:,:,1].tolist(), 'processed':chainDownsampled[:,1].tolist()},
        'mu_cost':{'unprocessed':chainRaw[:,:,2].tolist(), 'processed':chainDownsampled[:,2].tolist()},
        'sigma_cost':{'unprocessed':chainRaw[:,:,3].tolist(), 'processed':chainDownsampled[:,3].tolist()},
        'Bq':{'unprocessed':chainRaw[:,:,4].tolist(), 'processed':chainDownsampled[:,4].tolist()}, 
        'Neff':blobsDownsampled['Neff'].tolist(),
        'minNsamps':blobsDownsampled['minNsamps'].tolist(), 
        'logL':blobsDownsampled['logL'].tolist(), 
        'events_used':[str(k) for k in sampleDict.keys()]
        } 

# Save
savename = froot+f"PopulationInferenceOutput/{model}/{model_savename}.json"
with open(savename, "w") as outfile:
    json.dump(results, outfile)
print(f'Done! Run saved at {savename}')
