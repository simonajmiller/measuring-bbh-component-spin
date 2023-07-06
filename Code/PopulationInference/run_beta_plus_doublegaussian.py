import numpy as np
import argparse
import glob
import emcee as mc
import json
import sys
from posterior_helper_functions import draw_initial_walkers_uniform
from posteriors import betaPlusDoubleGaussian
from postprocessing import processEmceeChain 

"""
Definitions and loading data
"""

# Parse commandline arguments
p = argparse.ArgumentParser()
p.add_argument('--date')
p.add_argument('--pop')
p.add_argument('--nevents', type=int)
p.add_argument('--nsteps', type=int)
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()

# Pass population and number of events via commandline
pop = args.pop
nevents = args.nevents

# Model
model = "betaPlusDoubleGaussian"
if args.seed!=0:
    model_savename = f"{args.date}_{model}_pop{pop}_{nevents}events_{seed}" 
else:
    model_savename = f"{args.date}_{model}_pop{pop}_{nevents}events"

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

# Set prior bounds 
priorDict = {
    'mu_chi':(0., 1.),
    'sigma_chi':(0.07, 0.5),
    'mu_cost':(-1., 1.),
    'sigma_cost':(0.16, 0.8),
    'MF_cost':(0., 1.)
}

# Load sampleDict
pop_names = {
    '1':'population1_highSpinPrecessing', 
    '2':'population2_mediumSpin',
    '3':'population3_lowSpinAligned'
}
with open(froot+f"PopulationInferenceInput/sampleDict_{pop_names[pop]}_full_mass_range.json", 'r') as f: 
    sampleDict_full = json.load(f)
    
# Choose subset of sampleDict if necessary
if int(nevents)<400: 
    keys = [key for key in sampleDict_full.keys()]
    events = np.random.choice(keys, size=int(nevents), replace=False)
    sampleDict = {event:sampleDict_full[event] for event in events}
else: 
    sampleDict = sampleDict_full
    
# Load injectionDict
with open(froot+"PopulationInferenceInput/injectionDict_full_mass_range.json", 'r') as f: 
    injectionDict = json.load(f)

# Will save emcee chains temporarily in the .tmp folder in this directory
output_folder_tmp = froot+"PopulationInferenceOutput/.tmp/"
output_tmp = output_folder_tmp+model_savename


"""
Initializing emcee walkers or picking up where an old chain left off
"""

# Search for existing chains
old_chains = np.sort(glob.glob("{0}_r??.npy".format(output_tmp)))

# If no chain already exists, begin a new one
if len(old_chains)==0:
    
    print('\nNo old chains found, generating initial walkers ... ')

    run_version = 0

    # Initialize walkers 
    
    hyperparams = { 
         # True mu_chi, sigma_chi, mu1_cost, sigma1_cost, mu2_cost, sigma2_cost, f, Bq:
        '1':[0.55, 0.26, 0.19, 0.18, 0.42, 0.75, 0.55, 0.96],  
        '2':[0.32, 0.16, 0.33, 0.64, 0.59, 0.40, 0.36, 0.96],
        '3':[0.19, 0.12, -0.98, 0.44, 0.98, 0.31, 0.26, 0.96]
    }
    initial_walkers = np.transpose([draw_initial_walkers_uniform(nWalkers, (hyperparams[pop][i]-0.02, hyperparams[pop][i]+0.02)) for i in range(dim)])
            
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

    # make sampler object
    sampler = mc.EnsembleSampler(
        nWalkers,
        dim,
        betaPlusDoubleGaussian, # model in posteriors.py
        args=[sampleDict,injectionDict,priorDict], # arguments passed to betaPlusDoubleGaussian
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

# Save
savename = froot+f"PopulationInferenceOutput/{model}/{model_savename}.json"
with open(savename, "w") as outfile:
    json.dump(results, outfile)
print(f'Done! Run saved at {savename}')
