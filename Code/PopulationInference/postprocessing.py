import numpy as np
import sys
import acor.acor as acor

def processEmceeChain(sample_chain, blobs=None, removeStuckWalkers=True):
    
    """
    Script with function to do some basic post-processing of emcee chains.

    Specifically, this fxn will load raw emcee output, an array of format [# walkers, # steps, # variables],
    downsample each walker's chain to obtain independent samples, and collapse all walkers to produce a 2D array
    of the format [# steps, # variables]
    
    Parameters
    ----------
    sample_chain : `numpy.array`
        raw emcee output samples of dimension [# walkers, # steps, # variables]
    blobs : dict (optional)
        dictionary containing metadata that also needs to be downsampled in the same way as sample_chain
    removeStuckWalkers : boolean (optional)
        do you want to get rid of any walkers that got stuck?
        
    Returns
    -------
    chainDownsampled : `numpy.array`
        processed chains of dimension [# steps, # variables]
    blobsDownsampled : dict (optional)
        if blobs passed, return processed blobs 
    """

    print("Shape of sample chain:")
    print(np.shape(sample_chain))

    goodInds = np.where(sample_chain[0,:,0]!=0.0)[0]
    sample_chain = sample_chain[:,goodInds,:]
    
    # Get rid of stuck walkers 
    if removeStuckWalkers: 
        goodWalkers = np.unique(np.where(sample_chain[:,0,:]-sample_chain[:,-1,:] != 0)[0])
        sample_chain = sample_chain[goodWalkers,:,:]
        
        print('Shape of sample chain after getting rid of stuck walkers:')
        print(np.shape(sample_chain))
    
    # Read off the number of walkers, the number of steps, and the dimensionality of our parameter space
    nWalkers,nSteps,dim = sample_chain.shape
    
    # Burn first third of the chain (make sure to plot raw chains and make sure this is reasonable!)
    chainBurned = sample_chain[:,int(np.floor(nSteps/4.)):,:]
    print("Shape of burned chain:")
    print(np.shape(chainBurned))

    # To prepare for downsampling, compute the correlation length of our chains.
    # We'll compute separate correlation lengths for each of our parameters 
    corrTotal = np.zeros(dim)
    for i in range(dim):

        # For each parameter, find a mean correlation length averaged across all our walkers
        for j in range(nWalkers):
            (tau,mean,sigma) = acor.acor(chainBurned[j,:,i])
            corrTotal[i] += 2.*tau/(nWalkers)

    # Finally, for safety we'll choose the maximum correlation length over all our variables
    maxCorLength = np.max(corrTotal)
    print("Max correlation length across parameters:")
    print(maxCorLength)

    # Down-sample by the correlation length
    chainDownsampled = chainBurned[:,::int(maxCorLength),:]
    print("Shape of downsampled chain:")
    print(np.shape(chainDownsampled))

    # Flatten across all our walkers to produce a 2D array
    chainDownsampled = chainDownsampled.reshape((-1,len(chainDownsampled[0,0,:])))
    print("Shape of downsampled chain post-flattening:")
    print(np.shape(chainDownsampled))
    
    if blobs is None:
        return chainDownsampled 
    else: 
        blobsDownsampled = {}
        
        # if passed metadata, perform the same operations to it
        for key in blobs:
            
            # each blob chain has dimensions (# samples, # walkers)
            blob_chain = blobs[key]
            
            # take transpose and add dimension to get into (# walkers, # samples, 1) to match the chains
            blob_chain = np.transpose(blob_chain)
            blob_chain = np.expand_dims(blob_chain, -1)
            blob_chain = blob_chain[:,goodInds,:]
            
            if removeStuckWalkers:
                blob_chain = blob_chain[goodWalkers,:,:]
            
            # burn first quarter
            blobBurned = blob_chain[:, int(np.floor(nSteps/4.)):]
            
            # downsample by autocorrelation length
            blobDownsampled = blobBurned[:,::int(maxCorLength),:]
            print("Shape of downsampled blobs:")
            print(np.shape(blobDownsampled))
                    
            # flatten  
            blobDownsampled = blobDownsampled.reshape((-1,len(blobDownsampled[0,0,:])))
            print("Shape of downsampled blobs post-flattening:")
            print(np.shape(blobDownsampled))
                        
            # get rid of added extra dimension and store in dict to return
            blobsDownsampled[key] = blobDownsampled[:,0]
            
        return chainDownsampled, blobsDownsampled


'''
Usage:
    $ python postprocessing.py emcee_samples_example.npy
'''
if __name__=="__main__":

    fname = sys.argv[1]

    fname_root = fname.split('_r')[0]
    print(fname)
    print(fname_root)
    
    run_version = int(fname[-6:-4])
    print('run version:',run_version)

    # Load sample chain
    if run_version==0:
        sample_chain = np.load(fname)
    else: 
        sample_chain_arr = [np.load(fname_root+f'_r{r:02d}.npy') for r in np.arange(run_version+1)]
        for chain in sample_chain_arr: 
            print(chain.shape)
        sample_chain = np.concatenate(sample_chain_arr, axis=1) 
     
    # Run post-processing function defined above 
    chainDownsampled = processEmceeChain(sample_chain)
    
    # Save 
    np.save('{0}_processed.npy'.format(fname.split('.npy')[0]), chainDownsampled)
