import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.special import erf
import sys
import pandas as pd

# Pass number of of events to inject via commandline 
nevents = sys.argv[1]

# repository root
root = '/home/simona.miller/measuring-bbh-component-spin/'

# cycle through populations
pop_names = ['population1_highSpinPrecessing', 'population2_mediumSpin', 'population3_lowSpinAligned']

for j,pop_name in enumerate(pop_names):

    # Load in injected population
    injections = pd.read_json(f'../../Data/InjectedPopulationParameters/{pop_name}.json')
    injections.sort_index(inplace=True)
    n_total = len(injections)

    # Choose random set to inject
    to_inject = np.random.choice(range(n_total),size=int(nevents),replace=False)
    
    # Save as text file for reference
    np.savetxt(f'injlist_{pop_name}_{nevents}events.txt',to_inject,fmt="%d")

    # Write dag file in the condor subfolder
    dagfile=f'./condor/bilby_{pop_name}.dag'
    with open(dagfile,'w') as df: 
        for i in to_inject:
            df.write('JOB {0} {1}Code/IndividualInference/condor/bilby_{2}.sub\n'.format(int(i),root,pop_name))
            df.write('VARS {0} jobNumber="{0}" json="{1}Data/InjectedPopulationParameters/{2}.json" outdir="{1}Data/IndividualInferenceOutput/{2}"\n\n'.format(int(i),root,pop_name))