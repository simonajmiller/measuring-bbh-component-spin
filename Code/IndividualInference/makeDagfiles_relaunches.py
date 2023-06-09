import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.special import erf
import sys
import pandas as pd

sys.path.append('../../Data/IndividualInferenceOutput/inspect_results')
from inspect_results_helper_functions import find_all_missing,is_file_older_than_one_day 

# repository root
root = '/home/simona.miller/measuring-bbh-component-spin/'

# get today's date, as passed through the commandline, to tag the .dag file
todays_date = sys.argv[1] 

# cycle through populations
pop_names = ['population1_highSpinPrecessing', 'population2_mediumSpin', 'population3_lowSpinAligned']

for j,pop_name in enumerate(pop_names):
    
    num_injections = 300
    
    # Load all jobs
    jobs = np.loadtxt(f'injlist_{pop_name}_' + str(num_injections) + 'events.txt')
    
    # Find out which are missing
    print(f'Finding missing jobs for {pop_name}...')
    missing_jobs,_ = find_all_missing(pop_name, jobs, fname_template=f'{root}Data/IndividualInferenceOutput/'+'{1}/job_{0:05d}_result.json', returnExceptions=False)
    
    print(f'{len(missing_jobs)} missing jobs for {pop_name}: ', missing_jobs)

        
    # Re-launch those files whos bilby outputs haven't been edited in over 1 day. 
    # Use the bilby .log file as a proxy for this. 
    to_inject = []
    for job in missing_jobs: 
        fpath = f'{root}Data/IndividualInferenceOutput/{pop_name}/'+'job_{0:05d}.log'.format(int(job))
        if is_file_older_than_one_day(fpath): 
            to_inject.append(job)
            
    print(f'Re-launching the following {len(to_inject)} jobs for {pop_name}: ', to_inject, '\n')
    
    # Write dag file in the condor subfolder
    dagfile=f'./condor/bilby_{pop_name}_dags/bilby_{pop_name}_relaunch_{todays_date}.dag'
    with open(dagfile,'w') as df: 
        for i in to_inject:
            df.write('JOB {0} {1}Code/IndividualInference/condor/bilby_{2}.sub\n'.format(int(i),root,pop_name))
            df.write('VARS {0} jobNumber="{0}" json="{1}Data/InjectedPopulationParameters/{2}.json" outdir="{1}Data/IndividualInferenceOutput/{2}"\n\n'.format(int(i),root,pop_name))