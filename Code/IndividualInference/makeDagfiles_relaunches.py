import numpy as np
import json
import sys
import pandas as pd
import os 

# repository root
root = '/home/simona.miller/measuring-bbh-component-spin/'

# cycle through populations
pop_names = ['population1_highSpinPrecessing', 'population2_mediumSpin', 'population3_lowSpinAligned']

# today's date 
date = '070523'

# Load in dict of jobs to relaunch
to_inject_dict_fname = f'to_inject_dict_{date}.json'
with open(to_inject_dict_fname, 'r') as f:
    to_inject_dict = json.load(f)

for j,pop_name in enumerate(pop_names):

    # Load in injected population
    injections = pd.read_json(f'../../Data/InjectedPopulationParameters/{pop_name}_fullmassrange.json')
    injections.sort_index(inplace=True)
    n_total = len(injections)

    # Set to inject
    to_inject = to_inject_dict[pop_name]
    
    # Delete all old files for that job 
    for job in to_inject:
        fpath = f'{root}Data/IndividualInferenceOutput/{pop_name}/'+'job_{0:05d}'.format(int(job))
        os.system(f"rm {fpath}*")
        fpath2 = f'{root}Data/IndividualInferenceOutput/{pop_name}/'+'.job_{0:05d}'.format(int(job))
        os.system(f"rm {fpath2}*")
    
    # Write dag file in the condor subfolder
    dagfile=f'./condor/bilby_{pop_name}_dags/bilby_{pop_name}_relaunch_{date}.dag'
    with open(dagfile,'w') as df: 
        for i in to_inject:
            df.write('JOB {0} {1}Code/IndividualInference/condor/bilby_{2}.sub\n'.format(int(i),root,pop_name))
            df.write('VARS {0} jobNumber="{0}" json="{1}Data/InjectedPopulationParameters/{2}_fullmassrange.json" outdir="{1}Data/IndividualInferenceOutput/{2}"\n\n'.format(int(i),root,pop_name))