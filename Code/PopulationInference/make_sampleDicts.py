import numpy as np
import os
import json
import sys
sys.path.append('/home/simona.miller/measuring-bbh-component-spin/Code/GeneratePopulations/')
from helper_functions import dVdz

"""

*** ADD DOCUMENTATION HERE ***

"""

# Filepath for where bilby outputs saved
individual_inference_output_folder = '../../Data/IndividualInferenceOutput/'

# Cycle through the three populations
pop_names = ['population1_highSpinPrecessing', 'population2_mediumSpin', 'population3_lowSpinAligned']
for pop in pop_names: 
    
    print('\nCalculating for '+pop+' ...')

    # Read list of event names for this population
    pop_injlist = np.sort(np.loadtxt(f'../IndividualInference/injlist_{pop}_400events.txt'))

    sampleDict = {}
    
    # Cycle through events
    for event in pop_injlist: 
        print(str(int(event))+'        ', end='\r')
        
        job_name = "job_{0:05d}_result.json".format(int(event))
        fname = individual_inference_output_folder+f'{pop}/'+job_name

        # If the result exists, load in data + format correctly    
        if os.path.exists(fname): 
            
            with open(fname,'r') as jf:
                result = json.load(jf)
            
            try:
                
                # Fetch injected parameters
                injected_params = {
                    'm1':result['injection_parameters']['mass_1_source'],
                    'm2':result['injection_parameters']['mass_2_source'],
                    'z':result['injection_parameters']['redshift'],
                    'chi1':result['injection_parameters']['a_1'],
                    'chi2':result['injection_parameters']['a_2'],
                    'cost1':result['injection_parameters']['cos_tilt_1'],
                    'cost2':result['injection_parameters']['cos_tilt_2'],
                }
                
                # Fetch samples
                m1 = np.asarray(result['posterior']['content']['mass_1_source'])
                m2 = np.asarray(result['posterior']['content']['mass_2_source'])
                z = np.asarray(result['posterior']['content']['redshift'])
                chi1 = np.asarray(result['posterior']['content']['a_1'])
                chi2 = np.asarray(result['posterior']['content']['a_2'])
                cost1 =  np.asarray(result['posterior']['content']['cos_tilt_1'])
                cost2 =  np.asarray(result['posterior']['content']['cos_tilt_2'])
            
                # Downsample to 5000 samples per event
                idxs = np.random.choice(len(m1), size=min(len(m1),5000))

                sampleDict[str(int(event))] = {
                    'm1':m1[idxs].tolist(),
                    'm2':m2[idxs].tolist(),
                    'z':z[idxs].tolist(),
                    'a1':chi1[idxs].tolist(),
                    'a2':chi2[idxs].tolist(),
                    'cost1':cost1[idxs].tolist(),
                    'cost2':cost2[idxs].tolist(),
                    'dVc_dz': dVdz(z[idxs]).tolist(), # Precompute the differential comoving volume, dV/dz, for each sample redshift. 
                    'injected_params':injected_params
                }
                                
            except Exception as e:
                print(e, end='\r') 
                
        else:
            print(f"event {int(event)} not found")
            
    # Save sampleDict in folder where population inference input goes 
    with open(f'../../Data/PopulationInferenceInput/sampleDict_{pop}_full_mass_range_temp.json', 'w') as f:
        json.dump(sampleDict, f)
    
        
        
     