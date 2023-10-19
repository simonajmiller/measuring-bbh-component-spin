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

# Events to ignore (i.e. those that did not converge in bilby) 
with open('../../Data/PopulationInferenceInput/to_ignore.json','r') as jf:
    to_ignore = json.load(jf)
    
# Also we will impose a mass cut on the injected value 
mMin_cut = 8

# Cycle through the three populations
pop_names = ['population1_highSpinPrecessing', 'population2_mediumSpin', 'population3_lowSpinAligned']

for pop in pop_names: 
    
    print('\nCalculating for '+pop+' ...')

    # Read list of event names for this population
    pop_injlist = np.sort(np.loadtxt(f'../IndividualInference/injlist_{pop}_400events.txt'))
    
    # Events to ignore
    ignore_list = to_ignore[pop]
    
    # Events to run 
    list_for_sampleDict = [event for event in pop_injlist if event not in ignore_list]

    sampleDict = {}
    
    # Cycle through events
    for event in list_for_sampleDict: 
                
        job_name = "job_{0:05d}_result.json".format(int(event))
        fname = individual_inference_output_folder+f'{pop}/'+job_name

        # If the result exists, load in data + format correctly    
        if os.path.exists(fname):
            
            print(str(int(event))+'        ', end='\r')
            
            with open(fname,'r') as jf:
                result = json.load(jf)

            # Fetch injected parameters
            injected_params = {
                'm1':result['injection_parameters']['mass_1_source'],
                'm2':result['injection_parameters']['mass_2_source'],
                'z':result['injection_parameters']['redshift'],
                'a1':result['injection_parameters']['a_1'],
                'a2':result['injection_parameters']['a_2'],
                'cost1':result['injection_parameters']['cos_tilt_1'],
                'cost2':result['injection_parameters']['cos_tilt_2'],
            }

            # Ignore those with injected masses that don't pass our cut 
            if injected_params['m1'] < mMin_cut or injected_params['m2'] < mMin_cut: 
                continue
                
            # Cut out samples that don't mass the optimal SNR cut of 10 from our selection function
            H1_opt_snr = np.asarray(result['posterior']['content']['H1_optimal_snr'])
            L1_opt_snr = np.asarray(result['posterior']['content']['L1_optimal_snr'])
            V1_opt_snr = np.asarray(result['posterior']['content']['V1_optimal_snr'])
            optimal_snr = np.sqrt(H1_opt_snr**2 + L1_opt_snr**2 + V1_opt_snr**2)
            mask = optimal_snr > 10

            # Fetch samples
            m1 = np.asarray(result['posterior']['content']['mass_1_source'])[mask]
            m2 = np.asarray(result['posterior']['content']['mass_2_source'])[mask]
            z = np.asarray(result['posterior']['content']['redshift'])[mask]
            chi1 = np.asarray(result['posterior']['content']['a_1'])[mask]
            chi2 = np.asarray(result['posterior']['content']['a_2'])[mask]
            cost1 =  np.asarray(result['posterior']['content']['cos_tilt_1'])[mask]
            cost2 =  np.asarray(result['posterior']['content']['cos_tilt_2'])[mask]

            # Downsample to 5000 samples per event
            nsamps = min(len(m1),5000)
            idxs = np.random.choice(len(m1), size=nsamps)
                
            # If not enough samples pass the cut, ignore this event
            if nsamps<100:
                print(f'event {int(event)} has less than 100 samples with optimal snr > 10: {injected_params}') 
                continue

            # Generate fake gaussian spin posteriors with a variety of sigmas
            gaussian_spins_dict = {}
            sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 1]

            for param, bounds in zip(['a1', 'a2', 'cost1', 'cost2'], [(0,1), (0,1), (-1,1), (-1,1)]): 

                gaussian_spins_dict_param = {}         
                for sigma in sigmas: 

                    # go from true parameter to observed parameter for a given 
                    # measurement uncertainty
                    param_true = injected_params[param]
                    param_obs = param_true + np.random.normal(loc=0, scale=sigma)

                    # generate "posterior" 
                    gaussian_posterior_all = np.random.normal(loc=param_obs, scale=sigma, size=nsamps*10)

                    # impose bounds
                    mask = (gaussian_posterior_all>=bounds[0]) & (gaussian_posterior_all<=bounds[1])
                    gaussian_posterior = np.random.choice(gaussian_posterior_all[mask], size=nsamps)

                    # add to dict
                    gaussian_spins_dict_param[f'gaussian_sigma_{sigma}'] = gaussian_posterior.tolist()

                gaussian_spins_dict[param] = gaussian_spins_dict_param

            # Make dict
            sampleDict[str(int(event))] = {
                'm1':m1[idxs].tolist(),
                'm2':m2[idxs].tolist(),
                'z':z[idxs].tolist(),
                'a1':{'bilby_posterior':chi1[idxs].tolist(), **gaussian_spins_dict['a1']},
                'a2':{'bilby_posterior':chi2[idxs].tolist(), **gaussian_spins_dict['a2']},
                'cost1':{'bilby_posterior':cost1[idxs].tolist(), **gaussian_spins_dict['cost1']},
                'cost2':{'bilby_posterior':cost2[idxs].tolist(), **gaussian_spins_dict['cost2']},
                'dVc_dz': dVdz(z[idxs]).tolist(), # Precompute the differential comoving volume, dV/dz, for each sample redshift. 
                'injected_params':injected_params
            }

        else:
            print(f"event {int(event)} not found")
            pass
            
    print('\nNumber of events in pop: ')
    print(len(sampleDict.keys()))
    
    # Save sampleDict in folder where population inference input goes 
    with open(f'../../Data/PopulationInferenceInput/sampleDict_{pop}.json', 'w') as f:
        json.dump(sampleDict, f)
    
        
        
     