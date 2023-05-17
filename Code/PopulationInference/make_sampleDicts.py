import numpy as np
import os
import json
import astropy.cosmology as cosmo
import astropy.units as u
from astropy.cosmology import Planck15

"""
Prep for calculatiosn of z_prior and dVc_dz
"""

# Define constants
# Note that we assume a cosmology identical to that in pesummary
c = 299792458 # m/s
H_0 = 67900.0 # m/s/MPc
Omega_M = 0.3065 # unitless
Omega_Lambda = 1.0-Omega_M
year = 365.25*24.*3600

# Construct custom cosmology matching pesummary
cosmo = Planck15.clone(name='cosmo',Om0=Omega_M,H0=H_0/1e3,Tcmb0=0.)

def Hz(z):
    return H_0*np.sqrt(Omega_M*(1.+z)**3.+Omega_Lambda)

def dVc_dz(z,dl):
    dc = dl/(1.+z) # comoving distance 
    dVc_dz = 4*np.pi*c*(dc**2.)/Hz(z) # differential comoving volume 
    return dVc_dz

# build an interpolant between luminosity distance and redshift
zs_ref = np.linspace(0.,3.,1000)
DL_ref = cosmo.luminosity_distance(zs_ref).to(u.Mpc).value


"""
Genearate sampleDict
"""

# Filepath for where bilby outputs saved
individual_inference_output_folder = '../../Data/IndividualInferenceOutput/'

# Cycle through the three populations
pop_names = ['population1_highSpinPrecessing', 'population2_mediumSpin', 'population3_lowSpinAligned']
for pop in pop_names: 
    
    print('\nCalculating for '+pop+' ...')

    # Read list of event names for this population
    pop_injlist = np.sort(np.loadtxt(f'../IndividualInference/injlist_{pop}_300events.txt'))

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
                
                # Redshift prior
                # Note that DL is the observable, and so our mapping to z is dependent on our presumed cosmology.
                DL = np.interp(z,zs_ref,DL_ref) # interp from z -> luminosity distance

                # Precompute the differential comoving volume, dV/dz, for each sample redshift. 
                dVdz = dVc_dz(z, DL)

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
                    'dVc_dz': dVdz[idxs].tolist(), 
                    'injected_params':injected_params
                }
                                
            except Exception as e:
                print(e, end='\r') 
                
        else:
            print(f"event {int(event)} not found")
            
    # Save sampleDict in folder where population inference input goes 
    with open(f'../../Data/PopulationInferenceInput/sampleDict_{pop}_temp.json', 'w') as f:
        json.dump(sampleDict, f)
    
        
        
     