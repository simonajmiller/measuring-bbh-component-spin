import numpy as np
import os
import sys
import json
import pandas as pd
sys.path.append('/home/simona.miller/measuring-bbh-component-spin/Code/GeneratePopulations/')
from helper_functions import p_z as p_astro_z
from helper_functions import p_astro_masses
from helper_functions import dVdz


"""

*** ADD DOCUMENTATION HERE ***

"""


# Load in detected population from injections flat in chi and cos(theta)
detected_pop = pd.read_json(f'../../Data/InjectedPopulationParameters/flat_pop_full_mass_range_for_injection_dict.json')

# Detected values for paramters of interest (spins, masses, redshift, dVdz) 
chi1_det = np.sqrt(detected_pop['s1x'].values**2 + detected_pop['s1y'].values**2 + detected_pop['s1z'].values**2)
chi2_det = np.sqrt(detected_pop['s2x'].values**2 + detected_pop['s2y'].values**2 + detected_pop['s2z'].values**2)
cost1_det = detected_pop['s1z'].values/chi1_det
cost2_det = detected_pop['s2z'].values/chi2_det
m1_det = detected_pop['m1'].values
m2_det = detected_pop['m2'].values
z_det = detected_pop['z'].values
dVdz_det = dVdz(z_det)

# P_draw for spins = flat
p_draw_chi1 = np.ones(len(chi1_det))
p_draw_chi2 = np.ones(len(chi2_det))
p_draw_cost1 = 0.5*np.ones(len(cost1_det))
p_draw_cost2 = 0.5*np.ones(len(cost2_det))
p_draw_spins = p_draw_chi1*p_draw_chi2*p_draw_cost1*p_draw_cost2

# P_draw for masses and redshift - for our injections, this is analytic so just calculate directly
# (evaluate at detected values, but use injected distribution which here is the best fit GWTC3 
# power law + peak as found in GeneratePopulations/helper_functions.py)
p_draw_masses_redshift = p_astro_masses(m1_det, m2_det, mCut=8)*p_astro_z(z_det, dV_dz=dVdz_det)

# Get rid of events where p_draw==0
p_draw = p_draw_spins*p_draw_masses_redshift
good_idxs = np.where(p_draw != 0)

print(f'number of bad indices = {len(p_draw) - len(p_draw[good_idxs])}')

# Make injectionDict 
injectionDict = {
    'a1':chi1_det[good_idxs].tolist(),
    'a2':chi2_det[good_idxs].tolist(),
    'cost1':cost1_det[good_idxs].tolist(),
    'cost2':cost2_det[good_idxs].tolist(),
    'm1':m1_det[good_idxs].tolist(),
    'm2':m2_det[good_idxs].tolist(),
    'z':z_det[good_idxs].tolist(),
    'dVdz':dVdz_det[good_idxs].tolist(),
    'p_draw_a1a2cost1cost2':p_draw_spins[good_idxs].tolist(),
    'p_draw_m1m2z':p_draw_masses_redshift[good_idxs].tolist()
}

# Save injectionDict in folder where population inference input goes 
with open(f'../../Data/PopulationInferenceInput/injectionDict.json', 'w') as f:
    json.dump(injectionDict, f)
