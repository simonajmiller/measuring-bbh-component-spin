import numpy as np
from pycbc.filter import matchedfilter
from pycbc.waveform import get_fd_waveform
from pycbc.detector import Detector
from pycbc.psd import analytical
from pycbc.psd import read
from astropy.cosmology import Planck13,z_at_value
from scipy.special import erf,erfinv
import astropy.units as u
import sys
import pandas as pd
from helper_functions import *

'''
Preparation steps 
'''
# Waveform approximant to use 
APPROXIMANT = "IMRPhenomXPHM"

# Low and high freq cutoffs for everything 
f_lower = 15
f_upper = 4096

# Reference freq and delta_f for everything
delta_f = 1.
f_ref = 20.

# Prepare detector object
H1 = Detector("H1")
L1 = Detector("L1")
V1 = Detector("V1")
psd_H1 = read.from_txt('./aligo_O3actual_H1.txt',f_upper,delta_f,10,is_asd_file=True)
psd_L1 = read.from_txt('./aligo_O3actual_L1.txt',f_upper,delta_f,10,is_asd_file=True)
psd_V1 = read.from_txt('./avirgo_O3actual.txt',f_upper,delta_f,10,is_asd_file=True)

# Want to generate 50000 draws from the population
target = 50000 

# for mass distribution
lambda_peak=0.033

# Arrays to hold injection values
saved_m1s = np.zeros(target)
saved_m2s = np.zeros(target)
saved_s1zs = np.zeros(target)
saved_s2zs = np.zeros(target)
saved_s1xs = np.zeros(target)
saved_s2xs = np.zeros(target)
saved_s1ys = np.zeros(target)
saved_s2ys = np.zeros(target)
saved_zs = np.zeros(target)
saved_DLs = np.zeros(target)
saved_incs = np.zeros(target)
saved_ras = np.zeros(target)
saved_decs = np.zeros(target)
saved_phases = np.zeros(target)
saved_pols = np.zeros(target)
saved_snrs = np.zeros(target)

# Prepare interpolation grid for redshifts 
z_grid = np.linspace(0.,2.,1000)
p_z_grid = p_z(z_grid)
p_z_norm = np.trapz(p_z_grid,z_grid)
p_z_grid = p_z_grid/p_z_norm
z_cdf_grid = np.cumsum(p_z_grid)*(z_grid[1]-z_grid[0])

for i in range(target):
    
    # Random primary mass
    c_m1 = np.random.random()
    if np.random.rand() > lambda_peak: # in power law part of distribution 
        m1 = inv_cdf_m1_PL(c_m1)
    else: # in gaussian peak part of distribution
        m1 = inv_cdf_m1_gaussian(c_m1)

    # Random m2
    c_m2 = np.random.random()
    m2 = inv_cdf_m2(c_m2, m1)

    # Random redshift
    cz = np.random.random()
    z = np.interp(cz,z_cdf_grid,z_grid)
    DL = Planck13.luminosity_distance(z).to(u.Mpc).value

    # Random spin magnitudes between 0 and 1
    chi1 = draw_uniform(0, 1)
    chi2 = draw_uniform(0, 1)
    
    # Random cos(tilt) between -1 and 1 (isotropic)
    cost1 = draw_uniform(-1,1)
    cost2 = draw_uniform(-1,1)
    
    # From chi and cos(t), calculate sz 
    s1z = chi1*cost1
    s2z = chi2*cost2
    
    # Randomly split the "remaining" spin magnitude between sx and sy by
    # uniformly drawing in phi_12
    s1x,s1y = draw_xy_spins(chi1,s1z)
    s2x,s2y = draw_xy_spins(chi2,s2z)

    # Extrinsic parameters
    ra = draw_uniform(0, 2*np.pi)
    dec = np.arccos(draw_uniform(-1,1)) + np.pi/2.
    pol =  draw_uniform(0, 2*np.pi)
    inc = np.arccos(draw_uniform(-1,1))
    phase = draw_uniform(0, 2*np.pi)

    # Record
    saved_m1s[i] = m1
    saved_m2s[i] = m2
    saved_s1zs[i] = s1z
    saved_s2zs[i] = s2z
    saved_s1xs[i] = s1x
    saved_s2xs[i] = s2x
    saved_s1ys[i] = s1y
    saved_s2ys[i] = s2y
    saved_zs[i] = z
    saved_DLs[i] = DL
    saved_incs[i] = inc
    saved_ras[i] = ra
    saved_decs[i] = dec   
    saved_phases[i] = phase
    saved_pols[i] = pol
        
# Save as dictionary -> json
populationDict = {\
        'm1':saved_m1s,\
        'm2':saved_m2s,\
        's1z':saved_s1zs,\
        's2z':saved_s2zs,\
        's1x':saved_s1xs,\
        's2x':saved_s2xs,\
        's1y':saved_s1ys,\
        's2y':saved_s2ys,\
        'z':saved_zs,\
        'Dl':saved_DLs,\
        'ra':saved_ras,\
        'dec':saved_decs,\
        'inc':saved_incs,\
        'phase':saved_phases,\
        'pol':saved_pols,
        }

df = pd.DataFrame(populationDict)
df.to_json('../../Data/InjectedPopulationParameters/underlying_flat_pop_full_mass_range_for_injection_dict.json')
