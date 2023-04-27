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


# Want to generate 10000 draws from the population
target = 10000 

# for mass distribution
lambda_peak=0.033

# minimum mass for injections: 
M_MIN = 20

fnames = ['underlying_population1_highSpinPrecessing', 'underlying_population2_mediumSpin', 'underlying_population3_lowSpinAligned']

for fname in fnames:
    
    print(fname)

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
            m1 = inv_cdf_m1_PL(c_m1, mMin=M_MIN)
        else: # in gaussian peak part of distribution
            m1 = inv_cdf_m1_gaussian(c_m1, mMin=M_MIN)

        # Random m2
        c_m2 = np.random.random()
        m2 = inv_cdf_m2(c_m2, m1, mMin=M_MIN)

        # Random redshift
        cz = np.random.random()
        z = np.interp(cz,z_cdf_grid,z_grid)
        DL = Planck13.luminosity_distance(z).to(u.Mpc).value

        # Random s1z, s2z from a truncated normal distribution centered at 1 with sigma=0.15
        mu_sz = 0.1
        sigma_sz = 0.15
        szs = np.random.normal(loc=mu_sz, scale=sigma_sz, size=50) # draw from gaussian
        szs = szs[np.where(szs>=-1)] # truncate
        s1z,s2z = szs[np.where(szs<=1)][:2]

        # Only thing that differs between the three pops is the chi1,chi2 dist: 
        if fname=='underlying_population1_highSpinPrecessing': 
            # POP 1: HighSpinPrecessing
            # Random spin magnitudes between abs(sz) and 1
            chi1 = draw_uniform(np.abs(s1z), 1)
            chi2 = draw_uniform(np.abs(s2z), 1)
            
        elif fname=='underlying_population2_mediumSpin': 
            # POP 2: MediumSpin
            # Random spin magnitudes normally distributed about s_z with sigma=0.2
            sigma_chi = 0.2
            chi1s = np.random.normal(loc=np.abs(s1z), scale=sigma_chi, size=500)
            chi1s = chi1s[np.where(chi1s<=1)]
            chi1 = np.random.choice(chi1s[np.where(chi1s>=np.abs(s1z))])
            chi2s = np.random.normal(loc=np.abs(s2z), scale=sigma_chi, size=500)
            chi2s = chi2s[np.where(chi2s<=1)]
            chi2 = np.random.choice(chi2s[np.where(chi2s>=np.abs(s2z))])
            
        elif fname=='underlying_population3_lowSpinAligned':
            # POP 3: LowSpinAligned
            # Random spin magnitudes normally distributed about s_z with sigma=0.05
            sigma_chi = 0.05
            chi1s = np.random.normal(loc=np.abs(s1z), scale=sigma_chi, size=500)
            chi1s = chi1s[np.where(chi1s<=1)]
            chi1 = np.random.choice(chi1s[np.where(chi1s>=np.abs(s1z))])
            chi2s = np.random.normal(loc=np.abs(s2z), scale=sigma_chi, size=500)
            chi2s = chi2s[np.where(chi2s<=1)]
            chi2 = np.random.choice(chi2s[np.where(chi2s>=np.abs(s2z))])

        # Randomly split the "remaining" spin magnitude between sx and sy by
        # uniformly drawing in phi
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
            'pol':saved_pols      
            }

    df = pd.DataFrame(populationDict)
    df.to_json(f'../../Data/InjectedPopulationParameters/{fname}.json')
