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

# Want to generate 10000 draws from the population
target = 10000 

# for mass distribution
lambda_peak=0.033

# Prepare interpolation grid for redshifts 
z_grid = np.linspace(0.,2.,1000)
p_z_grid = p_z(z_grid)
p_z_norm = np.trapz(p_z_grid,z_grid)
p_z_grid = p_z_grid/p_z_norm
z_cdf_grid = np.cumsum(p_z_grid)*(z_grid[1]-z_grid[0])

# Calculate a "horizon" as a function of mass which we cannot see beyond
horizon_component_masses = np.logspace(0.5,2.5,10)
horizon_zs = np.zeros(horizon_component_masses.size)
print("\nHorizon calculation")
print("--------------------")
print("mass\t\t  redshift")
for i,m in enumerate(horizon_component_masses):

    # Select initial trial distance
    DL = 50.

    trial_snr = np.inf
    while trial_snr>6.:
        DL = DL*1.5
        hp, hc = get_fd_waveform(approximant=APPROXIMANT, mass1=m, mass2=m,
                                    spin1z=0.95, spin2z=0.95, 
                                    inclination=0., distance=DL, f_ref=f_ref,
                                    f_lower=f_lower, delta_f=delta_f, f_final=f_upper)
        sqSNR1 = matchedfilter.overlap(hp,hp,psd=psd_H1,low_frequency_cutoff=f_lower,high_frequency_cutoff=f_upper,normalized=False)
        sqSNR2 = matchedfilter.overlap(hp,hp,psd=psd_L1,low_frequency_cutoff=f_lower,high_frequency_cutoff=f_upper,normalized=False)
        sqSNR3 = matchedfilter.overlap(hp,hp,psd=psd_V1,low_frequency_cutoff=f_lower,high_frequency_cutoff=f_upper,normalized=False)
        trial_snr = np.sqrt(sqSNR1+sqSNR2+sqSNR3)
        #print(m,trial_snr,DL,z_at_value(Planck13.luminosity_distance,DL*u.Mpc))

    horizon_zs[i] = z_at_value(Planck13.luminosity_distance,DL*u.Mpc)
    print(m,horizon_zs[i])
    
'''
Generate the populations
'''

fnames = ['population1_highSpinPrecessing', 'population2_mediumSpin', 'population3_lowSpinAligned']

for fname in fnames:

    # Counters for successful draws
    n_det = 0
    n_trials = 0
    n_hopeless = 0
    
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

    print(f"\nGenerating population for {fname}")
    print("-----------------------")
    while n_det<target:

        n_trials += 1

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

        # Compare against precomputed horizons - if there is no hope that this will pass 
        # the snr cut then reject and move on
        z_reject = np.interp((m1+m2)*(1.+z),2.*horizon_component_masses,horizon_zs)
        if z>z_reject:
            n_hopeless += 1
            continue

        # Random s1z, s2z from a truncated normal distribution 
        mu_sz = 0.1
        sigma_sz = 0.15
        szs = np.random.normal(loc=mu_sz, scale=sigma_sz, size=50) # draw from gaussian
        szs = szs[np.where(szs>=-1)] # truncate
        s1z,s2z = szs[np.where(szs<=1)][:2]

        # Only thing that differs between the three pops is the chi1,chi2 dist: 
        if fname=='population1_highSpinPrecessing': 
            # POP 1: HighSpinPrecessing
            # Random spin magnitudes between abs(sz) and 1
            chi1 = draw_uniform(np.abs(s1z), 1)
            chi2 = draw_uniform(np.abs(s2z), 1)

        elif fname=='population2_mediumSpin': 
            # POP 2: MediumSpin
            # Random spin magnitudes normally distributed about s_z with sigma=0.2
            sigma_chi = 0.2
            chi1s = np.random.normal(loc=np.abs(s1z), scale=sigma_chi, size=500)
            chi1s = chi1s[np.where(chi1s<=1)]
            chi1 = np.random.choice(chi1s[np.where(chi1s>=np.abs(s1z))])
            chi2s = np.random.normal(loc=np.abs(s2z), scale=sigma_chi, size=500)
            chi2s = chi2s[np.where(chi2s<=1)]
            chi2 = np.random.choice(chi2s[np.where(chi2s>=np.abs(s2z))])

        elif fname=='population3_lowSpinAligned':
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

        # Generate waveform
        hp, hc = get_fd_waveform(approximant=APPROXIMANT, mass1=m1*(1.+z), mass2=m2*(1.+z),
                                        spin1z=s1z, spin2z=s2z,
                                        spin1x=s1x, spin2x=s2x,
                                        spin1y=s1y, spin2y=s2y,
                                        coa_phase=phase,
                                        inclination=inc, distance=DL, f_ref=f_ref,
                                        f_lower=f_lower, delta_f=delta_f)

        # Project onto detectors
        time = 1126259642.413
        Hp, Hx = H1.antenna_pattern(ra, dec, pol, time)
        Lp, Lx = L1.antenna_pattern(ra, dec, pol, time)
        Vp, Vx = V1.antenna_pattern(ra, dec, pol, time)
        s1 = Hp*hp + Hx*hc
        s2 = Lp*hp + Lx*hc
        s3 = Vp*hp + Vx*hc

        # Compute network SNR
        try:
            sqSNR1 = matchedfilter.overlap(s1,s1,psd=psd_H1,low_frequency_cutoff=f_lower,high_frequency_cutoff=f_upper,normalized=False)
            sqSNR2 = matchedfilter.overlap(s2,s2,psd=psd_L1,low_frequency_cutoff=f_lower,high_frequency_cutoff=f_upper,normalized=False)
            sqSNR3 = matchedfilter.overlap(s3,s3,psd=psd_V1,low_frequency_cutoff=f_lower,high_frequency_cutoff=f_upper,normalized=False)
        except ValueError:
            print('m1=',m1)
            print('m2=',m2)
            print('z=',z)
            print('s1z, s2z, s1x, s2x, s1y, s2y =',[s1z, s2z, s1x, s2x, s1y, s2y])
            print('inc=',inc)
            print('DL=',DL)
            print(len(s1))
            print(len(psd_H1))
            sys.exit()
            break

        snr = np.sqrt(sqSNR1+sqSNR2+sqSNR3)

        # If this injection passes the SNR threshold ... 
        if snr>=10.:

            # Record
            saved_m1s[n_det] = m1
            saved_m2s[n_det] = m2
            saved_s1zs[n_det] = s1z
            saved_s2zs[n_det] = s2z
            saved_s1xs[n_det] = s1x
            saved_s2xs[n_det] = s2x
            saved_s1ys[n_det] = s1y
            saved_s2ys[n_det] = s2y
            saved_zs[n_det] = z
            saved_DLs[n_det] = DL
            saved_incs[n_det] = inc
            saved_ras[n_det] = ra
            saved_decs[n_det] = dec
            saved_phases[n_det] = phase
            saved_pols[n_det] = pol
            saved_snrs[n_det] = snr

            n_det += 1

            print(n_trials,n_det,n_hopeless, end='\r')


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
            'snr':saved_snrs,\
            'seed':np.random.randint(1000000,size=n_det)\
            }

    df = pd.DataFrame(populationDict)
    df.to_json(f'../../Data/InjectedPopulationParameters/{fname}_fullmassrange.json')
