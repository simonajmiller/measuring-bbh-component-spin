import numpy as np
import bilby
import argparse
import sys
import pandas as pd
from bilby.core.prior.base import Constraint
from lalsimulation import SimInspiralTransformPrecessingWvf2PE
from lal import GreenwichMeanSiderealTime

# Parser to load args from commandline
parser = argparse.ArgumentParser()
parser.add_argument('-json',help='Json file with population instantiation')
parser.add_argument('-job',help='Job number',type=int)
parser.add_argument('-outdir',help="Output directory")
args = parser.parse_args()

# Directory 
directory = '/home/simona.miller/measuring-bbh-component-spin/'

# Specify the output directory and the name of the simulation.
label = "job_{0:05d}".format(args.job)
bilby.core.utils.setup_logger(outdir=args.outdir, label=label)

# Load dataframe and select injection
injections = pd.read_json(args.json)
injections.sort_index(inplace=True)
injection = injections.loc[args.job]

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(int(injection.seed))

# Reference frequency and phase
fRef = 20.
phiRef = injection.phase

# Source frame -> detector frame masses
m1_det_inj = injection.m1*(1.+injection.z)
m2_det_inj = injection.m2*(1.+injection.z)

# Convert spin parameters from components --> angles and magnitudes using the  
# lalsimulation function XLALSimInspiralTransformPrecessingWvf2PE()
theta_jn_inj, phi_jl_inj, tilt_1_inj, tilt_2_inj, phi_12_inj, a1_inj, a2_inj = SimInspiralTransformPrecessingWvf2PE(
    injection.inc, 
    injection.s1x, injection.s1y, injection.s1z, 
    injection.s2x, injection.s2y, injection.s2z, 
    m1_det_inj, m2_det_inj, 
    fRef, phiRef
)

# Convert from ra and rec to zenith and azimuth 
t_geocenter = 1126259642.413 # hardcode geocenter time
gmst = GreenwichMeanSiderealTime(t_geocenter)
zenith_inj, azimuth_inj = bilby.core.utils.conversion.ra_dec_to_theta_phi(injection.ra, injection.dec, gmst)

# Make dictionary of BBH parameters that includes all of the different waveform parameters, including masses and spins of both black holes
injection_parameters = dict(
    mass_1=m1_det_inj,\
    mass_2=m2_det_inj,\
    a_1=a1_inj,\
    a_2=a2_inj,\
    tilt_1=tilt_1_inj,\
    tilt_2=tilt_2_inj,\
    phi_12=phi_12_inj,\
    phi_jl=phi_jl_inj,
    luminosity_distance=injection.Dl,\
    theta_jn=theta_jn_inj,\
    psi=injection.pol,\
    phase=phiRef,\
    geocent_time=t_geocenter,\
    zenith=zenith_inj,\
    azimuth=azimuth_inj,\
    ra=injection.ra, \
    dec=injection.dec
)

print(injection_parameters)

# Sampling frequency and mininmum frequency of the data segment that we're going to inject the signal into
sampling_frequency = 2048.
fMin = 15.

# There is a downsampling and filters applied so the data aren't actually free of aliasing/corruption to the nyquist limit
fMax = sampling_frequency/2 * 0.9 

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant='IMRPhenomXPHM',\
    reference_frequency=fRef,\
    minimum_frequency=fMin, \
    maximum_frequency=fMax
)

# Set up interferometers.  In this case we'll use two interferometers (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to
# their design sensitivity
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
ifos[0].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=directory+"Code/GeneratePopulations/aligo_O3actual_H1.txt")
ifos[1].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=directory+"Code/GeneratePopulations/aligo_O3actual_L1.txt")
ifos[2].power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=directory+"Code/GeneratePopulations/avirgo_O3actual.txt")

# Create the waveform_generator using a LAL BinaryBlackHole source function the generator will convert all the parameters and inject the signal into the
# into the ifos. Need to inject the signal into the smallest possible time window: 4 seconds, 8 seconds, or 16 seconds. 
for duration in [4,8,16]: 
    
    try: 
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, 
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments
        )
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, 
            duration=duration,
            start_time=injection_parameters['geocent_time'] - (duration-2) # canonically, geocenter time is set to be 2 sec from the end of the segment
        )
        ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)
        
        # if this duration works, exit out of the for loop
        duration_used = duration
        break
    
    except Exception as e: 
        print(e) 
          
print('duration:',duration_used)

sys.exit()
        
# For this analysis, we implement most of the standard precessing BBH priors defined in the prior.prior file ...
priors = bilby.gw.prior.BBHPriorDict(directory+"Code/IndividualInference/prior.prior")

# ... except for the definition of the time prior, which is defined as uniform about the injected value ...
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,                                             
    maximum=injection_parameters['geocent_time'] + 0.1,                                                  
    name='geocent_time',                                                   
    latex_label='$t_c$',                                                   
    unit='$s$'
)

# ... and we constrain chirp mass to be +/- 10 solar masses about the injected value.
inj_m1 = injection_parameters['mass_1']
inj_m2 = injection_parameters['mass_2']
inj_chirpmass = np.power(inj_m1*inj_m2, 3./5)/np.power(inj_m1+inj_m2, 1./5)
minChirpMass = max(2, inj_chirpmass-10)
maxChirpMass = min(200, inj_chirpmass+10)
chirp_mass = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=minChirpMass, maximum=maxChirpMass)               


# Initialize the likelihood by passing in the interferometer data (ifos) and
# the waveoform generator, as well the priors.
# For precessing signals, we want
#    * phase marginalization turned off
#    * time marginalization turned on, but sample in azimuth and zenith instead of
#      ra and dec; this is done by setting reference_frame='H1L1'
#    * distance marginalization turned on (this is always safe)
likelihood_kwargs = dict(
    interferometers=ifos, 
        waveform_generator=waveform_generator, 
        priors=priors, 
        reference_frame="H1L1",
        distance_marginalization=True, 
        phase_marginalization=False, 
        time_marginalization=True
)
# The likelihood we use depends on the duration of the signal 
# (see https://git.ligo.org/pe/O4/o4a-rota/-/wikis/Samplers) 
if duration_used==4 or duration_used==8:
    likelihood = bilby.gw.GravitationalWaveTransient(**likelihood_kwargs)
else: 
    likelihood = bilby.gw.MBGravitationalWaveTransient(**likelihood_kwargs)

# Set up properties for sampler
sampler_kwargs = dict(
    sampler='dynesty',
    nlive=1000, 
    naccept=60, 
    sample="acceptance-walk", 
    npool=8, 
    request_cpus=8, 
    nparallel=2
)
if duration_used==16: 
    sample_kwargs['min_ncall'] = 1000000

# Run sampler
result = bilby.run_sampler(
    **sampler_kwargs,
    likelihood=likelihood, 
    priors=priors, 
    injection_parameters=injection_parameters, 
    outdir=args.outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters
)