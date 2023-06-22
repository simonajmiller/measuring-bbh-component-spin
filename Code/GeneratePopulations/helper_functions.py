import numpy as np
from astropy.cosmology import Planck13,z_at_value
import astropy.units as u
from scipy.special import erf, erfinv


def draw_uniform(low, high, num=1): 
    """
    Generate draw from uniform distribution between low and high
    
    Parameters
    ----------
    low : float 
        lower bound of uniform distribution 
    high : float 
        upper bound of uniform distribution 
    num : int 
        number of samples to draw
        
    Returns
    -------
    numpy array of length num with random draws from the uniform dist
    or float if num==1
    """
    draws = np.random.rand(num)*(high-low) + low
    if num==1: 
        return draws[0]
    else: 
        return draws

def inv_cdf_m1_PL(c, alpha=-3.51, mMin=6.00, mMax=88.21): 
    """
    Function to calculate the inverse cdf for the power law part of the power law +
    peak mass model from gwtc-3
    
    Parameters
    ----------
    c : `numpy.array`
        cdf samples
    alpha : float
        Spectral index for the power-law of the primary mass distribution
    mMin : float
        Minimum mass of the power-law component of the primary mass distribution
    mMax : float
        Maximum mass of the power-law component of the primary mass distribution
    
    Returns
    -------
    inverse cdf evaluated at c
    """
    tmp = c*(np.power(mMax, 1+alpha) - np.power(mMin, 1+alpha)) + np.power(mMin, 1+alpha)
    return np.power(tmp, 1/(1+alpha))


def inv_cdf_m1_gaussian(c, m0=33.61, sigM=4.72, mMin=6.00, mMax=88.21): 
    """
    Function to calculate the inverse cdf for the gaussian peak part of the power law +
    peak mass model from gwtc-3
    
    Parameters
    ----------
    c : `numpy.array`
        cdf samples
    m0 : float
        Mean of the Gaussian component in the primary mass distribution
    sigM : float
        Width of the Gaussian component in the primary mass distribution
    mMin : float
        Minimum mass of the power-law component of the primary mass distribution
    mMax : float
        Maximum mass of the power-law component of the primary mass distribution
    
    Returns
    -------
    inverse cdf evaluated at c
    """
    erfa = erf((mMin - m0)/(np.sqrt(2)*sigM))
    erfb = erf((mMax - m0)/(np.sqrt(2)*sigM))
    tmp = erfinv(erfa + c*(erfb - erfa))
    return m0 + np.sqrt(2)*sigM*tmp


def inv_cdf_m2(c, m1, bq=0.96, mMin=6.00): 
    """
    Function to calculate the inverse cdf for secondary mass under a power
    law
    
    Parameters
    ----------
    c : `numpy.array`
        cdf samples
    m1 : `numpy.array`
        primary mass samples
    bq : float
        Spectral index for the power-law of the mass ratio distribution
    mMin : float
        Minimum mass of the power-law component of the primary mass distribution
   
    Returns
    -------
    inverse cdf evaluated at c
    """
    tmp = c*(np.power(m1, 1+bq) - np.power(mMin, 1+bq)) + np.power(mMin, 1+bq)
    return np.power(tmp, 1/(1+bq))

    
def dVdz(z):
    """
    Function to calculate the derivative of comoving volume with respect to redshift 
    for a given redshift
    
    Parameters
    ----------
    z : `numpy.array`
        redshift samples
        
    Returns
    -------
    dV_dz : `numpy.array`
        derivative of comoving volume with respect to redshift at input redshift 
    """
    dV_dz = 4.*np.pi*Planck13.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value
    return dV_dz


def p_z(z, dV_dz=None, kappa=2.7):
    """
    Function to calculate p(z) for a power law model in (1+z)

    Parameters
    ----------
    z : `numpy.array`
        redshift samples
    dVdz : `numpy.array`
        d(comoving volume)/dz samples
    
    Returns
    -------
    p_z : `numpy.array`
        p_astro(z) evaluated at the input samples
    """
    if dV_dz is None:
        dV_dz = dVdz(z)
        
    p_z = dV_dz*np.power(1.+z,kappa-1.)
    
    return p_z


def draw_xy_spins(chi,sz):

    """
    Function to randomly calculate the x- and y-components of the spin 
    based off the spin magnitude and z-component

    Parameters
    ----------
    chi : float
        spin magnitude
    sz : float
        z-component of spin
    
    Returns
    -------
    sx : float
        x-component of spin
    sy : float
        y-component of spin
    """
    
    # calculate in plane spin component from sz and chi
    s_perp = np.sqrt(chi**2 - sz**2)
    
    # draw a random phi uniformly btwn 0 to 2pi
    phi = draw_uniform(0, 2*np.pi)
    
    # calculate sx and sy from s_perp and phi
    sx = s_perp*np.cos(phi)
    sy = s_perp*np.sin(phi)
        
    return sx,sy 

    
def smoothing_fxn(m, deltaM): 
    
    """
    Smoothing function that goes into the p_astro(m1,m2) calculation for the power law + peak mass model.
    See eqn. B5 in https://arxiv.org/pdf/2111.03634.pdf
    
    Parameters
    ----------
    m : `numpy.array`
        mass samples to calculate smoothing over
    deltaM : float
        Range of mass tapering at the lower end of the mass distribution
    
    Returns
    -------
    S : `numpy.array`
        the smoothing function evaluated at the input samples m
    """
    
    f = np.exp(deltaM/m + deltaM/(m-deltaM))
    S = 1/(f+1)
    
    return S

def p_astro_masses(m1, m2, alpha=-3.51, bq=0.96, mMin=6.00, mMax=88.21, lambda_peak=0.033, m0=33.61, sigM=4.72, deltaM=4.88): 
    
    """
    Function to calculate for p_astro(m1,m2) for the power law + peak mass model. 
    See table VI in https://arxiv.org/pdf/2111.03634.pdf
    
    Default parameters are those corresponding to the median values reported in 
    https://arxiv.org/pdf/2111.03634.pdf
    
    Parameters
    ----------
    m1 : `numpy.array`
        primary mass samples
    m2 : `numpy.array`
        secondary mass  samples
    alpha : float
        Spectral index for the power-law of the primary mass distribution
    bq : float
        Spectral index for the power-law of the mass ratio distribution
    mMin : float
        Minimum mass of the power-law component of the primary mass distribution
    mMax : float
        Maximum mass of the power-law component of the primary mass distribution
    lambda_peak : float
        Fraction of BBH systems in the Gaussian component
    m0 : float
        Mean of the Gaussian component in the primary mass distribution
    sigM : float
        Width of the Gaussian component in the primary mass distribution
    deltaM : float
        Range of mass tapering at the lower end of the mass distribution
    
    Returns
    -------
    p_masses : `numpy.array`
        the power law + peak mass model evaluated at the input samples m1 and m2
    """
    
    # p(m1):
    # power law for m1:
    p_m1_pl = (1.+alpha)*m1**alpha/(mMax**(1.+alpha) - mMin**(1.+alpha))
    p_m1_pl[m1>mMax] = 0.
    # gaussian peak
    p_m1_peak = np.exp(-0.5*(m1-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
    p_m1 = lambda_peak*p_m1_peak + (1.-lambda_peak)*p_m1_pl
    # smoothing fxn 
    p_m1[m1<mMin+deltaM] = p_m1[m1<mMin+deltaM]*smoothing_fxn(m1[m1<mMin+deltaM]-mMin,deltaM)
    
    # p(m2):
    # power law for m2 conditional on m1:
    p_m2 = (1.+bq)*np.power(m2,bq)/(np.power(m1,1.+bq)-mMin**(1.+bq))
    p_m2[m2<mMin]=0
    
    p_masses = p_m1*p_m2
    
    return p_masses
    
    
def mu_sigma2_to_a_b(mu, sigma2): 
    
    """
    Function to transform between the mean and variance of a beta distribution 
    to the shape parameters a and b.
    See https://en.wikipedia.org/wiki/Beta_distribution.
    
    Parameters
    ----------
    x : `numpy.array`
        input samples on which to evaluate beta distribution
    mu : float
        mean of the beta distributoin
    sigma2 : float
        variance of the beta distribution
    
    Returns
    -------
    a,b : floats
        shape parameters of the beta distribution
    """
    
    a = (mu**2.)*(1-mu)/sigma2 - mu
    b = mu*((1-mu)**2.)/sigma2 + mu - 1
    
    return a,b



def p_astro_m1(m1, alpha=-3.51, mMin=6.00, mMax=88.21, lambda_peak=0.033, m0=33.61, sigM=4.72, deltaM=4.88): 
    
    # power law for m1:
    p_m1_pl = (1.+alpha)*m1**alpha/(mMax**(1.+alpha) - mMin**(1.+alpha))
    p_m1_pl[m1>mMax] = 0.
    
    # gaussian peak
    p_m1_peak = np.exp(-0.5*(m1-m0)**2./sigM**2)/np.sqrt(2.*np.pi*sigM**2.)
    p_m1 = lambda_peak*p_m1_peak + (1.-lambda_peak)*p_m1_pl
    
    # smoothing fxn 
    p_m1[m1<mMin+deltaM] = p_m1[m1<mMin+deltaM]*smoothing_fxn(m1[m1<mMin+deltaM]-mMin,deltaM)
    
    p_m1[m1<mMin] = 0.
    
    return p_m1