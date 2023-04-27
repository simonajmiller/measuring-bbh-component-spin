import numpy as np
from scipy.special import erf
from scipy.special import beta
from math import gamma

def draw_initial_walkers_uniform(num_walkers, bounds): 
    
    """
    Function to draw the initial walkers for emcee from a uniform distribution
    
    Parameters
    ----------
    num_walkers : int
        number of walkers
    bounds : tuple
        upper and lower bounds for the uniform distribution
        
    Returns
    -------
    walkers : `numpy.array`
        random array of length num_walkers
    """
    
    upper_bound = bounds[1]
    lower_bound = bounds[0]
    walkers = np.random.random(num_walkers)*(upper_bound-lower_bound)+lower_bound
    
    return walkers

def asym(x):
    
    """
    Asymptotic expansion for error function
    
    Parameters
    ----------
    x : `numpy.array`
        input samples on which to evaluate asymptotic expansion for error function
        
    Returns
    -------
    y : `numpy.array`
        asymptotic expansion for error function evaluated on input samples x
    """
    
    y = -np.exp(-x**2)/np.sqrt(np.pi)/x*(1.-1./(2.*x**2))
    
    return y

def calculate_Gaussian_1D(x, mu, sigma, low, high): 
    
    """
    Function to calculate 1D truncated normalized gaussian
    
    Parameters
    ----------
    x : `numpy.array`
        input samples on which to evaluate truncated gaussian 
    mu : float
        mean of gaussian 
    sigma : float
        width (std. dev.) of gaussian 
    low : float
        lower truncation bound of gaussian
    high : float
        upper truncation bound of gaussian
    
    Returns
    -------
    y : `numpy.array`
        truncated gaussian function evaluated on input samples x
    """
    
    try: # if high and low are single values
        assert high>low, "Higher bound must be greater than lower bound"
    except: # if they are arrays
        assert np.all(np.asarray(high)>np.asarray(low)), "Higher bound must be greater than lower bound"

    sigma2 = sigma**2.0
    a = (low-mu)/np.sqrt(2*sigma2)
    b = (high-mu)/np.sqrt(2*sigma2)
    norm = np.sqrt(sigma2*np.pi/2)*(-erf(a) + erf(b))

    # If difference in error functions produce zero, overwrite with asymptotic expansion
    if np.isscalar(norm):
        if norm==0:
            norm = (np.sqrt(sigma2*np.pi/2)*(-asym(a) + asym(b)))
    elif np.any(norm==0):
        badInds = np.where(norm==0)
        norm[badInds] = (np.sqrt(sigma2*np.pi/2)*(-asym(a) + asym(b)))[badInds]

    # If differences remain zero, then our domain of interest (-1,1) is so many std. deviations
    # from the mean that our parametrization is unphysical. In this case, discount this hyperparameter.
    # This amounts to an additional condition in our hyperprior
    # NaNs occur when norm is infinitesimal, like 1e-322, such that 1/norm is set to inf and the exponential term is zero
    y = (1.0/norm)*np.exp((-1.0*(x-mu)*(x-mu))/(2.*sigma2))
    if np.any(norm==0) or np.any(y!=y):
        return np.zeros(x.size)

    else:
        y[x<low] = 0
        y[x>high] = 0
        return y
    
    
def calculate_Double_Gaussian(x, mu1, sigma1, mu2, sigma2, f, low, high):

    """
    Function to calculate mixture of gaussian centered at 1 and
    isotropic distribution, as used for the cos(theta) distributions.

    Parameters
    ----------
    x : `numpy.array`
        input samples on which to evaluate the model
    mu1 : float
        mean of gaussian 1
    sigma1 : float
        width (std. dev.) of gaussian 1
    mu2 : float
        mean of gaussian 2
    sigma2 : float
        width (std. dev.) of gaussian 2
    f : float
        fraction in  gaussian 1
    low : float
        lower truncation bound of the whole distribution
    high : float
        upper truncation bound of the whole distribution

    Returns
    -------
    y : `numpy.array`
        model evaluated on input samples x
    """
    
    assert f>=0 and f<=1, f'mixing fraction must be between 0 and 1; given f = {f}'

    if type(x) == list:
        x = np.array(x)
        
    # first gaussian
    gaussian1 = f*calculate_Gaussian_1D(x, mu1, sigma1, low, high)

    # second part
    gaussian2 = (1-f)*calculate_Gaussian_1D(x, mu2, sigma2, low, high)

    # combine them and implement bounds
    y = gaussian1 + gaussian2
    y[x<low] = 0
    y[x>high] = 0

    return y
    
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

def calculate_betaDistribution(x, a, b): 
    
    """
    Beta distribution, as used for the spin magnitude distributions.
    See https://en.wikipedia.org/wiki/Beta_distribution.
    
    Parameters
    ----------
    x : `numpy.array`
        input samples on which to evaluate beta distribution
    a : float
        first shape parameter for beta distribution
    b : float
        second chape parameter for beta distribution
    
    Returns
    -------
    y : `numpy.array`
        beta distribution evaluated on input samples x
    """
    
    B = beta(a,b) # from scipy package
    y = np.power(x, a-1)*np.power(1-x, b-1)/B
    
    return y

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

def p_astro_masses(m1, m2, alpha=-3.51, bq=0.96, mMin=5.00, mMax=88.21, lambda_peak=0.033, m0=33.61, sigM=4.72, deltaM=4.88): 
    
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
    
def p_astro_z(z, dVdz, kappa=2.7):
    
    """
    Function to calculate p_astro(z) for a power law model in (1+z)

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
    
    p_z = dVdz*np.power(1.+z,kappa-1.)
    
    return p_z
