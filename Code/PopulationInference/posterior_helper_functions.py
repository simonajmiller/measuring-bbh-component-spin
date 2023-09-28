import numpy as np
from scipy.special import erf
from scipy.special import beta
from math import gamma
import sys
sys.path.append('/home/simona.miller/measuring-bbh-component-spin/Code/GeneratePopulations/')
from helper_functions import mu_sigma2_to_a_b
from helper_functions import p_astro_masses, smoothing_fxn
from helper_functions import p_z as p_astro_z


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


def chirpmass(m1, m2): 
    q = m2/m1
    Mc = (q/(1.+q)**2)**(3./5.)*(m1+m2)
    return Mc