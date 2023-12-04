import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib import style
import os
style.use(os.path.dirname(os.path.realpath(__file__))+'/plotting.mplstyle')
import seaborn as sns

def getBounds(data):

    """
    Helper function to obtain 90% credible bounds from a list of samples
    Invoked by plot_corner to create labels on 1D posteriors

    Parameters
    ----------
    data : list or numpy.array
        1D array of samples

    Returns
    -------
    med : float
        Median of samples
    upperError : float
        Difference between 95th and 50th percentiles of data
    lowerError : float
        Difference between 50th and 5th percentiles of data
    """

    # Transform to a numpy arry
    data = np.array(data)

    # Get median, 5% and 95% quantiles
    med = np.median(data)
    upperLim = np.sort(data)[int(0.95*data.size)]
    lowerLim = np.sort(data)[int(0.05*data.size)]
 
    # Turn quantiles into upper and lower uncertainties
    upperError = upperLim-med
    lowerError = med-lowerLim
    
    return med,upperError,lowerError
    
def plot_corner(fig,plot_data,hist_alpha=0.7,bins=20,labelsize=14,logscale=False,vmax=None,plot_titles=True):

    """
    Helper function to generate corner plots of posterior samples.
    The primary input, `plot_data`, should be a nested dictionary containing data to be plotted.
    Every item in the parent dictionay corresponds to a parameter column, and should possess the following keys:

    * `data`: Dict of datasets of posterior sample values + plotting color in hex code
    * `plot_bounds`: Tuple of min/max values to display on plot
    * `label`: A latex string for axis labeling
    * `true_val`: Optional, true value of underlying population parameter

    Parameters
    ----------
    fig : matplotlib figure object
        Figure object to populate
    plot_data : dict
        Dictionary containing data to plot; see above
    color : str
        Hexcode defining plot color
    hist_alpha : float
        Defines transparency of 1D histograms (optional; default 0.7)
    bins : int
        Defines number of 1D histogram bins and 2D hexbins to use (optional; default 20)
    labelsize : int
        Defines fontsize of axis labels (optional; default 14)
    logscale : bool
        If true, a logarithmic color scale is adopted for 2D posteriors (optional; default False)
    vmax : None or float
        User-specified maximum for 2D colorscale (optional; default None)

    Returns
    -------
    fig : matplotlib figure
        Populated figure
    """
    
    if logscale==True:
        hexscale='log'
    else:
        hexscale=None
    
    # Loop across dimensions that we want to plot
    keys = list(plot_data)    
    ndim = len(keys)
    
    for i,key in enumerate(keys):
       
        # Plot the marginal 1D posterior(s) (i.e. top of a corner plot column)
        ax = fig.add_subplot(ndim,ndim,int(1+(ndim+1)*i))
        
        npops = len(plot_data[key]['data'])
        for n, pop in enumerate(plot_data[key]['data']): 
                        
            posterior = plot_data[key]['data'][pop]['posterior']
            color = plot_data[key]['data'][pop]['color']
        
            ax.hist(posterior,bins=np.linspace(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds'][1],bins),\
                   rasterized=True,color=color,alpha=hist_alpha,density=True,zorder=0)

            ax.hist(posterior,bins=np.linspace(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds'][1],bins),\
                    histtype='step',color='black',density=True,zorder=2)
            
             # Plot the error bar values
            if plot_titles:
                ebar_str = r"${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(*getBounds(posterior))
                ymax = 0.85 + npops*0.26
                ax.text(0.5, ymax - 0.26*n, ebar_str, fontsize=14, transform=ax.transAxes,
                             horizontalalignment='center', verticalalignment='bottom', color=color)
            
        if 'true_val' in plot_data[key].keys(): 
            ax.axvline(plot_data[key]['true_val'], ls='--', color='k')

        ax.grid(True,dashes=(1,3))
        ax.set_xlim(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds'][1])

        # Turn off tick labels if this isn't the first dimension
        ax.set_yticklabels([])

        # If this is the last dimension add an x-axis label
        if i==ndim-1:
            ax.set_xlabel(plot_data[key]['label'],fontsize=labelsize)
            
        # If not the last dimension, loop across other variables and fill in the rest of the column with 2D plots
        else:
            
            ax.set_xticklabels([])
            for j,k in enumerate(keys[i+1:]):
                
                # Make a 2D density plot
                ax = fig.add_subplot(ndim,ndim,int(1+(ndim+1)*i + (j+1)*ndim))
                
                for pop in plot_data[key]['data']: 
                    
                    if pop in plot_data[k]['data'].keys():
                        posterior1 = plot_data[key]['data'][pop]['posterior']
                        posterior2 = plot_data[k]['data'][pop]['posterior']
                        color = plot_data[key]['data'][pop]['color']

                        # Define a linear color map(s)
                        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(1,1,1,0.2),color])
                        
                        # Plot 2d hist
                        ax.hexbin(
                            posterior1,posterior2,
                            cmap=cmap,mincnt=1,gridsize=bins,bins=hexscale,
                            rasterized=True,
                            linewidths=(0,),zorder=0,vmax=vmax,
                            extent=(plot_data[key]['plot_bounds'][0],
                                    plot_data[key]['plot_bounds'][1],
                                    plot_data[k]['plot_bounds'][0],
                                    plot_data[k]['plot_bounds'][1]),
                        )
                        
                        # Plot contours
                        sns.kdeplot(x=posterior1, y=posterior2, 
                                    ax=ax, color=color, 
                                    levels=[0.5, 0.95],
                                    fill=False,
                                    zorder=2)
                    
                if 'true_val' in plot_data[key].keys(): 
                    ax.axvline(plot_data[key]['true_val'], ls='--', color='k')
                if 'true_val' in plot_data[k].keys(): 
                    ax.axhline(plot_data[k]['true_val'], ls='--', color='k')
                
                # Set plot bounds
                ax.set_xlim(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds'][1])
                ax.set_ylim(plot_data[k]['plot_bounds'][0],plot_data[k]['plot_bounds'][1])
                ax.grid(True,dashes=(1,3))
                
                # If still in the first column, add a y-axis label
                if i==0:
                    ax.set_ylabel(plot_data[k]['label'],fontsize=labelsize)
                else:
                    ax.set_yticklabels([])
               
                # If on the last row, add an x-axis label
                if j==ndim-i-2:
                    ax.set_xlabel(plot_data[key]['label'],fontsize=labelsize)
                else:
                    ax.set_xticklabels([])
                    
    return fig


def add_mu_sigma_prior(ax): 
    # Add constraint on mu_chi sigma_chi from prior 
    mu_range = np.linspace(0,1,100)
    constraint1 = np.sqrt((mu_range**2.)*(1-mu_range)/(1+mu_range))
    constraint2 = np.sqrt(mu_range*((1-mu_range)**2.)/(2-mu_range))
    total_constraint = np.minimum(constraint1, constraint2)
    ax.fill_between(mu_range, total_constraint, y2=np.zeros(100)+0.5, alpha=0.1, color='k')