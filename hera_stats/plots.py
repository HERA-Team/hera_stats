import numpy as np
from . import stats, utils
from pyuvdata import UVData
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_spectra(jkset, fig=None, show_groups=False, with_errors=True,
                 z_method="weightedsum", zlim=5, logscale=True):
    """
    Plots a pair of spectra, their errors, and normalized residuals.

    Parameters
    ----------
    jkset: hera_stats.jkset.JKSet
        The jackknife set to plot. Must have shape[0] == 1.

    fig: matplotlib figure, optional
        Where to plot the spectra. If None, creates figure. Default: None.

    show_groups: boolean, optional
        Whether to label spectra arbitrarily or by the antenna used.
        Default: False.

    with_errors: boolean, optional
        If true, plots errorbars and zscores.

    z_method: string, optional
        The method to use for calculating and displaying zscores. Default: weightedsum.

    zlim: int or float, optional
        Limit of the zscore subplot y axis.

    logscale: boolean, optional
        If true, plots spectra on a logarithmic y axis. Default: True.
    """
    assert jkset.ndim == 1, "Input jkset must have shape[0] == 1."
    spec, er = jkset.spectra, jkset.errs

    if fig is None:
        fig = plt.figure(figsize=(8, 5))

    if with_errors:
        ax = fig.add_axes([0.1, 0.3, 0.8, 0.7])
    else:
        ax = fig.add_subplot(111)

    # Choose labels appropriately
    if show_groups:
        labels = jkset.grps[0]
    else:
        labels = ["Group %i" % i
                  for i in range(len(jkset.grps[0]))]

    if logscale:
        spec = np.abs(spec)

    # Plot spectra with errorbars
    if with_errors == True:
        p_l = [ax.errorbar(jkset.dlys, spec[i], er[i]) for i in range(len(spec))]
    else:
        p_l = ax.plot(jkset.dlys, spec.T)

    # Set ylims based on range of power spectrum
    mx, mn = np.max(spec), np.min(spec)
    if logscale:
        ax.set_yscale("log")
        ax.set_ylim(mn*0.6, mx*10.)
    else:
        whitespace = (mx - mn)/8
        ax.set_ylim(mn - whitespace, mx + whitespace)

    ax.set_ylabel(r"P($\tau$)")
    ax.set_xlabel("Delay (ns)")
    ax.grid(True)

    # Set other details
    if len(p_l) < 10:
        ax.legend(p_l, labels, fontsize=8, loc=1)
    ax.set_title("Power Spectrum")

    if with_errors and len(spec) > 1:
        # Create second plot for z scores
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])

        # plot z scores as scatterplot
        zs = stats.zscores(jkset, z_method=z_method)
        [ax2.scatter(zs.dlys, z, marker="+", color="green") for z in zs.spectra]
        xlims = ax.get_xlim()

        # Plot zero line
        ax2.hlines([0], [-10000], [10000], linestyles="--", alpha=0.6)

        # Calculate yticks.
        zmt = zlim//2*2
        ticks = list(range(-zmt, zmt+1, 2))

        # Reinstate limits and set other paramters
        ax2.set_ylim(-zlim, zlim)
        ax2.set_xlim(xlims[0], xlims[1])
        ax2.set_yticks(ticks)
        ax2.set_xlabel("Delay (ns)")
        ax2.set_ylabel("Z-Score")
        ax2.grid()

def scatter(jkset, ax=None, ylim=None, compare=True, logscale=True):
    """
    Plots a scatterplot of the two groups of data. If sorted, makes it
    clear which group contains abnormalities.

    Parameters
    ----------
    jkset: hera_stats.jkset.JKSet
        The jackknife set to plot. Must have shape[0] == 1.

    ax: matplotlib axis, optional
        If specified, plots on this axis. Default: None.

    ylim: tuple, optional
        The (min, max) y limits on the scatter plot. Default: None.

    compare: boolean, optional
        If False, plots everything in gray. If True, separates colors
        based on group.

    logscale: boolean, optional
        If true, plots points on a logarithmic y axis. Default: True.
    """
    assert jkset.ndim == 1, "Input jkset must have ndim == 1."
    dlys, spectra = jkset.dlys, jkset.spectra

    if logscale: spectra = np.abs(spectra)
    
    if ax is None:
        f = plt.figure(figsize=(8, 5))
        ax = f.add_subplot(111)

    # Get data to plot
    wid = (dlys[1] - dlys[0]) 
    xs = np.hstack([dlys + np.random.uniform(-wid/2, wid/2, len(dlys)) 
                    for i in spectra])
    ys = spectra
    #y2 = np.hstack([sp[1] for sp in spectra])

    x = xs
    y = np.hstack(ys)

    if compare:
        # Set colors
        colors = [["C%i" % (i%10)]*len(dlys) for i in range(len(spectra))]
        colors = np.hstack(colors)

        # Shuffle order so colors are randomly in front or behind others
        inds = list(range(len(x)))
        shuffle = np.random.choice(inds, len(inds)-1, replace=False)

        xrand, yrand, crand = [], [], []
        for i in shuffle:
            xrand += [x[i]]
            yrand += [y[i]]
            crand += [colors[i]]

        x, y, colors = xrand, yrand, crand
    else:
        colors = "black"

    if ylim is None:
        ylim = [np.min(y), np.max(y)]

    # Set alpha based on how many points there are, plot
    alpha = 0.7
    ax.scatter(x, y, color=colors, edgecolors="none", alpha=alpha)

    # Set other data
    ax.set_ylabel("Delay (ns)")
    ax.set_ylabel(r"P($\tau$)")
    if logscale:
        ax.set_yscale("log")
        border = (ylim[0], ylim[1] * 10)
    else:
        extra = (ylim[1] - ylim[0]) / 8
        border = (ylim[0] - extra, ylim[1] + extra)

    ax.set_ylim(border[0], border[1])
    ax.set_title("Scatter Plot of Data")

def hist_2d(jkset, ax=None, ybins=40, display_stats=True,
            vmax=None, normalize=False, ylim=None, logscale=True):
    """
    Plots 2d power spectrum for the data.

    Parameters
    ----------
    jkset: hera_stats.jkset.JKSet
        The jackknife set to plot. Must have shape[0] == 1.

    ax: matplotlib axis, optional
        If specified, plots the heatmap on this axis.

    ybins: int or list, optional
        If int, the number of ybins to use. If list, the entries are used
        as bin locations. Note: "raw" requires logarithmic binning.
        Default: 40

    display_stats: boolean, optional
        If true, plots the average and 1-sigma confidence interval of the
        data. Default: True

    vmax: float or int, optional
        If set, sets the value for the maximum color. Useful if comparing
        data sets. Default: None.

    normalize: boolean, optional
        If true, normalizes along the vertical axis, so that the sum of all
        values in a single delay bin is 1. Default: False

    ylim: tuple, optional
        Will plot histogram acoording to these limits if supplied. Default: None.

    logscale: boolean, optional
        If true, plots histogram on a logarithmic y axis. Default: True.
    """
    assert jkset.ndim == 1, "Input jkset must have ndim == 1."
    dlys, spectra = jkset.dlys, jkset.spectra

    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

    if logscale:
        spectra = np.abs(spectra)
        auto_ylim = (np.log10(np.min(spectra)), np.log10(np.max(spectra)))
    else:
        auto_ylim = (np.min(spectra), np.max(spectra))

    if ylim == None: ylim = auto_ylim

    # Make bins
    if type(ybins) == int:
        ybins = np.linspace(ylim[0], ylim[1], ybins+1)
        if logscale:
            ybins = 10**ybins

    xbins = np.linspace(min(dlys)-1, max(dlys)+1, len(dlys)+1)

    x = np.hstack([dlys]*len(spectra))
    y = np.hstack(spectra)

    if display_stats:
        # Plot average and stdev if requested
        avs = np.average(spectra, axis=0)
        stdevs = np.std(spectra, axis=0)
        av, = ax.plot(dlys, avs, c="black", lw=2)
        sigma, = ax.plot(dlys, avs+stdevs, c="red", ls="--", lw=2)
        ax.plot(dlys, avs-stdevs, c="red", ls="--", lw=2)
        ax.legend([av, sigma], ["Average", "1-Sigma"], loc=1)
        ax.set_ylim(min(ybins), max(ybins))

    # Calculate histogram
    hist, xpts, ypts = np.histogram2d(x, y, bins=[xbins, ybins])

    if normalize:
        hist /= len(spectra)
    if vmax is None:
        vmax = np.max(hist)/2

    # Plot hist
    c = ax.pcolor(xpts, ypts, hist.T, vmax=vmax)
    plt.colorbar(c, ax=ax)

    if logscale:
        ax.set_yscale("log")
    # Set labels
    ax.set_title("Jackknife Data Histogram", fontsize=16)
    ax.set_xlabel("Delay (ns)")


def plot_kstest(jkset, ax=None, cdf=None):
    """
    Plots the Kolmogorov-Smirnov test for each delay mode.

    The KS test is a test of normality, in this case, for a gaussian
    (avg, stdev) as specified by the parameter norm. If the p-value
    is below the KS stat, the null hypothesis (gaussian curve (0, 1))
    is rejected.

    Parameters
    ----------
    jkset: hera_stats.jkset.JKSet
        The jackknife set to plot. Must have shape[0] == 1.

    ax: matplotlib axis, optional
        The axis on which to plot the ks test. Default: None.

    cdf: function, optional
        A scipy.stats cumulative distribution function. If None, uses
        scipy.stats.norm(0, 1).cdf
    """

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

    # Get ks information
    ks, pvals = stats.kstest(jkset, summary=False, cdf=cdf)

    failfrac = 100. * sum(pvals < ks)/ len(ks)

    # Plot it
    p2, = ax.plot(jkset.dlys, pvals)
    p1, = ax.plot(jkset.dlys, ks)

    # Set text
    ax.legend([p1, p2], ["ks-stat", "p-val"], loc=1)
    ax.text(-5100, 1.1, "Fail Fraction: %.1f" % failfrac + "%")
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Statistics")
    ax.set_title("Kolmogorov-Smirnov Test by Delay Bin")
    ax.set_ylim(0, 1.2)
    ax.grid(True)

def plot_anderson(jkset, ax=None):
    """
    Plots the Anderson-Darling test for the normality of each delay mode.

    Confidence levels are plotted as horizontal colored lines. The numbers
    on the left hand side of are the observed rates at which the levels are
    exceeded. These are similar to alpha levels, so if the Anderson Darling
    statistic surpasses the confidence level, it indataates a rejection of
    the null hypothesis (a normal distribution) with a certainty of the
    confidence level exceeded.

    One would expect the fraction of times the null hypothesis is rejected
    to be roughly the same as the confidence level if the distribution is
    normal, so the observed failure rates should match their respective
    confidence levels.

    Parameters
    ----------
    jkset: hera_stats.jkset.JKSet
        The jackknife set to plot. Must have shape[0] == 1.

    ax: matplotlib axis, optional
        The axis on which to plot the Anderson Darling Test. Default: None.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

    # Get anderson statistics
    stat, crit = stats.anderson(jkset, summary=False)

    # Plot them
    p1 = ax.plot(jkset.dlys, stat)
    lp = ax.plot(jkset.dlys, crit)[::-1]
    sigs = ["1%", "2.5%", "5%", "10%", "15%"]

    # Plot significance and failure info
    ax.legend(p1+lp, ["Stat"]+sigs, loc=1)
    fails = [sum(np.array(stat) > c) for c in crit[0]]

    for i in range(5):
        ax.text(-5000, crit[0][i]+0.02, "%.1f" % fails[i]+"%")

    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Statistics")
    ax.set_title("Anderson Darling Test by Delay Mode")
    ax.set_ylim(0, max(crit[0])*1.1)
    ax.grid(True)


def long_waterfall(uvd_list, bl, pol, title=None, cmap='viridis', 
                   starting_lst=[], mode='nsamples', file_type='uvh5', 
                   transform=None):
    """    
    Generates a waterfall plot of flags or nsamples with axis sums from an
    input array.

    Parameters
    ----------
    uvd_list : list of UVData objects or list of str
        List of UVData objects to be stacked and displayed. If a list of 
        strings is specified, each UVData object will be loaded one at a time 
        (reduces peak memory consumption).
    
    bl : int or tuple
        Baseline integer or antenna pair tuple of the baseline to plot.
    
    pol : str or int
        Polarization string or integer.
    
    title : str, optional
        Title of the plot. Default: none.
    
    cmap : str, optional
        Colormap parameter for the waterfall plot. Default: 'viridis'.
        
    starting_lst : list, optional
        list of starting lst to display in the plot
    
    mode : str, optional
        Which array to plot from the UVData objects. Options: 'data', 'flags', 
        'nsamples'. Default: 'nsamples'. 
    
    file_type : str, optional
        If `uvd_list` is passed as a list of strings, specifies the file type 
        of the data files to assume when loading them. Default: 'uvh5'.
    
    transform : function, optional
        Vectorized function to apply to array before plotting, e.g. np.abs. 
        Default: None.
    
    Returns
    -------
    main_waterfall : matplotlib.axes
        Matplotlib Axes instance of the main plot
        
    freq_histogram : matplotlib.axes
        Matplotlib Axes instance of the sum across times
        
    time_histogram : matplotlib.axes
        Matplotlib Axes instance of the sum across freqs
        
    data : numpy.ndarray
        A copy of the stacked_array output that is being displayed
    """
    arr_list = []
    for _uvd in uvd_list:
        
        # Try to load UVData from file
        if isinstance(_uvd, str):
            uvd = UVData()
            uvd.read(_uvd, file_type=file_type)
        elif isinstance(_uvd, UVData):
            # Already loaded into UVData object
            uvd = _uvd
        else:
            raise TypeError("uvd_list must contain either filename strings or "
                            "UVData objects")
    
        # Construct key to access data
        if isinstance(bl, (int, np.int)):
            bl = uvd.baseline_to_antnums(bl)
        key = (bl[0], bl[1], pol)
    
        # Get requested data
        if mode == 'data':
            arr_list.append(uvd.get_data(key))
        elif mode == 'flags':
            arr_list.append(uvd.get_flags(key))
        elif mode == 'nsamples':
            arr_list.append(uvd.get_nsamples(key))
        else:
            raise ValueError("mode '%s' not recognized." % mode)
    
    # Stack arrays into one big array
    data = utils.stacked_array(arr_list)
    
    # Apply transform
    if transform is not None:
        data = transform(data)
    
    # Set up the figure and grid
    fig = plt.figure()
    fig.suptitle(title, fontsize=30, horizontalalignment='center')
    grid = gridspec.GridSpec(ncols=10, nrows=15)
    
    # Create main components of figure
    main_waterfall = fig.add_subplot(grid[0:14, 0:8])
    freq_histogram = fig.add_subplot(grid[14:15, 0:8], sharex=main_waterfall)
    time_histogram = fig.add_subplot(grid[0:14, 8:10], sharey=main_waterfall)
    
    # Set sizes
    fig.set_size_inches(20, 80)
    grid.tight_layout(fig)
    counter = data.shape[0] // 60
    
    # Waterfall plot
    main_waterfall.imshow(data, aspect='auto', cmap=cmap, 
                          interpolation='none')
    main_waterfall.set_ylabel('Integration Number')
    main_waterfall.set_yticks(np.arange(0, counter*60 + 1, 30))
    main_waterfall.set_ylim(60*(counter+1), 0)
    
    # Red lines separating files
    for i in range(counter+1):
        main_waterfall.plot(np.arange(data.shape[1]),
                            60*i*np.ones(data.shape[1]), '-r')
    for i in range(len(starting_lst)):
        if not isinstance(starting_lst[i], str):
            raise TypeError("starting_lst must be a list of strings")
    
    # Add text of filenames
    if len(starting_lst) > 0:
        for i in range(counter):
            short_name = 'first\nintegration LST:\n'+starting_lst[i]
            plt.text(-20, 26 + i*60, short_name, rotation=-90, size='small',
                     horizontalalignment='center')
    main_waterfall.set_xlim(0, data.shape[1])
    
    # Frequency sum plot
    counts_freq = np.sum(data, axis=0)
    max_counts_freq = max(np.amax(counts_freq), data.shape[0])
    normalized_freq = 100 * counts_freq/max_counts_freq
    freq_histogram.set_xticks(np.arange(0, data.shape[1], 50))
    freq_histogram.set_yticks(np.arange(0, 101, 5))
    freq_histogram.set_xlabel('Channel Number (Frequency)')
    freq_histogram.set_ylabel('Occupancy %')
    freq_histogram.grid()
    freq_histogram.plot(np.arange(0, data.shape[1]), normalized_freq, 'r-')
    
    # Time sum plot
    counts_times = np.sum(data, axis=1)
    max_counts_times = max(np.amax(counts_times), data.shape[1])
    normalized_times = 100 * counts_times/max_counts_times
    time_histogram.plot(normalized_times, np.arange(data.shape[0]), 'k-',
                        label='all channels')
    time_histogram.set_xticks(np.arange(0, 101, 10))
    time_histogram.set_xlabel('Flag %')
    time_histogram.autoscale(False)
    time_histogram.grid()
    
    # Returning the axes
    return main_waterfall, freq_histogram, time_histogram, data

