
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
        The method to use for calculating and displaying zscores. Default: 
        weightedsum.

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


def kstest(jkset, ax=None, cdf=None):
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


def anderson(jkset, ax=None):
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


def weighted_average(jkset, axis=0, error_field='bs_std'):
    """
    Calculates the variance-weighted average of the spectra over a specific 
    axis. The averages and errors are calculated as follows (for spectra x_i 
    with errors err_i):
        
        w_i = 1 / (err_i)^2
        mean = \sum_i(w_i x_i) / \sum_i(w_i)
        var_mean = 1. / sum_i(w_i)
        std = sqrt(var_mean * len(x))

    Thus, std is not the uncertainty on the average but the standard deviation
    of the distribution of x.

    Parameters
    ----------
    jkset: hera_stats.jkset.JKSet
        JKSet with which to perform the weighted sum.

    axis: int, 0 or 1, optional
        Axis along which to do the weighted sum. Default: 0.

    Returns
    -------
    jkset_avg: 1D JKSet
        The average jkset, with errors that describe the standard deviation
        of the samples given.
    """
    jk = copy.deepcopy(jkset)

    if isinstance(axis, int): axis = (axis,)
    assert isinstance(jkset, jkset_lib.JKSet), \
        "Expected jkset to be hera_stats.jkset.JKSet instance."
    assert all([ax < jk.ndim for ax in axis]), \
        "Axis %s was specified but jkset only has %d axes." % (axis, jk.ndim)
    
    print("jk.spectra:", jk.spectra.shape)
    
    # Do weighted sum calculation
    w = 1. / jk.errs**2. # weights
    var_avg = 1. / np.sum(w, axis=axis) # variance of weighted mean
    avg = var_avg * np.sum(w * jk.spectra, axis=axis) # weighted mean
    std = np.sqrt(var_avg * len(jk.spectra)) 

    # Average/sum metadata, as appropriate
    nsamp = np.sum(jk.nsamples, axis=axis)
    integrations = np.average(jk.integrations, axis=axis)
    times = np.average(jk.times, axis=axis)
    
    # If the spectra were averaged down to a single spectrum, of shape (Ndlys,), 
    # expand to (1, Ndlys).
    """
    if len(av.shape[:-1]) == 0:
        targ_shape = (1, )
        av = av[None]
        std = std[None]
        nsamp = nsamp[None]
        integrations = integrations[None]
        times = times[None]
    """
    
    # Slice jk to match shape of avg and std
    key = [0] * jk.ndim
    for i in range(jk.ndim):
        if i not in axis:
            key[i] = slice(None, None, 1)
    jkav = jk[tuple(key)]

    # Set average and error of jk
    jkav.set_data(avg, std, error_field=error_field)

    # Set UVPSpec attrs
    for i, uvp in enumerate(jkav._uvp_list):
        uvp.integration_array[0] = integrations[i][None]
        uvp.time_avg_array[0] = times[i][None]
        uvp.time_1_array[0] = times[i][None]
        uvp.time_2_array[0] = times[i][None]
        uvp.nsample_array[0] = nsamp[i][None]
        uvp.labels = np.array(["Weighted Sum"])

    return jkav


def weightedsum(jkset, axis=0, error_field='bs_std'):
    return weighted_average(jkset, axis, error_field)


def zscores(jkset, z_method="weightedsum", axis=0, error_field='bs_std'):
    """
    Calculates the z scores for a JKSet along a specified axis. This
    returns another JKSet object, which has the zscore data and errors
    equal to 0.

    Parameters
    ----------
    jkset: hera_stats.jkset.JKSet
        The jackknife set to use for calculating zscores.

    z_method: string, optional
        Method used to calculate z-scores. 
        Options: ["varsum", "weightedsum"]. Default: varsum.
        
        "varsum" works only with two
        jackknife groups, and the standard deviation is calculated by:
        
        zscore = (x1 - x2) / sqrt(err1**2 + err2**2)
        
        "weightedsum" works for any number of spectra and
        calculates the weighted mean with stats.weightedsum, then
        calculates zscores like this:
        
        zscore = (x1 - avg) / avg_err

    Returns
    -------
    zs: list
        Returns zscores for every spectrum pair given.
    """
    assert isinstance(jkset, jkset_lib.JKSet), "Expected jkset to be hera_stats.jkset.JKSet instance."

    if isinstance(axis, int): axis = (axis, )
    assert all([ax < jkset.ndim for ax in axis]), "Axes %s was specified butn jkset has only axes <= %i." % (axis, jkset.ndim - 1)

    spectra = jkset.spectra
    errs = jkset.errs

    jkout = copy.deepcopy(jkset)

    spectra = jkset.spectra
    errs = jkset.errs
    
    jkout = copy.deepcopy(jkset)

    # Or Calculate z scores with weighted sum
    if z_method == "weightedsum":
        # Calculate weighted average and standard deviation.
        aerrs = 1. / np.sum(errs ** -2, axis=axis)
        av = aerrs * np.sum(spectra * errs**-2, axis=axis)
        N = [spectra.shape[ax] for ax in axis]
        std = np.sqrt(aerrs * reduce(lambda x,y: x*y, N))
        if len(axis) == 1:
            av = np.expand_dims(av, axis[0])
            std = np.expand_dims(std, axis[0])

        z = (spectra - av)/(std)
        jkout.set_data(z, 0*z, error_field=error_field)

    # Calculate z scores using sum of variances
    elif z_method == "varsum":
        assert len(axis) == 1, "Varsum can only work over one axis."
        assert jkout.shape[axis[0]] == 2, "Varsum can only take axes of length 2, got {}.".format(jkout.shape[axis[0]])

        # Make keys for 2 datasets, slicing into 2 groups along specified axes
        key1 = [slice(None, None, 1)] * jkout.ndim
        key1[axis[0]] = 0
        key2 = copy.deepcopy(key1)
        key2[axis[0]] = 1
        key1 = tuple(key1); key2 = tuple(key2)

        # Calculate combined error and sum of zscores
        comberr = np.sqrt(errs[key1]**2 + errs[key2]**2).clip(10**-10, np.inf)
        z = ((spectra[key1] - spectra[key2])/comberr)

        # Use weightedsum to shrink jkset to size, then replace data
        jkout = weightedsum(jkout, axis=axis, error_field=error_field)
        print(jkout.shape, z.shape)
        jkout.set_data(z, 0*z, error_field=error_field)
    else:
        raise NameError("Z-score calculation method not recognized")

    if axis == 1:
        jkout = jkout.T()

    jkout.jktype = "zscore_%s" % z_method
    return jkout


def kstest(jkset, summary=False, cdf=None, verbose=False):
    """
    Does a kstest on spectra in jkset. The input jkset must have shape[0] == 1,
    As the kstest is run by delay bin along the 1st axis. Indexing by row or column
    of jkset will do the trick.

    The KS test is a test of normality, in this case, for a gaussian (avg,
    stdev) as specified by the parameter norm. If the p-value is below the
    KS stat, the null hypothesis (gaussian curve (0, 1)) is rejected.

    Parameters
    ----------
    jkset: hera_stats.jkset.JKSet, ndim=1
        The jackknife set to use for running the ks test.

    summary: boolean, optional
        If true, returns the overall failure rate. Otherwise, returns
        the ks and p-value spectra.

    cdf: function, optional
        If a test is needed against a cumulative distribution function that
        is not a (0, 1) gaussian, then a different cdf can be supplied as a
        scipy stats function. If None, automatically chooses
        scipy.stats.norm(0, 1).cdf. Default: None.

    verbose: boolean, optional
        If true, prints information for every delay mode instead of
        summary. Default: False

    Returns
    -------
    ks: ndarray
        If summary == False, returns all of the ks values as a spectrum over delay modes.

    pval: ndarray
        If summary == False, returns all of the p values as a spectrum over delay modes.

    failfrac: float
        The fraction of delay modes that fail the KS test, if summary == True.
    """
    assert isinstance(jkset, jkset_lib.JKSet), "Expected jkset to be hera_stats.jkset.JKSet instance."
    assert jkset.ndim == 1, "Input jkset must have 1 dimension."

    if cdf == None:
        cdf = spstats.norm(0, 1).cdf

    spectra = jkset.spectra

    ks_l, pval_l = [], []
    fails = 0.
    for i, col in enumerate(spectra.T):
        [ks, pval] = spstats.kstest(col, cdf)
        # Save result
        ks_l += [ks]
        pval_l += [pval]
        isfailed = int(pval < ks)
        fails += isfailed

        if verbose:
            st = ["pass", "fail"][isfailed]
            print("%.1f, %s" % (jkset.dlys[i], st))

    # Return proper data
    if summary == False:
        return np.array(ks_l), np.array(pval_l)
    else:
        failfrac = fails/len(jkset.dlys)
        return failfrac


def anderson(jkset, summary=False, verbose=False):
    """
    Does an Anderson-Darling test on the z-scores of the data. Prints
    results.

    One would expect the fraction of times the null hypothesis is rejected
    to be roughly the same as the confidence level if the distribution is
    normal, so the observed failure rates should match their respective
    confidence levels.

    Parameters
    ----------
    jkset: hera_stats.jkset.JKSet, ndim=1
        The jackknife set to use for running the anderson darling test.

    summary: boolean, optional
        If true, returns only the confidence intervals and the failure rates.
        Otherwise, returns them for every delay mode. Default: False.

    verbose: boolean, optional
        If true, prints out values neatly as well as returning them. Default: False.

    Returns
    -------
    sigs: list
        Significance levels for the anderson darling test. Returned if summary == True.

    fracs: list
        Fraction of anderson darling failures for each significance level.
        Returned if summary == True.

    stat_l: ndarray
        The Anderson statistic, as a spectrum with a value for every delay mode in jkset.
        Returned if summary == False.

    crit_l: ndarray
        The critical values, also returned as a spectrum that can be immediately plotted.
    """
    assert isinstance(jkset, jkset_lib.JKSet), "Expected jkset to be hera_stats.jkset.JKSet instance."
    assert jkset.ndim == 1, "Input jkset must have first dimension 1."

    spectra = jkset.spectra

    # Calculate Anderson statistic and critical values for each delay mode
    stat_l = []
    for i, col in enumerate(np.array(spectra).T):
        stat, crit, sig = spstats.anderson(col.flatten(), dist="norm")
        stat_l += [stat]

    if verbose:
        print("Samples: %i" % len(stat_l))

    # Print and save failure rates for each significance level
    fracs = []
    for i in range(5):
        frac = float(sum(np.array(stat_l) >= crit[i]))/len(stat_l) * 100
        if verbose:
            print("Significance level: %.1f \tObserved "
                  "Failure Rate: %.1f" % (sig[i], frac))
        fracs += [frac]

    # Return if specified
    if summary == False:
        return np.array(stat_l), np.array([list(crit)]*len(stat_l))
    else:
        return list(sig), fracs
