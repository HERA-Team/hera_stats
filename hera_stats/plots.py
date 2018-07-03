# Plots for visualizing data

import matplotlib.pyplot as plt
import numpy as np
import stats
import utils
from stats import get_pspec_stats


def plot_spectra(pc, n=0, fig=None, sortby=None, show_groups=False,
                      with_errors=True, proj=None, zlim=5):
    """
    Plots a pair of spectra, their errors, and normalized residuals.

    Parameters
    ----------
    pc: hera_pspec.container.PSpecContainer
        PSpecContainer containing jackknife information. Must be
        same style as that outputted by jackknives module.

    n: int, optional
        Number (out of total) position of the spectra pair to use
        [1 - n_jacks]. Default: 1.

    fig: matplotlib figure, optional
        Where to plot the spectra. If None, creates figure. Default: None.

    sortby: int, list, or string, optional
        Item (e.g. antenna_num) to use to sort data. Default: None.

    show_groups: boolean, optional
        Whether to label spectra arbitrarily or by the antenna used.
        Default: False.

    with_errors: boolean, optional
        If true, plots errorbars and zscores.

    proj: function, optional
        Projection function to use on data before displaying.
        Default: lambda x: x.real.

    zlim: int or float, optional
        Limit of the zscore subplot y axis.
    """
    dic = stats.get_pspec_stats(pc, proj=proj, sortby=sortby, jkf=n)
    dlys, spec, er = dic["dlys"], np.abs(dic["spectra"][0]), dic["errs"][0]

    if len(spec) < 2:
        with_errors = False

    if fig is None:
        fig = plt.figure(figsize=(8, 5))

    if with_errors:
        ax = fig.add_axes([0.1, 0.3, 0.8, 0.7])
    else:
        ax = fig.add_subplot(111)

    # Choose labels appropriately
    if show_groups:
        labels = dic["grps"][0]
    else:
        labels = ["Group %i" % i
                  for i in range(len(dic["grps"][0]))]

    # Plot spectra with errorbars
    if with_errors == True:
        p_l = [ax.errorbar(dlys, spec[i], er[i]) for i in range(len(spec))]

    else:
        p_l = ax.plot(dlys, spec.T)
    #noise = ax.plot(dlys, noise[n], ls="--")

    # Set ylims based on range of power spectrum
    mx, mn = np.max(np.array(spec)), np.min(np.array(spec))
    ax.set_ylim(mn*0.6, mx*10.)
    ax.set_yscale("log")
    ax.set_ylabel(r"P($\tau$)")
    ax.set_xlabel("Delay (ns)")
    ax.grid(True)

    # Set other details
    if len(p_l) < 10 or show_groups is True:
        ax.legend(p_l, labels + ["noise"], fontsize=8, loc=1)
    ax.set_title("Power Spectrum for Jackknife %i %s" % (n, dic["sortstring"]))

    if with_errors:
        # Create second plot for z scores
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])

        # plot z scores as scatterplot
        zs = dic["zscores"]
        if len(zs[0]) > 1:
            [ax2.scatter(dlys, z, marker="+", color="green") for z in zs[0]]
        else:
            ax2.scatter(dlys, zs[0], marker="+", color="green")
        xlims = ax.get_xlim()

        # Plot zero line
        ax2.hlines([0], [-10000], [10000], linestyles="--", alpha=0.6)

        # Calculate yticks.
        zmt = zlim//2*2
        ticks = range(-zmt, zmt+1, 2)

        # Reinstate limits and set other paramters
        ax2.set_ylim(-zlim, zlim)
        ax2.set_xlim(xlims[0], xlims[1])
        ax2.set_yticks(ticks)
        ax2.set_xlabel("Delay (ns)")
        ax2.set_ylabel("Z-Score")
        ax2.grid()

def scatter(pc, ax=None, ylim=None, compare=True, proj=None, sortby=None):
    """
    Plots a scatterplot of the two groups of data. If sorted, makes it
    clear which group contains abnormalities.

    Parameters
    ----------
    pc: hera_pspec.container.PSpecContainer
        PSpecContainer containing jackknife information. Must be
        same style as that outputted by jackknives module.

    ax: matplotlib axis, optional
        If specified, plots on this axis. Default: None.

    ylim: tuple, optional
        The (min, max) y limits on the scatter plot. Default: None.

    compare: boolean, optional
        If False, plots everything in gray. If True, separates colors
        based on group.

    proj: function, optional
        Projection function to use on data before displaying.
        Default: lambda x: x.real.

    sortby: int, list, or string, optional
        Item (e.g. antenna_num) to use to sort data. Default: None.
    """
    dic = stats.get_pspec_stats(pc, proj=lambda x: np.abs(x.real).clip(1e-10, np.inf), sortby=sortby)
    dlys, spectra, errs = dic["dlys"], dic["spectra"], dic["errs"]
    
    if ax is None:
        f = plt.figure(figsize=(8, 5))
        ax = f.add_subplot(111)

    # Get data to plot
    wid = (dlys[1] - dlys[0]) 
    xs = np.hstack([dlys + np.random.uniform(-wid/2, wid/2, len(dlys)) 
                    for i in range(len(spectra)*len(spectra[0]))])
    ys = [np.hstack([sp[i] for sp in spectra]) for i in range(len(spectra[0]))]
    #y2 = np.hstack([sp[1] for sp in spectra])

    x = xs
    y = np.hstack(ys)

    if compare and len(ys):
        # Set colors
        colors = [["C%i" % (i//10)] * len(ys[i]) for i in range(len(ys))]
        colors = np.hstack(colors)

        # Shuffle order so colors are randomly in front or behind others
        inds = range(len(x))
        ys = np.hstack(ys)
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
    alpha = 0.4
    ax.scatter(x, y, color=colors, edgecolors="none", alpha=alpha)

    # Set other data
    ax.set_ylabel("Delay (ns)")
    ax.set_ylabel(r"P($\tau$)")
    ax.set_yscale("log")
    ax.set_ylim(ylim[0], ylim[1]*10)
    ax.set_title("Scatter Plot of All Data")

    # Set legend and title if sorted
    if dic["sortitem"] is not None:
        sc1 = ax.scatter([], [], color="red")
        sc2 = ax.scatter([], [], color="blue")
        ax.set_title("Scatterplot of All Data " +
                     "(Sorted by \"%s\")" % dic["sortitem"])
        ax.legend([sc1, sc2],
                  ["Groups with %s" % dic["sortitem"],
                   "Groups without "])

def hist_2d(pc, plottype="varsum", ax=None, ybins=40, sortby=None, display_stats=True,
            vmax=None, normalize=False, returned=False):
    """
    Plots 2d power spectrum for the data.

    Parameters
    ----------
    pc: hera_pspec.container.PSpecContainer
        PSpecContainer containing jackknife information. Must be
        same style as that outputted by jackknives module.

    plottype: string, optional
        Type of plot. Options: ["varsum", "raw", "norm", "weightedsum"].
        Default: "varsum"

    ax: matplotlib axis, optional
        If specified, plots the heatmap on this axis.

    ybins: int or list, optional
        If int, the number of ybins to use. If list, the entries are used
        as bin locations. Note: "raw" requires logarithmic binning.
        Default: 40

    sortby: int, list, or string, optional
        Item (e.g. antenna_num) to use to sort data. Default: None.

    display_stats: boolean, optional
        If true, plots the average and 1-sigma confidence interval of the
        data. Default: True

    vmax: float or int, optional
        If set, sets the value for the maximum color. Useful if comparing
        data sets. Default: None.

    normalize: boolean, optional
        If true, normalizes along the vertical axis, so that the sum of all
        values in a single delay bin is 1. Default: False

    returned: boolean, optional
        If true, returns the histogram as well as plotting it. Default:
        False.
    """

    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

    if plottype == "varsum":
        # Plot zscores using sum of variances
        dic = stats.get_pspec_stats(pc, sortby=sortby)
        data = dic["zscores"]
        ylims = (-5, 5)

    elif plottype == "raw":
        # Plot raw spectra, combine jackknife pairs
        dic = get_pspec_stats(pc, proj=lambda x: np.abs(x.real).clip(10**-10, np.inf), sortby=sortby)
        data = dic["spectra"]
        ylims = (np.log10(np.min(data)),
                 np.log10(np.max(data)))
        ax.set_yscale("log")

    elif plottype == "weightedsum":
        # Plot zscores using sum of variances method
        dic = stats.get_pspec_stats(pc, sortby=sortby, zscore="weightedsum")
        data = dic["zscores"]
        ylims = (-5, 5)

    elif plottype == "imag":
        # Plots zscores of imaginary values.
        dic = stats.get_pspec_stats(pc, sortby=sortby, proj=lambda x: x.imag)
        data = dic["zscores"]
        ylims = (-5, 5)
    else:
        # Otherwise, type not recognized
        raise NameError("Plot type not recognized")

    dlys = dic["dlys"]
    # Make bins
    if type(ybins) == int:
        ybins = np.linspace(ylims[0], ylims[1], ybins+1)

        if plottype == "raw":
            ybins = 10**ybins

    xbins = np.linspace(min(dlys)-1,
                        max(dlys)+1,
                        len(dlys)+1)

    x = np.hstack([dlys]*len(data)*len(data[0]))
    y = np.hstack(data).flatten()

    if display_stats:
        # Plot average and stdev if requested
        avs = np.average(data, axis=(0,1))
        stdevs = np.std(data, axis=(0,1))
        av, = ax.plot(dlys, avs, c="black", lw=2)
        sigma, = ax.plot(dlys, avs+stdevs, c="red", ls="--",
                         lw=2)
        ax.plot(dlys, avs-stdevs, c="red", ls="--", lw=2)
        ax.legend([av, sigma], ["Average", "1-Sigma"], loc=1)
        ax.set_ylim(min(ybins), max(ybins))

    # Calculate histogram
    hist, xpts, ypts = np.histogram2d(x, y, bins=[xbins, ybins])

    if normalize:
        hist /= len(data)
    if vmax is None:
        vmax = np.max(hist)/2

    # Plot hist
    c = ax.pcolor(xpts, ypts, hist.T, vmax=vmax)
    plt.colorbar(c, ax=ax)

    # Set labels
    ax.set_title("Jackknife Data Histogram %s"
                 % dic["sortstring"], fontsize=16)
    ax.set_xlabel("Delay (ns)")

    if plottype == "norm":
        ax.set_ylabel("Normalized Difference")
    elif plottype == "raw":
        ax.set_ylabel(r"P($\tau$)")
    else:
        ax.set_ylabel("Z-Score")

    if returned:
        return hist

def plot_kstest(pc, ax=None, sortby=None, bins=None, method="varsum"):
    """
    Plots the Kolmogorov-Smirnov test for each delay mode.

    The KS test is a test of normality, in this case, for a gaussian
    (avg, stdev) as specified by the parameter norm. If the p-value
    is below the KS stat, the null hypothesis (gaussian curve (0, 1))
    is rejected.

    Parameters
    ----------
    pc: hera_pspec.container.PSpecContainer
        PSpecContainer containing jackknife information. Must be
        same style as that outputted by jackknives module.

    ax: matplotlib axis, optional
        The axis on which to plot the ks test. Default: None.

    sortby: int, list, or string, optional
        Item (e.g. antenna_num) to use to sort data. Default: None.

    bins: int, optional
        If specified, bins the data before doing a ks test. Can
        be useful for small data sets. Default: None.

    method: string, optional
        Method used to calculate z-scores for Kolmogorov-Smirnov Test.
        Options: ["varsum", "weighted"]. Default: varsum.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

    # Get ks information
    dlys, ks, pvals = stats.kstest(pc, sortby=sortby, 
                                   method=method, bins=bins, summary=False)

    failfrac = sum(np.array(pvals) < np.array(ks))

    # Plot it
    p2, = ax.plot(dlys, pvals)
    p1, = ax.plot(dlys, ks)

    # Set text
    ax.legend([p1, p2], ["ks-stat", "p-val"], loc=1)
    ax.text(-5100, 1.1, "Fail Fraction: %.1f" % failfrac + "%")
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Statistics")
    sortstring = ""
    ax.set_title("Kolmogorov-Smirnov Test by Delay Bin %s"
                 % sortstring)
    ax.set_ylim(0, 1.2)
    ax.grid(True)

def plot_anderson(pc, ax=None, sortby=None, method="varsum"):
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
    pc: hera_pspec.container.PSpecContainer
        PSpecContainer containing jackknife information. Must be
        same style as that outputted by jackknives module.

    ax: matplotlib axis, optional
        The axis on which to plot the Anderson Darling Test. Default: None.

    sortby: int, list, or string, optional
        Item (e.g. antenna_num) to use to sort data. Default: None.

    method: string, optional
        Method used to calculate z-scores for Anderson Darling Test.
        Options: ["varsum", "weighted"]. Default: varsum.
    """

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

    # Get anderson statistics
    dlys, stat, crit = stats.anderson(pc, summary=False, sortby=sortby, method=method)

    # Plot them
    p1 = ax.plot(dlys, stat)
    lp = ax.plot(dlys, crit)[::-1]
    sigs = ["1%", "2.5%", "5%", "10%", "15%"]

    # Plot significance and failure info
    ax.legend(p1+lp, ["Stat"]+sigs, loc=1)
    fails = [sum(np.array(stat) > c) for c in crit[0]]

    for i in range(5):
        ax.text(-5000, crit[0][i]+0.02, "%.1f" % fails[i]+"%")

    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Statistics")
    sortstring = ""
    ax.set_title("Anderson Darling Test by Delay Bin %s"
                 % sortstring)
    ax.set_ylim(0, max(crit[0])*1.1)
    ax.grid(True)

def plot_zscore_stats(pc, ax=None, proj=None, sortby=None, method="varsum"):
    """
    Plots the average and standard deviation of the data as a spectrum.

    Parameters
    ----------
    pc: hera_pspec.container.PSpecContainer
        PSpecContainer containing jackknife information. Must be
        same style as that outputted by jackknives module.

    ax: matplotlib axis, optional
        Axis on which to plot average and stdev. Default: None.

    proj: function, optional
        Projection function to use on data before displaying.
        Default: lambda x: x.real.

    sortby: int, list, or string, optional
        Item (e.g. antenna_num) to use to sort data. Default: None.

    method: string, optional
        Method used to calculate z-scores.
        Options: ["varsum", "weighted"]. Default: varsum.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

    # Get average and stdev
    dic = stats.get_pspec_stats(pc, sortby=sortby, proj=proj, zscore=method)
    dlys = dic["dlys"]
    av = np.average(np.vstack(dic["zscores"]), axis=0)
    std = np.std(np.vstack(dic["zscores"]), axis=0)

    # plot av and stdev and set other parameters
    p = [ax.errorbar(dlys, av, std/np.sqrt(len(dic["zscores"])))]
    p.extend(ax.plot(dlys, std))

    ax.set_title("Average Z-Score %s" % dic["sortstring"])
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Z-Score")
    ax.legend(p, ["Average Z-Score", "Std Dev of Z-Scores"])
    ax.set_ylim(-1, 3)
    ax.grid(True)

def plot_with_and_without(pc, item, fig=None, proj=None,
                          with_errors=True, zlim=4):
    """
    Plots the average spectra with and without a certain item.

    Parameters
    ----------
    pc: hera_pspec.container.PSpecContainer
        PSpecContainer containing jackknife information. Must be
        same style as that outputted by jackknives module.

    item: int, list, or string, optional
        Item (e.g. antenna_num) to use to sort data. Default: None.

    proj: function, optional
        Projection function to use on data before displaying.
        Default: lambda x: x.real.

    with_errors: boolean, optional
        If true, also includes a plot of the zscores. Default: False.

    zlim: float, optional
        The limit on the z-axis of the zscore plot. Default: 4.
    """
    
    if fig is None:
        fig = plt.figure(figsize=(8, 5))
    
    ax = fig.add_subplot(111)
    dic = stats.get_pspec_stats(pc, sortby=item, proj=lambda x: x.real)
    spectra, errs = dic["spectra"], dic["errs"]

    av, err = [], []
    for i in [0, 1]:
        er = np.sqrt(1./np.sum(errs[i]**-2, axis=0))
        av.append(er**2*np.sum(spectra[i]*errs[i]**-2, axis=0))
        err.append(er)

    p = [ax.errorbar(dic["dlys"], np.abs(av[i]), err[i]) for i in [0, 1]]

    ax.legend(p, ["Avg Spectra w/ item %s" % str(dic["sortitem"]),
                  "Avg Spectra w/o"])
    ax.set_title("Average spectrum with vs. without %r" % item)
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Z-Score")
    ax.set_yscale("log")
    ax.grid(True)

    if with_errors:
        # Create second plot for z scores
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])

        # plot z scores as scatterplot
        zs = stats.standardize(av, err)
        ax2.scatter(dic["dlys"], zs, marker="+", color="green")
        xlims = ax.get_xlim()

        # Plot zero line
        ax2.hlines([0], [-10000], [10000], linestyles="--", alpha=0.6)

        # Calculate yticks.
        zmt = zlim//2*2
        ticks = range(-zmt, zmt+1, 2)

        # Reinstate limits and set other paramters
        ax2.set_ylim(-zlim, zlim)
        ax2.set_xlim(xlims[0], xlims[1])
        ax2.set_yticks(ticks)
        ax2.set_xlabel("Delay (ns)")
        ax2.set_ylabel("Z-Score")
        ax2.grid()

def split_hist(pc, fig=None, bins=11, hist_bins=30, ylim=None, xlim=4,
               proj=None, sortby=None):
    """
    Calculates z-scores for spectrum pairs, bins them by delay mode, and
    plots histogram for each bin.

    Parameters
    ----------
    pc: hera_pspec.container.PSpecContainer
        PSpecContainer containing jackknife information. Must be
        same style as that outputted by jackknives module.

    fig: matplotlib figure, optional
        If specified, uses this figure to plot histogram table.
        Default: None.

    bins: int, optional
        Number of delay bins to use. Default: 11.

    hist_bins: int, optional
        Number of histogram bins to use. Default: 30.

    ylim: float, optional
        Upper limit on the y axis of the graph. Default: 35

    proj: function, optional
        Projection function to use on data before displaying.
        Default: lambda x: x.real.

    sortby: int, list, or string, optional
        Item (e.g. antenna_num) to use to sort data. Default: None.
    """

    dic = stats.get_pspec_stats(pc, proj=proj, sortby=sortby)
    dlys, spectra, errs = dic["dlys"], dic["spectra"], dic["errs"]
    if fig is None:
        fig = plt.figure(figsize=(12, 8))

    # Bin Z-scores into delay buckets
    edges, binned = stats.bin_data_into_dlys(dlys, dic["zscores"][:, 0], bins, return_edges=True)

    # Plot sample spectrum
    layout = utils.plt_layout(len(binned)+1)
    ax = fig.add_subplot(layout[0], layout[1], 1)
    ax.plot(dlys, np.abs(spectra[0][0]), lw=2)
    ax.set_yscale("log")

    # Plot buckets on top of spectrum
    lo, hi = min(np.abs(spectra[0][0])), max(np.abs(spectra[0][0]))
    ax.fill_between(edges, lo/5, hi*5, color="black", alpha=0.08)
    ax.vlines(edges, lo/5, hi*5, linestyles="--")
    plt.ylim(lo/10, hi*100)

    ax.set_title("Bin Range Display")

    # plot bucket numbers
    width = edges[1]-edges[0]
    [ax.text(edges[a]+width/2 - 300, hi*10, a+1)
     for a in range(len(binned))]

    # these bins are for the histogram
    hbins = np.linspace(-xlim, xlim, hist_bins)
    dlys, kspec, p_spec = stats.kstest(pc, summary=False, bins=bins)

    if ylim is None:
        std = np.median(np.std(dic["zscores"], axis=0))
        N = 1. * len(dic["dlys"]) * len(dic["zscores"]) / len(edges)
        dx = 2. * xlim / hist_bins
        ylim = 1.5 * N * dx / (std * np.sqrt(2 * np.pi))

    for i in range(len(binned)):

        ax = fig.add_subplot(layout[0], layout[1], i+2)

        # Plot histogram of data
        d, k, p = binned[i], kspec[i], p_spec[i]
        
        if p < k:
            color = "red"
        else:
            color = "green"

        ax.hist(d, color=color, alpha=0.6, bins=hbins)
        ax.set_title("Delay range: %.0f" % edges[i] +
                     " to %.0f" % edges[i+1],
                     color=color, fontsize=10)

        # Calculate average and standard deviation and display on graph
        average = np.average(d)
        stdev = np.std(d)
        ax.text(-0.9*xlim, 0.75*ylim, ("Avg: %.2f\nStd Dev: %.2f\nN: %.0f")
                % (average, stdev, len(np.hstack(d))))

        ax.text(0.35*xlim, 0.75*ylim,
                "Bin: %.0f \nkstat: %.2f \npval: %.2f" % (i+1, k, p))

        # Set graph limits
        ax.set_ylim(0, ylim)
        ax.set_xlim(-xlim, xlim)

    fig.tight_layout()