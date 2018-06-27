# Statistics and such for hera data
# Duncan Rocha, June 2018

import numpy as np
import pickle as pkl
from scipy import stats as spstats
import hera_pspec as hp
import utils

def get_data(pc, jkf=None, proj=None, sortby=None, zscore="varsum"):
    """
    Retrieves pspec information from a PSpecContainer.

    Parameters
    ----------
    pc: PSpecContainer
        The container in which jackknved data lives.

    jkf: int, optional
        If specified, only retrieves the data for this jackknife run.
        Default: None.

    proj: func, optional
        The projection function applied to the data before returned.
        If none is specfied, uses only the real components. Default: None.

    sortby: item, optional
        Sorts the data by the item before returning. If split_ants is used,
        and one specifies sortby=1, then the first group will always contain
        the antenna number 1. Default: None.

    zscore: string, optional
        Method of standardization to use. Automatically uses weighted sum if
        more than 3 jackknife groups are detected.
        Options: "varsum", "weightedsum". Default: "varsum".

    Returns
    -------
    dic: dict
        Dictionary contianing all of the data found in the PSpecContainer.
    """
    assert isinstance(pc, hp.container.PSpecContainer), "Expected pc to be PSpecContainer, not %r" % type(pc)

    if proj == None:
        proj = lambda x: x.real
    
    spectra, errs, grps, zs = [],[],[],[]
    jkftypes = []

    # Find all (or only one) jackknife groups
    all_jkfs = pc.groups()
    if isinstance(jkf, int):
        all_jkfs = [j for j in all_jkfs if int(j.split(".")[1]) == jkf]

    if len(all_jkfs) == 0:
        raise IndexError("Jackknife number requested not found.")

    nfail = 0
    for jk in sorted(all_jkfs):
        groups = pc.spectra(jk)
        spec_l, err_l, grp_l = [],[],[]

        for g in groups:
            # Get delays, spectra, and errors
            uvp = pc.get_pspec(jk,g)
            dlys = uvp.get_dlys(0) * 10**9
            key = uvp.get_all_keys()[0]
            avspec = proj(uvp.get_data(key)[0])
            errspec = proj(uvp.get_stats("bootstrap_errs", key)[0])
            #noise = proj(uvp_avg.generate_noise_spectra(0, pol, 300)[bl])[0]

            spec_l.append(avspec)
            err_l.append(errspec)
            grp_l.append(uvp.labels)
            jkftypes.append(jk.split(".")[0])

        # Sort by specific item if needed.
        if isinstance(sortby, int) and len(spec_l) == 2:
            ingrp = [str(sortby) in g for g in grp_l]
            if all([i == False for i in ingrp]):
                if sortby is not None:
                    #print "Item not found in jackknife %i" % int(jk.split(".")[1])
                    nfail += 1
                ingrp = [True, False]

            if all([ingrp[i] == i for i in [False, True]]):
                spec_l.reverse()
                grp_l.reverse()
                err_l.reverse()

        elif sortby == "reverse":
            spec_l.reverse()
            grp_l.reverse()
            err_l.reverse()

        z = standardize(spec_l, err_l, method=zscore)

        spectra.append(spec_l)
        errs.append(err_l)
        grps.append(grp_l)
        zs.append(z)

    if nfail >= len(spectra):
        raise ValueError("Sortby item %s not found" % str(sortby))

    if all([j == jkftypes[0] for j in jkftypes]):
        jkftype = jkftypes[0]
    else:
        raise ValueError("All jackknifes must be of the same type")

    if sortby is None:
        sortstring = ""
    else:
        sortstring = "(sorted by %s)" % str(sortby)

    dic = {"dlys": dlys, "spectra": np.array(spectra), "errs": np.array(errs),
           "grps": grps, "jkftype": jkftype, "sortitem": sortby,
           "sortstring": sortstring, "sortby": sortby, "zscores": zs}

    return dic

def standardize(spectra, errs, method="weightedsum"):
    """
    Calculates the z scores for a list of split spectra and their errors.

    Parameters
    ----------
    spectra: list (n_jacks x 2 x n_dlys)
        A list of pairs of spectra, to be used to calculate z scores.

    errs: list (n_jacks x 2 x n_dlys)
        List of pairs of errors corresponding to the above spectra

    method: string, optional
        Method used to calculate z-scores.
        Options: ["varsum", "weighted"]. Default: varsum.

    Returns
    -------
    zs: list
        Returns zscores for every spectrum pair given.
    """

    if len(spectra) != len(errs):
        raise AttributeError("Spectra pair list and corresponding errors "
                             "must be the same length")
    if len(spectra) < 2:
        raise AttributeError("Need more than two spectra")

    spectra = np.array(spectra)
    errs = np.array(errs)

    # Calculate z scores using sum of variances
    if method == "varsum" and len(spectra) == 2:
        comberr = np.sqrt(errs[0]**2 + errs[1]**2).clip(10**-10, np.inf)
        z = (spectra[0] - spectra[1])/comberr

    # Or Calculate z scores with weighted sum
    elif method == "weightedsum" or len(spectra) > 2:
        # Calculate weighted average and standard deviation.
        cerrs = np.sqrt(1./np.sum(errs**-2, axis=0))
        av = cerrs**2*np.sum(spectra*errs**-2, axis=0)
        std = cerrs*np.sqrt(len(spectra))
        if len(spectra) == 2:
            z = (spectra[0]-spectra[1])/(np.sqrt(2)*std)
        elif len(spectra) > 2:
            z = np.vstack([(spec - av)/(std) for spec in spectra])
    else:
        raise NameError("Z-score calculation method not recognized")

    return z

def kstest(pc, asspec=False, sortby=None, proj=None, bins=None, method="varsum",
           verbose=False):
    """
    The KS test is a test of normality, in this case, for a gaussian (avg,
    stdev) as specified by the parameter norm. If the p-value is below the
    KS stat, the null hypothesis (gaussian curve (0, 1)) is rejected.

    Parameters
    ----------
    pc: PSpecContainer
        The container in which jackknved data lives.

    asspec: boolean, optional
        If true, returns the ks test as a spectra, one value for each delay
        mode.

    sortby: item, optional
        Sorts the data by the item before returning. If split_ants is used,
        and one specifies sortby=1, then the first group will always contain
        the antenna number 1. Default: None.

    proj: func, optional
        The projection function applied to the data before returned.
        If none is specfied, uses only the real components. Default: None.

    bins: int or list, optional
        If int, number of bins in which to bin data before conducting
        ks test. If list, bins themselves. Default: None.

    method: string, optional
        Method used to calculate z-scores for Kolmogorov-Smirnov Test.
        Options: ["varsum", "weighted"]. Default: varsum.

    verbose: boolean, optional
        If true, prints information for every delay mode instead of
        summary. Default: False

    Returns
    -------
    failfrac: float
        The fraction of delay modes that fail the KS test.
    """
    # Calculate zscores
    dic = get_data(pc, sortby=sortby, proj=proj, zscore=method)
    dlys, zs = dic["dlys"], dic["zscores"]

    ks_l, pval_l = [], []
    fails = 0.
    if bins is not None:
        dlys, data = bin_data(dlys, zs, bins)
    else:
        data = np.array(zs).T

    for i, d in enumerate(data):

        # Do ks test on delay mode
        [ks, pval] = spstats.kstest(d, spstats.norm(0, 1).cdf)

        # Save result
        ks_l += [ks]
        pval_l += [pval]
        isfailed = int(pval < ks)
        fails += isfailed

        if verbose:
            st = ["pass", "fail"][isfailed]
            print "%i" % dlys[i], st

    # Return proper data
    if asspec:
        return dlys, ks_l, pval_l
    else:
        failfrac = fails/len(dlys)
        return failfrac

def anderson(pc, asspec=False, proj=None, sortby=None, method="varsum", verbose=False):
    """
    Does an Anderson-Darling test on the z-scores of the data. Prints
    results.

    One would expect the fraction of times the null hypothesis is rejected
    to be roughly the same as the confidence level if the distribution is
    normal, so the observed failure rates should match their respective
    confidence levels.

    Parameters
    ----------
    pc: PSpecContainer
        The container in which jackknved data lives.

    asspec: boolean, optional
        If true, returns the ks test as a spectra, one value for each delay
        mode.

    proj: func, optional
        The projection function applied to the data before returned.
        If none is specfied, uses only the real components. Default: None.

    sortby: item, optional
        Sorts the data by the item before returning. If split_ants is used,
        and one specifies sortby=1, then the first group will always contain
        the antenna number 1. Default: None.

    method: string, optional
        Method used to calculate z-scores for Anderson Darling Test.
        Options: ["varsum", "weighted"]. Default: varsum.

    showmore: boolean, optional
        If true, prints out values neatly as well as returning them

    Returns
    -------
    sigs: list
        Significance levels for the anderson darling test.

    fracs: list
        Fraction of anderson darling failures for each significance level.
    """
    # Calculate z-scores
    dic = get_data(pc, proj=proj, sortby=sortby, zscore=method)
    dlys, zs = dic["dlys"], dic["zscores"]

    # Calculate Anderson statistic and critical values for each delay mode
    statl = []
    for i, zcol in enumerate(np.array(zs).T):
        stat, crit, sig = spstats.anderson(zcol, dist="norm")
        statl += [stat]

    if verbose:
        print "Samples: %i" % len(statl)

    # Print and save failure rates for each significance level
    fracs = []
    for i in range(5):
        frac = float(sum(np.array(statl) >= crit[i]))/len(statl) * 100
        if verbose:
            print ("Significance level: %.1f \tObserved "
                   "Failure Rate: %.1f" % (sig[i], frac))
        fracs += [frac]

    # Return if specified
    if asspec:
        return dlys, statl, [list(crit)]*len(statl)
    else:
        return list(sig), fracs

def avspec_with_and_without(pc, item, proj=None, method="varsum"):
    """
    Returns the average spectrum for groups with and withou item,
    and errors.

    Parameters
    ----------
    pc: PSpecContainer
        The container in which jackknved data lives.

    item: int
        The item to sort by, if groups is a list of items.

    proj: func, optional
        The projection function applied to the data before returned.
        If none is specfied, uses only the real components. Default: None.

    method: string, optional
        Method used to calculate z-scores for average spectra.
        Options: ["varsum", "weighted"]. Default: varsum.
    """
    # Get data
    dic = get_data(pc, proj=proj, sortby=item)
    dlys, spectra, errs = dic["dlys"], dic["spectra"], dic["errs"]

    # Slice data according into two groups
    spectra = [np.array(spectra)[:, i, :] for i in [0, 1]]
    errs = [np.array(errs)[:, i, :] for i in [0, 1]]

    avspecs, averrs = [], []
    for i in [0, 1]:
        er = np.sqrt(1./np.sum(errs[i]**-2, axis=0))
        sp = er**2*np.sum(spectra[i]*errs[i]**-2, axis=0)
        avspecs.append(sp)
        averrs.append(er)

    return avspecs, averrs

def item_info(pc, item, proj=None):
    """
    Returns the summary for an item.

    pc: PSpecContainer
        The container in which jackknved data lives.

    item: int
        The item to sort by, if groups is a list of items.

    proj: func, optional
        The projection function applied to the data.
        If none is specfied, uses only the real components. Default: None.
    """
    dic = get_data(pc, sortby=item, proj=proj)

    avg = np.average(dic["zscores"], axis=0)
    std = np.std(dic["zscores"], axis=0)
    err = std/np.sqrt(len(dic["zscores"]))

    failed = np.where(abs((avg) / err) > 4)[0]
    if len(failed) > 0:
        dlysfl = dic["dlys"][failed]
        print "Item %r has a weird point at: %r" % (item, dlysfl)

    n = len(dic["dlys"])
    fails = kstest(pc, sortby=item, proj=None) * n

    print "Sorting by item %r fails the ks test in %i/%i delay modes" % (item,
                                                                         fails, n)
    anderson(pc, verbose=True)

def bin_data(dlys, spectra, bins, return_edges=False):
    """
    Bins spectra or zscores into delay ranges.
    """

    spectra = np.array(spectra)
    if isinstance(bins, int):
        walls = np.linspace(min(dlys)-1,
                            max(dlys)+1,
                            bins+1)
    else:
        walls = bins

    select = [(dlys > walls[i]) * (dlys < walls[i+1])
              for i in range(len(walls)-1)]

    binned = np.array([np.hstack(spectra[:, sel]) for sel in select])

    if return_edges:
        return walls, binned
    else:
        dlys = walls[:-1] + (walls[1] - walls[0])/2
        return dlys, binned
