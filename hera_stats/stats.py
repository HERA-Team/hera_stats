# Statistics and such for hera data
# Duncan Rocha, June 2018

import numpy as np
import pickle as pkl
from scipy import stats as spstats
import hera_pspec as hp
import utils
import copy 
import jkset as jkset_lib


def weightedsum(jkset, axis=1):
    """
    Calculates the weighted sum average of the spectra over a specific axis.

    The averages and errors are calculated in the following way:

        avg_err = 1. / sum(err ** -2)
        avg = avg_err * sum(x * err**-2)
        std = sqrt(avg_err * len(x))

    Thus, std is not the uncertainty on the average but the standard deviation
    of the distribution of x.

    Parameters
    ----------
    jkset: hera_stats.jkset.JKSet
        JKSet with which to perform the weighted sum.

    axis: int, 0 or 1, optional
        Axis along which to do the weighted sum. Default: 1.

    Returns
    -------
    jkset_avg: JKSet
        The average jkset, with errors that describe the standard deviation of the samples given.
    """
    jk = copy.deepcopy(jkset)

    assert isinstance(jkset, jkset_lib.JKSet), "Expected jkset to be hera_stats.jkset.JKSet instance."
    assert axis in [0, 1], "Axis must be either 1 or 0."

    # Do weighted sum calculation
    aerrs = 1. / np.sum(jk.errs ** -2, axis=axis)
    av = aerrs * np.sum(jk.spectra * jk.errs**-2, axis=axis)
    std = (aerrs * len(jkset.spectra)) ** 0.5
    
    # Transpose jkset if necessary
    if axis == 1:
        jk = jk.T()

    # Average or sum metadata
    nsamp = np.sum(jk.nsamples, axis=0)
    integrations = np.average(jk.integrations, axis=0)
    times = np.average(jk.times, axis=0)

    uvp_list = jk._uvp_list[0]

    # Set UVPSpec attrs
    for i, uvp in enumerate(uvp_list):
        uvp.data_array[0] = np.expand_dims(av[i][None], 2)
        uvp.stats_array["bootstrap_errs"][0] = np.expand_dims(std[i][None], 2)

        uvp.integration_array[0] = integrations[i][None]
        uvp.time_avg_array[0] = times[i][None]
        uvp.time_1_array[0] = times[i][None]
        uvp.time_2_array[0] = times[i][None]
        uvp.nsample_array[0] = nsamp[i][None]
        uvp.labels = np.array(["Weighted Sum"])

    # Create new JKSet
    uvp_list = np.expand_dims(uvp_list, axis)
    return jkset_lib.JKSet(uvp_list, "weightedsum")

def zscores(jkset, method="weightedsum", axis=1):
    """
    Calculates the z scores for a JKSet along a specified axis. This
    returns another JKSet object, which has the zscore data and errors
    equal to 0.

    Parameters
    ----------
    jkset: hera_stats.jkset.JKSet
        The jackknife set to use for calculating zscores.

    method: string, optional
        Method used to calculate z-scores. 
        Options: ["varsum", "weightedsum"]. Default: varsum.
        
        "Varsum" works only with two
        jackknife groups, and the standard deviation is calculated by:
        
        zscore = (x1 - x2) / sqrt(err1**2 + err2**2)
        
        Method "weightedsum" works for any number of spectra and
        calculates the weighted mean with stats.weightedsum, then
        calculates zscores like this:
        
        zscore = (x1 - avg) / avg_err

    Returns
    -------
    zs: list
        Returns zscores for every spectrum pair given.
    """
    assert isinstance(jkset, jkset_lib.JKSet), "Expected jkset to be hera_stats.jkset.JKSet instance."

    shape = jkset.shape
    assert axis in [0, 1], "Axis must be either 0 or 1."
    assert shape[axis] >= 2, "Need at least two spectra."

    if axis == 1:
        jkset = jkset.T()

    spectra = jkset.spectra
    errs = jkset.errs
    
    jkout = copy.deepcopy(jkset)

    # Or Calculate z scores with weighted sum
    if method == "weightedsum":
        # Calculate weighted average and standard deviation.
        aerrs = 1. / np.sum(errs ** -2, axis=0)
        av = aerrs * np.sum(spectra * errs**-2, axis=0)
        std = np.sqrt(aerrs * len(spectra))
        z = np.array([(spec - av)/(std) for spec in spectra])
        jkout.set_data(z, 0*z)

    # Calculate z scores using sum of variances
    elif method == "varsum":
        assert shape[axis] == 2, "Varsum can only take axes of length 2, got {}.".format(shape[axis])
        comberr = np.sqrt(errs[0]**2 + errs[1]**2).clip(10**-10, np.inf)
        z = ((spectra[0] - spectra[1])/comberr)[None]

        # Use weightedsum to shrink jkset to size, then replace data
        jkout = weightedsum(jkout, axis=0)
        jkout.set_data(z, 0*z)

    else:
        raise NameError("Z-score calculation method not recognized")

    if axis == 1:
        jkout = jkout.T()

    jkout.jktype = "zscore_%s" % method
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
    jkset: hera_stats.jkset.JKSet
        The jackknife set to use for running the ks test. Must have shape[0] == 1.

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
    assert jkset.shape[0] == 1, "Input jkset must have first dimension 1."

    if cdf == None:
        cdf = spstats.norm(0, 1).cdf

    spectra = jkset.spectra[0]

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
            print jkset.dlys[i], st

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
    jkset: hera_stats.jkset.JKSet
        The jackknife set to use for running the anderson darling test.
        Must have shape[0] == 1.

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
    assert jkset.shape[0] == 1, "Input jkset must have first dimension 1."

    spectra = jkset.spectra[0]

    # Calculate Anderson statistic and critical values for each delay mode
    stat_l = []
    for i, col in enumerate(np.array(spectra).T):
        stat, crit, sig = spstats.anderson(col.flatten(), dist="norm")
        stat_l += [stat]

    if verbose:
        print "Samples: %i" % len(stat_l)

    # Print and save failure rates for each significance level
    fracs = []
    for i in range(5):
        frac = float(sum(np.array(stat_l) >= crit[i]))/len(stat_l) * 100
        if verbose:
            print ("Significance level: %.1f \tObserved "
                   "Failure Rate: %.1f" % (sig[i], frac))
        fracs += [frac]

    # Return if specified
    if summary == False:
        return np.array(stat_l), np.array([list(crit)]*len(stat_l))
    else:
        return list(sig), fracs
