import numpy as np
import pickle as pkl
from scipy import stats as spstats
import hera_pspec as hp
import utils
import copy 
import jkset as jkset_lib


def uvp_zscore(uvp, error_field='bs_std', inplace=False):
    """
    Calculate a zscore of a UVPSpec object using
    entry 'error_field' in its stats_array. This
    assumes that the UVPSpec object has been already
    mean subtracted using
    hera_pspec.uvpspec_utils.subtract_uvp().

    The resultant zscore is stored in the stats_array
    as error_field + "_zscore".

    Parameters
    ----------
    uvp : UVPSpec object

    error_field : str, optional
        Key of stats_array to use as z-score normalization.

    inplace : bool, optional
        If True, add zscores into input uvp, else
        make a copy of uvp and return with zscores.

    Returns
    -------
    if inplace:
        uvp : UVPSpec object
    """
    if not inplace:
        uvp = copy.deepcopy(uvp)

    # check error_field
    assert error_field in uvp.stats_array.keys(), "{} not found in stats_array" \
           .format(error_field)
    new_field = "{}_zscore".format(error_field)

    # iterate over spectral windows
    for i, spw in enumerate(uvp.spw_array):
        # iterate over polarizations
        for j, pol in enumerate(uvp.pol_array):
            # iterate over blpairs
            for k, blp in enumerate(uvp.blpair_array):
                key = (spw, blp, pol)

                # calculate z-score: real and imag separately
                d = uvp.get_data(key)
                e = uvp.get_stats(error_field, key)
                zsc = d.real / e.real + 1j * d.imag / e.imag

                # set into uvp
                uvp.set_stats(new_field, key, zsc)

    if not inplace:
        return uvp


## Everything below is currently broken due to JKSet.py being broken
## See test_stats.py for example runs

def weightedsum(jkset, axis=0, error_field='bs_std'):
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
        Axis along which to do the weighted sum. Default: all axes.

    Returns
    -------
    jkset_avg: 1D JKSet
        The average jkset, with errors that describe the standard deviation
        of the samples given.
    """
    jk = copy.deepcopy(jkset)

    if isinstance(axis, int): axis = (axis,)
    assert isinstance(jkset, jkset_lib.JKSet), "Expected jkset to be hera_stats.jkset.JKSet instance."
    assert all([ax < jk.ndim for ax in axis]), "Axes %s was specified butn jkset has only axes <= %i." % (axis, jk.ndim - 1)

    # Do weighted sum calculation
    aerrs = 1. / np.sum(jk.errs ** -2, axis=axis)
    av = aerrs * np.sum(jk.spectra * jk.errs**-2, axis=axis)
    std = (aerrs * len(jk.spectra)) ** 0.5

    # Average or sum metadata
    nsamp = np.sum(jk.nsamples, axis=axis)
    integrations = np.average(jk.integrations, axis=axis)
    times = np.average(jk.times, axis=axis)

    # IF the spectra were averaged down to a single spectrum, of shape (Ndlys,), expand to (1, Ndlys).
    if len(av.shape[:-1]) == 0:
        targ_shape = (1, )
        av = av[None]
        std = std[None]
        nsamp = nsamp[None]
        integrations = integrations[None]
        times = times[None]

    # Slice jk to match shape of av and std
    key = [0] * jk.ndim
    for i in range(jk.ndim):
        if i not in axis:
            key[i] = slice(None, None, 1)
    jkav = jk[tuple(key)]

    # Set average and error of jk
    jkav.set_data(av, std, error_field=error_field)

    # Set UVPSpec attrs
    for i, uvp in enumerate(jkav._uvp_list):
        uvp.integration_array[0] = integrations[i][None]
        uvp.time_avg_array[0] = times[i][None]
        uvp.time_1_array[0] = times[i][None]
        uvp.time_2_array[0] = times[i][None]
        uvp.nsample_array[0] = nsamp[i][None]
        uvp.labels = np.array(["Weighted Sum"])

    return jkav


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

        # Calculate combined error and sum of zscores
        comberr = np.sqrt(errs[key1]**2 + errs[key2]**2).clip(10**-10, np.inf)
        z = ((spectra[key1] - spectra[key2])/comberr)

        # Use weightedsum to shrink jkset to size, then replace data
        jkout = weightedsum(jkout, axis=axis, error_field=error_field)
        #print jkout.shape, z.shape
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
            print "%.1f, %s" % (jkset.dlys[i], st)

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
