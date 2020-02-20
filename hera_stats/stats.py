import numpy as np
import pickle as pkl
from scipy import stats as spstats
import hera_pspec as hp
from . import utils
import copy 
from functools import reduce


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
    assert error_field in list(uvp.stats_array.keys()), "{} not found in stats_array" \
           .format(error_field)
    new_field = "{}_zscore".format(error_field)

    # iterate over spectral windows
    for i, spw in enumerate(uvp.spw_array):
        # iterate over polarizations
        for j, polpair in enumerate(uvp.polpair_array):
            # iterate over blpairs
            for k, blp in enumerate(uvp.blpair_array):
                key = (spw, blp, polpair)

                # calculate z-score: real and imag separately
                d = uvp.get_data(key)
                e = uvp.get_stats(error_field, key)
                zsc = d.real / e.real + 1j * d.imag / e.imag

                # set into uvp
                uvp.set_stats(new_field, key, zsc)

    if not inplace:
        return uvp


def redgrp_pspec_covariance(uvp, red_grp, dly_idx, spw, polpair, mode='cov', 
                            verbose=False):
    """
    Calculate the covariance or correlation matrix for all pairs of delay 
    spectra in a redundant group, for a single delay bin. The matrix is 
    estimated by averaging over all LST samples.
    
    Parameters
    ----------
    uvp : UVPSpec
        Input UVPSpec object.
    
    red_grp : list
        List of redundant baseline pairs within a group.
    
    dly_idx : int
        Index of the delay bin to calculate the covariance matrix for.
        
    spw : int
        Index of spectral window to use.
    
    polpair : int or str or tuple
        Polarization pair.
    
    mode : str, optional
        Whether to calculate the covariance matrix ('cov') or correlation 
        matrix ('corr'). Default: 'cov'.
    
    verbose : bool, optional
        Whether to print status messages. Default: false.
    
    Returns
    -------
    cov_real, cov_imag : ndarrays
        Real and imaginary covariance or correlation matrices, of shape 
        (Nblps, Nblps).
    """
    # Check inputs
    if mode not in ['cov', 'corr']:
        raise ValueError("")
    if not isinstance(red_grp, list):
        raise TypeError("red_grp must be a list of blpairs.")
    
    # Load data to calculate covmat with
    dat = np.zeros((len(red_grp), uvp.Ntimes), dtype=np.complex64)
    for i, blp in enumerate(red_grp):
        if i % 100 == 0 and verbose:
            print("%d / %d" % (i, len(red_grp)))
        dat[i] = uvp.get_data((spw, blp, polpair))[:,dly_idx]
    
    # Calculate covariance or correlation matrix    
    if mode == 'corr':
        # Correlation matrix
        corr_re = np.corrcoef(dat.real)
        corr_im = np.corrcoef(dat.imag)
        return corr_re, corr_im
    else:
        # Covariance matrix
        cov_re = np.cov(dat.real)
        cov_im = np.cov(dat.imag)
        return cov_re, cov_im

