import numpy as np
import hera_pspec as hp
from . import utils
import copy

def _stripe_array(x, stripes, width=1):
    """
    Split 1D array into a number of stripes of given width.
    
    Parameters
    ----------
    x : array_like
        Input array of values to be striped.
    
    stripes : int
        Number of stripes.
    
    width : int, optional
        Width of each stripe. Default: 1.
    
    Returns
    -------
    x_stripes : list of array_like
        List containing each stripe of the input array.
    """
    blocks = []
    # Loop over no. of stripes
    for i in range(stripes): 
        x_stripe = []
        for j in range(width): # loop over width of stripe 
            x_stripe = np.concatenate((x_stripe, 
                                       x[i*(width)+j::stripes*width]))
        blocks.append( np.sort(x_stripe) )
    return blocks


def lst_blocks(uvp, blocks=2, lst_range=(0., 2.*np.pi)):
    """
    Split a UVPSpec object into multiple UVPSpec objects, each containing 
    spectra within different contiguous LST ranges. There is no guarantee that 
    each block will contain the same number of spectra or samples.
    
    N.B. This function uses the `lst_avg_array` property of the input UVPSpec 
    object to split the LSTs (and not the LSTs of the individual visibilities 
    that went into creating each delay spectrum).
    
    Parameters
    ----------
    uvp : UVPSpec
        Object containing delay spectra.
    
    blocks : int, optional
        How many blocks to return. Default: 2.
    
    lst_range : tuple, optional
        Tuple containing the minimum and maximum LST to retain. This is the 
        range that will be split up into blocks. Default: (0., 2*pi)
    
    Returns
    -------
    uvp_list : list of UVPSpec
        List of UVPSpec objects, one for each LST range. Empty blocks will 
        appear as None in the list.
    
    lst_bins : array_like
        Array of LST bin edges. This has dimension (blocks+1,).
    """
    # Check validity of inputs
    assert isinstance(uvp, hp.UVPSpec), "uvp must be a single UVPSpec object."
    assert lst_range[0] >= 0. and lst_range[1] <= 2.*np.pi, \
        "lst_range must be in the interval (0, 2*pi)"
    assert isinstance(blocks, (int, np.int, np.integer)), \
        "'blocks' must be an integer"
    assert blocks > 0, "Must have blocks >= 1"
    
    # Get LSTs
    lsts = np.unique(uvp.lst_avg_array)
    
    # Define bin edges
    lst_bins = np.linspace(lst_range[0], lst_range[1], blocks+1)
    
    # Loop over bins and select() the LST ranges required
    uvp_list = []
    for i in range(lst_bins.size - 1):
        idxs = np.where( np.logical_and(lsts >= lst_bins[i], 
                                        lsts < lst_bins[i+1]) )[0]
        _uvp = None
        if idxs.size > 0:
            # Select LSTs in this range
            _uvp = uvp.select(lsts=lsts[idxs], inplace=False)
        uvp_list.append(_uvp)
    
    return uvp_list, lst_bins


def lst_stripes(uvp, stripes=2, width=1, lst_range=(0., 2.*np.pi)):
    """
    Split a UVPSpec object into multiple UVPSpec objects, each containing 
    spectra within alternating stripes of LST.
    
    N.B. Gaps in LST are ignored; this function stripes based on the ordered 
    list of available LSTs in `uvp` only.
    
    N.B. This function uses the `lst_avg_array` property of the input UVPSpec 
    object to split the LSTs (and not the LSTs of the individual visibilities 
    that went into creating each delay spectrum).
    
    Parameters
    ----------
    uvp : UVPSpec
        Object containing delay spectra.
    
    stripes : int, optional
        How many stripes to return. Default: 2.
    
    width : int, optional
        Width of each stripe, in number of LST bins. Default: 1.
    
    lst_range : tuple, optional
        Tuple containing the minimum and maximum LST to retain. This is the 
        range that will be split up into blocks. Default: (0., 2*pi)
    
    Returns
    -------
    uvp_list : list of UVPSpec
        List of UVPSpec objects, one for each LST range.
    """
    # Check validity of inputs
    assert isinstance(uvp, hp.UVPSpec), "uvp must be a single UVPSpec object."
    assert lst_range[0] >= 0. and lst_range[1] <= 2.*np.pi, \
        "lst_range must be in the interval (0, 2*pi)"
    assert isinstance(stripes, (int, np.int, np.integer)), \
        "'stripes' must be an integer"
    assert stripes > 0, "Must have stripes >= 1"
    assert width > 0, "Must have width >= 1"
    
    # Get sorted list of LSTs within the specified range
    lsts = np.unique(uvp.lst_avg_array)
    idxs = np.where( np.logical_and(lsts >= lst_range[0], 
                                    lsts < lst_range[1]) )
    lsts = np.sort(lsts[idxs])
    
    # Loop over bins and select() the LST ranges required 
    uvp_list = []
    lst_stripes = _stripe_array(lsts, stripes=stripes, width=width)
    for lstripe in lst_stripes:
        _uvp = uvp.select(lsts=lstripe, inplace=False)
        uvp_list.append(_uvp)
    
    return uvp_list
    
