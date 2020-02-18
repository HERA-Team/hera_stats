# Extra utilities for hera_stats
import numpy as np
import os, copy

def plt_layout(req):
    """
    Helper function for figuring out best layout of histogram table
    """
    # req is the requested number of plots. All numbers below it are 
    # potential divisors
    divisors = np.arange(1,req)

    # How many empty plots would be left
    left = [req%a for a in divisors]

    # difference between axis lengths
    howsquare = [np.abs(req//a - a) for a in divisors]

    lenientness = 0
    while True:
        for a in divisors-1:
            # Look for smallest number of empty plots left and close to equal 
            # side lengths
            if left[a] <= lenientness and howsquare[a] <= lenientness:
                return (divisors[a],int(np.ceil(float(req)/divisors[a])))
        # If nothing found, decrease strictness
        lenientness += 1

def bin_wrap(angles, n):

    angles = np.array(angles)
    degs = np.linspace(-360, 360, 720/10+1)
    ha = np.hstack([angles-360, angles])

    isindeg = [bool(sum((ha > degs[i]) * (ha < degs[i+1]))) for i in range(len(degs)-1)]
    longest = -1
    rng = None
    for i, val in enumerate(isindeg):
        if val == True:
            seq = isindeg[i: ]
            try:
                seq = seq[:seq.index(False)]
            except ValueError:
                pass
            if len(seq) > longest:
                longest = len(seq)
                rng = (i, i + len(seq) + 1)

    wrapped = ha[(ha > degs[rng[0]]) * (ha <= degs[rng[1]])]
    bins = np.linspace(min(wrapped), max(wrapped), n+1)
    bins -= bins//360 * 360
    return bins

def is_in_wrap(low, hi, angle):

    if angle >= low and angle < hi:
        return True
    angle += 180
    low += 180
    hi += 180
    if angle % 360 >= low % 360 and angle % 360 < hi % 360:
        return True

    return False


def trim_empty_bls(uvd, bl_grp):
    """
    Remove any baselines with zero data from a list of baseline groups.
    
    Parameters
    ----------
    uvd : UVData
        Object containing data.
    
    bl_grp : list of list of baseline tuples or ints
        List of groups of baselines.
    
    Returns
    -------
    bl_grp_trimmed : list of baseline tuples or ints
        Copy of bl_grp with all zeroed bls removed.
    """
    # Try to convert to list of lists if it isn't already
    if not isinstance(bl_grp[0], list):
        bl_grp = [bl_grp,]
    bl_grp_trimmed = []
    
    # Loop over groups
    for grp in bl_grp:
        
        # Loop over bls in group
        new_grp = []
        for bl in grp:
            flags = uvd.get_flags(bl)
            if not np.all(flags == True):
                new_grp.append(bl)
        bl_grp_trimmed.append(new_grp)
    return bl_grp_trimmed


def trim_empty_blpairs(uvp, bl_grp, spw=0, pol='pI'):
    """
    Remove any baseline-pairs with zero data from a list of blpairs.
    
    Parameters
    ----------
    uvp : UVPSpec
        Contains power spectra.
    
    bl_grp : list of lists of tuples
        List of redundant groups of baseline pairs, e.g. output by 
        hera_pspec.utils.get_blvec_reds().
        
    spw : int, optional
        ID of the spectral window to test for zero data. 
        Default: 0.
    
    pol : str, optional
        Which polarization to test for zero data. Default 'pI'.
    
    Returns
    -------
    bl_grp_trimmed : list of lists of tuples
        Copy of bl_grp with all zeroed blpairs removed. Redundant 
        groups with no data on any baseline pair will remain in 
        the list.
    """
    new_bl_grp = []
    
    # Loop over redundant groups
    for grp in bl_grp:
        new_grp = []
        for i, blp in enumerate(grp):
            dsum = uvp.get_data((spw, blp, pol)).real.sum()
            if dsum != 0.:
                new_grp.append(blp)
        new_bl_grp.append(new_grp)
    return new_bl_grp


def stacked_array(array_list):
    """
    Generates a long stacked array from a list of arrays.
    
    Parameters
    ----------
    array_list : list
        List of np.ndarray objects to be stacked.
    
    Returns
    -------
    array_total : np.ndarray
        Array of all arrays in array_list stacked in list index order.
    """
    counter = 0
    if len(array_list) == 0:
        raise ValueError("input array list cannot be empty")
    
    # Looping through all the arrays and stacking them up
    for i in range(len(array_list)):
        array_new = np.zeros(array_list[i].shape)
        if counter == 0:
            array_total = array_list[i]
        elif counter != 0:
            array_new = array_list[i]
            array_total = np.vstack((array_total, array_new))
        counter += 1
    return array_total


