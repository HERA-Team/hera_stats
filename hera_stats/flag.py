
import numpy as np
from pyuvdata import UVData
import matplotlib
from matplotlib import gridspec
import copy

def apply_random_flags(uvd, flag_frac, seed=None, inplace=False, 
                       zero_flagged_data=False):
    """
    Randomly flag a set of frequency channels. Flags are applied 
    on top of any existing flags, and are applied to all 
    baselines, times, and polarizations.
    
    Parameters
    ----------
    uvd : UVData object
        Input UVData object to be flagged.
    
    flag_frac : float
        Fraction of channels to flag. This is the fraction of 
        channels to apply flags to; the actual fraction of flagged 
        channels may be greater than this, depending on if there 
        were already flagged channels in the input UVData object.
    
    seed : int, optional
        Random seed to use. Default: None.
    
    inplace : bool, optional
        Whether to apply the flags to the input UVData object 
        in-place, or return a copy that includes the new flags. 
        Default: False.
    
    zero_flagged_data : bool, optional
        Whether to set the flagged channels in the data_array to 
        zero. This is useful for identifying functions that are 
        ignoring the mask. All flagged data will be zeroed, not 
        just the new flags added by this function.
    
    Returns
    -------
    uvd : UVData object
        Returns UVData object with flags applied.
    """
    assert flag_frac < 1. and flag_frac >= 0., \
        "flag_frac must be in the range 0, 1"
    
    # Get all available bls
    bls = np.unique(uvd.baseline_array)
    
    # Get total no. of channels and randomly select channels to flag
    freqs = uvd.freq_array
    chans = np.arange(freqs.size)
    nflagged = int(flag_frac * float(chans.size))
    if seed is not None: np.random.seed(seed)
    flagged = np.random.choice(chans, size=nflagged, replace=False)
    
    # Whether to apply mask in-place, or return a copy
    if inplace:
        new_uvd = uvd
    else:
        new_uvd = copy.deepcopy(uvd)
    
    # Apply flags
    new_uvd.flag_array[:,:,flagged,:] = True
    if zero_flagged_data:
        new_uvd.data_array[new_uvd.flag_array] = 0.
    return new_uvd


def flag_channels(uvd, spw_ranges, inplace=False):
    """
    Flags a given range of channels entirely for a list of UVData objects
    
    Parameters
    ----------
    uvd : UVData
        UVData object to be flagged.
        
    spw_ranges : list
        list of tuples of the form (min_channel, max_channel) defining which
        channels to flag.
    
    inplace : bool, optional
        If True, then the input UVData objects' flag arrays are modified, 
        and if False, new UVData objects identical to the inputs but with
        updated flags are created and returned (default is False).
    
    Returns:
    -------
    uvd_new : list
        Flagged UVData object.
    """
    # Check inputs
    if not isinstance(uvd, UVData):
        raise TypeError("uvd must be a UVData object")
    if not inplace:
        uvd_new = copy.deepcopy(uvd)            
    
    # Loop over all spw ranges to be flagged
    for spw in spw_ranges:
        if not isinstance(spw, tuple):
            raise TypeError("spw_ranges must be a list of tuples")
        
        # Loop over pols
        for pol in range(uvd.Npols):
            unique_bls = np.unique(uvd.baseline_array)
            
            # Loop over baselines
            for bl in unique_bls:
                bl_inds = np.where(np.in1d(uvd.baseline_array, bl))[0]
                fully_flagged = np.ones(uvd.flag_array[bl_inds, 0, 
                                                       spw[0]:spw[1], 
                                                       pol].shape, dtype=bool)
                if inplace:
                    uvd.flag_array[bl_inds, 0, spw[0]:spw[1], pol] = fully_flagged
                else:
                    uvd_new.flag_array[bl_inds, 0, spw[0]:spw[1], pol] \
                        = fully_flagged
    if inplace:
        return uvd
    else:
        return uvd_new
        
