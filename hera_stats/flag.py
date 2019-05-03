


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
        ignoring the mask.
    
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
        new_uvd.data_array[:,:,flagged,:] = 0.
    return new_uvd
