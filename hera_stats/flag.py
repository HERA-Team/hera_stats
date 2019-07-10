
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


def construct_factorizable_mask(uvd_list, spw_ranges, first='col', 
                                greedy_threshold=0.3, n_threshold=1, 
                                retain_flags=True, unflag=False, greedy=True, 
                                inplace=False):
    """
    Generates a factorizable mask using a 'greedy' flagging algorithm, run on a 
    list of UVData objects. In this context, factorizable means that the flag 
    array can be written as F(freq, time) = f(freq) * g(time), i.e. entire rows 
    or columns are flagged.
    
    First, flags are added to the mask based on the minimum number of samples 
    available for each data point. Next, depending on the `first` argument, 
    either full columns or full rows that have flag fractions exceeding the 
    `greedy_threshold` are flagged. Finally, any rows or columns with remaining 
    flags are fully flagged. (Unflagging the entire array is also an option.)
    
    Parameters
    ----------
    uvd_list : list
        list of UVData objects to operate on

    spw_ranges : list
        list of tuples of the form (min_channel, max_channel) defining which
        spectral window (channel range) to flag. `min_channel` is inclusive,
        but `max_channel` is exclusive.
    
    first : str, optional
        Either 'col' or 'row', defines which axis is flagged first based on
        the `greedy_threshold`. Default: 'col'.
        
    greedy_threshold : float, optional
        The flag fraction beyond which a given row or column is flagged in the
        first stage of greedy flagging. Default: 0.3.
        
    n_threshold : float, optional
        The minimum number of samples needed for a pixel to remain unflagged. 
        Default: 1.
    
    retain_flags : bool, optional
        If True, then data points that were originally flagged in the input 
        data remain flagged, even if they meet the `n_threshold`. Default: True.
        
    unflag : bool, optional
        If True, the entire mask is unflagged. No other operations (e.g. greedy 
        flagging) will be performed. Default: False.
        
    greedy : bool, optional
        If True, greedy flagging takes place. If False, only `n_threshold` 
        flagging is performed (so the resulting mask will not necessarily be 
        factorizable). Default: True.
        
    inplace : bool, optional
        Whether to return a new copy of the input UVData objects, or modify 
        them in-place. Default: False (return copies).
    
    Returns
    -------
    uvdlist_new : list
        if inplace=False, a new list of UVData objects with updated flags 
    """
    # Check validity of input args
    if first not in ['col', 'row']:
        raise ValueError("'first' must be either 'row' or 'col'.")
    if not isinstance(uvd_list, list):
        raise TypeError("uvd_list must be a list of UVData objects")
    
    # Check validity of thresholds
    allowed_types = (float, np.float, int, np.integer)
    if not isinstance(greedy_threshold, allowed_types) \
      or not isinstance(n_threshold, allowed_types):
        raise TypeError("greedy_threshold and n_threshold must be float or int")
    if greedy_threshold >= 1. or greedy_threshold <= 0.:
        raise ValueError("greedy_threshold must be in interval [0, 1]")
    
    # List of output objects
    uvdlist_new = []
    
    # Loop over datasets
    for uvd in uvd_list:
        if not isinstance(uvd, UVData):
            raise TypeError("uvd_list must be a list of UVData objects")
        if not inplace:
            uvd_new = copy.deepcopy(uvd)
        
        # Loop over defined spectral windows
        for spw in spw_ranges:
            if not isinstance(spw, tuple):
                raise TypeError("spw_ranges must be a list of tuples")
            
            # Unflag everything and return
            if unflag:                
                if inplace:
                    uvd.flag_array[:, :, spw[0]:spw[1], :] = False
                    continue
                else:
                    uvd_new.flag_array[:, :, spw[0]:spw[1], :] = False
                    uvdlist_new.append(uvd_new)
                    continue
            
            # Greedy flagging algorithm
            # Loop over polarizations
            for n in range(uvd.Npols):
                
                # iterate over unique baselines
                ubl = np.unique(uvd.baseline_array)
                for bl in ubl:
                    
                    # Get baseline-times indices
                    bl_inds = np.where(np.in1d(uvd.baseline_array, bl))[0]
                    
                    # create a new array of flags with only those indices
                    flags = uvd.flag_array[bl_inds, 0, :, n].copy()
                    nsamples = uvd.nsample_array[bl_inds, 0, :, n].copy()
                    
                    Ntimes = int(flags.shape[0])
                    Nfreqs = int(flags.shape[1])
                    
                    narrower_flags_window = flags[:, spw[0]:spw[1]]
                    narrower_nsamples_window = nsamples[:, spw[0]:spw[1]]
                    flags_output = np.zeros(narrower_flags_window.shape)
                    
                    # If retaining flags, an extra condition is added to the 
                    # threshold filter
                    if retain_flags:
                        flags_output[(narrower_nsamples_window >= n_threshold) 
                                   & (narrower_flags_window == False)] = False
                        flags_output[(narrower_nsamples_window < n_threshold) 
                                   | (narrower_flags_window == True)] = True
                    else:
                        flags_output[(narrower_nsamples_window >= n_threshold)] \
                                    = False
                        flags_output[(narrower_nsamples_window < n_threshold)] \
                                    = True
                    
                    # Perform greedy flagging
                    if greedy:
                        if first == 'col':
                            # Flag all columns that exceed the greedy_threshold
                            col_indices = np.where(np.sum(flags_output, axis=0)
                                                   / Ntimes > greedy_threshold)
                            flags_output[:, col_indices] = True
                            
                            # Flag all remaining rows
                            remaining_rows = np.where(
                                                np.sum(flags_output, axis=1) \
                                                > len(list(col_indices[0])) )
                            flags_output[remaining_rows, :] = True
                            
                        else:
                            # Flag all rows that exceed the greedy_threshold
                            row_indices = np.where(
                                                np.sum(flags_output, axis=1) 
                                                / (spw[1]-spw[0]) \
                                                > greedy_threshold )
                            flags_output[row_indices, :] = True
                            
                            # Flag all remaining columns
                            remaining_cols = np.where(
                                                np.sum(flags_output, axis=0) \
                                                > len(list(row_indices[0])) )
                            flags_output[:, remaining_cols] = True
                            
                    # Update the UVData object's flag_array if inplace
                    if inplace:
                        dset.flag_array[bl_inds,0,spw[0]:spw[1],n] \
                            = flags_output
                    else:
                        uvd_new.flag_array[bl_inds,0,spw[0]:spw[1],n] \
                            = flags_output
                            
        if not inplace:
            uvdlist_new.append(uvd_new)
    
    # Return an updated list of UVData objects if not inplace
    if not inplace:
        return uvdlist_new

