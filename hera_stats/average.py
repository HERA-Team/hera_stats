
import numpy as np
from .utils import trim_empty_bls

def average_spectra_cumul(uvp, blps, spw, polpair, mode='time', min_samples=1, 
                          shuffle=False, time_avg=True, verbose=False):
    """
    Cumulatively average a set of delay spectra as a function of time 
    or blpair. The cumulative average can be performed sequentially, 
    e.g. in time order, or in a random (shuffled) order.
    
    Parameters
    ----------
    uvp : UVPSpec
        Set of power spectra to be cumulatively averaged.
    
    blps : list of ints or tuples
        List of blpair ints or tuples. Only one redundant set of 
        blpairs should be passed at once.
    
    spw, polpair : int, str
        Spectral window ID (integer) and polarization-pair (integer or str) of 
        the power spectra to average.
    
    mode : str, optional
        Whether to cumulatively average in time or blpair. The other 
        dimension will be averaged over non-cumulatively. (See the 
        'time_avg' kwarg below for other behaviors.) Possible options 
        are 'time' or 'blpair'. Default: 'time'.
    
    min_samples : int, optional
        Minimum number of samples to allow in the cumulative average. 
        Default: 1.
    
    shuffle : bool, optional
        Whether to randomly shuffle the order of the cumulative 
        averaging, or to keep it in order. Default: False.
    
    time_avg : bool, optional
        Whether to average over times. This option is only used if 
        mode='blpair'; otherwise it will be ignored. Default: True.
    
    verbose : bool, optional
        Whether to print status messages as the cumulative averaging 
        progresses. Default: False.
    
    Returns
    -------
    ps : array_like
        Cumulative averages of delay spectra, in a 2D array of shape 
        (Nsamp, Ndelay), where Nsamp = Ntimes or Nblpairs.
    
    dly : array_like
        Power spectrum delay modes (in s).
    
    n_samples : array_like
        Number of times or blpairs that went into each average in 
        the ps array.
    """
    # Check for valid mode
    if mode not in ['time', 'blpair']:
        raise ValueError("mode must be either 'time' or 'blpair'.")
    
    # Check for valid blpair format
    if not (   isinstance(blps[0], tuple) 
            or isinstance(blps[0], (int, np.integer, np.int))):
        raise TypeError("blps must be a list of baseline-pair tuples or "
                        "integers only.")
    
    if mode == 'time':
        # Get unique times from UVPSpec object
        avail_times = np.unique(uvp.time_avg_array)
        if avail_times.size <= min_samples:
            raise ValueError("min_samples is larger than or equal to the number "
                             "of available samples.")
        if verbose: print("Unique time samples:", avail_times.size)
        
        # Either shuffle or sort available times
        if shuffle:
            np.random.shuffle(avail_times)
        else:
            avail_times.sort()
        
        # Perform initial downselect of times and blpairs (and make copy)
        uvp_t = uvp.select(times=avail_times[:], blpairs=blps, inplace=False, 
                           spws=[spw,], polpairs=[polpair,])
        
        # Loop over times (in reverse size order)
        avg_spectra = []; n_samples = []
        for t in range(avail_times.size, min_samples, -1):
            if verbose: print("  Sample %d / %d" % (t, avail_times.size))
            
            # Select subset of samples (good for performance)
            uvp_t.select(times=avail_times[:t], blpairs=blps)

            # Average over blpair and time (use time_avg_array as the time key, 
            # as this is what UVPSPec.select() uses)
            _avg = uvp_t.average_spectra([blps,], time_avg=True, inplace=False)
            n_samples.append( np.unique(uvp_t.time_avg_array).size )
            
            # Unpack data into array (spw=0 since we only selected one spw)
            ps = _avg.get_data((0, _avg.blpair_array[0], polpair)).flatten()
            avg_spectra.append(ps)
            dly = _avg.get_dlys(0)
            
    else:
        # Perform initial downselect of blpairs (and make copy)
        uvp_b = uvp.select(blpairs=blps, inplace=False, spws=[spw,], 
                           polpairs=[polpair,])
        avail_blps = np.unique(uvp_b.blpair_array) # available blpairs
        if avail_blps.size <= min_samples:
            raise ValueError("min_samples is larger than or equal to the number "
                             "of available samples.")
        if verbose: print("Unique blpairs:", avail_blps.size)
        
        # Either shuffle or sort available blpairs
        if shuffle:
            np.random.shuffle(avail_blps)
        else:
            avail_blps.sort()
        
        # Loop over times (in reverse size order)
        avg_spectra = []; n_samples = []
        for b in range(avail_blps.size, min_samples, -1):
            if verbose and b % 10 == 0: 
                print("  Sample %d / %d" % (b, avail_blps.size))
            
            # Select subset of samples (good for performance)
            uvp_b.select(blpairs=avail_blps[:b])
            
            # Average over blpair and time
            _avg = uvp_b.average_spectra([list(avail_blps[:b]),], 
                                         time_avg=time_avg, inplace=False)
            n_samples.append( np.unique(uvp_b.blpair_array).size )
            
            # Unpack data into array (spw=0 since we only selected one spw)
            ps = _avg.get_data((0, _avg.blpair_array[0], polpair))
            if time_avg: ps = ps.flatten() # remove extra dim if time_avg=true
            avg_spectra.append(ps)
            dly = _avg.get_dlys(0)
        
    
    # Convert stored data into arrays and return
    avg_spectra = np.array(avg_spectra)
    n_samples = np.array(n_samples)
    return avg_spectra, dly, n_samples


def redundant_diff(uvd, bls, pol, return_mean=False):
    """
    Calculate the difference between all baselines in a redundant group and 
    the mean of the redundant group (as a fn. of frequency and time).
    
    Empty (fully flagged) baselines are excluded from the average.
    
    N.B. The averaging does not currently take into account data weights or 
    number of samples.
    
    Parameters
    ----------
    uvd : UVData
        UVData object containing the data that will be averaged and differenced.
    
    bls : list of tuple or int
        List of baseline tuples or integers to be treated as a group. The mean 
        will be calculated over all 
    
    pol : str
        Which polarization to extract from the UVData file.
    
    return_mean : bool, optional
        If True, return the mean over the redundant group too. Default: False
    
    Returns
    -------
    bls : list of baseline tuple or int
        List of baselines that were kept in the average.
         
    diffs : list of array_like
        List of arrays of differences between each baseline and the group mean.
    
    mean : array_like, optional
        Mean over data from all non-flagged baselines, as a function of freq. 
        and time. Only retuend if `return_mean` is True.
    """
    # Check bls
    assert isinstance(bls, list), "bls must be a list of baseline ints/tuples"
    
    # Clean out empty baselines from list
    trimmed_bls = trim_empty_bls(uvd, bls)[0] # returns list of lists (unpack)
    
    # Calculate mean
    grp_mean = 0
    for bl in trimmed_bls:
        if isinstance(bl, (int, np.int)):
            bl = uvd.baseline_to_antnums(bl)
        grp_mean += uvd.get_data(bl, pol)
    grp_mean /= float(len(trimmed_bls))
    
    # Calculate differences
    diffs = []
    for bl in trimmed_bls:
        if isinstance(bl, (int, np.int)):
            bl = uvd.baseline_to_antnums(bl)
        diff = uvd.get_data(bl, pol) - grp_mean
        diffs.append(diff)
    
    if return_mean:
        return bls, diffs, grp_mean
    return bls, diffs

