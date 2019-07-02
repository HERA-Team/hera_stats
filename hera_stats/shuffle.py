import numpy as np
from pyuvdata import UVData
import copy

def shuffle_data_redgrp(uvd, redgrps):
    """
    Construct a new set of visibilities by randomly shuffling samples between 
    baselines in a redundant group.
    
    Different random shuffles are performed at each frequency, time, and 
    polarization. This creates a new set of visibilities that are made up of 
    samples from a random combination of redundant baselines, but that do not 
    mix or shuffle times, frequencies or polarizations.
    
    The samples are shuffled _without_ replacement, so each sample is only ever 
    used once.
    
    Example:
      Original visibility (fixed time and pol, for freq. channels 1, 2, 3...):
      
        bl_a = |a1|a2|a3|a4|...
        bl_b = |b1|b2|b3|b4|...
        bl_c = |c1|c2|c3|c4|...
    
      Shuffled visibility:
      
        new_a = |b1|a2|c3|b4|...
        new_b = |c1|c2|a3|a4|...
        new_c = |a1|b2|b3|c4|...
    
    Parameters
    ----------
    uvd : UVData object 
        Input visibilities.
        
    redgrps : list of lists of bls
        List of redundant baseline groups.
    
    Returns
    -------
    uvd_new : UVData object
        Copy of `uvd` with baseline-shuffled visibilities.
    """
    # Check validity of inputs
    if not isinstance(redgrps, list) or not isinstance(redgrps[0], list):
        raise TypeError("redgrps must be a list of lists of baseline identifiers.")
    assert isinstance(uvd, UVData), "uvd must be a UVData object."
    
    # Make a copy of the UVPSpec object
    uvd_new = copy.deepcopy(uvd)
    
    # Get data types of arrays
    dtype = uvd.data_array.dtype # data array
    ftype = uvd.flag_array.dtype # flag array
    ntype = uvd.nsample_array.dtype # nsample array
    
    # Shuffle baselines in each redgrp
    for grp in redgrps:
        # Create array to store ushuffled data
        dshape = (uvd.Ntimes, len(grp), uvd.Nfreqs, uvd.Npols)
        orig_data = np.zeros(dshape, dtype=dtype)
        orig_flag = np.zeros(dshape, dtype=ftype)
        orig_nsamp = np.zeros(dshape, dtype=ntype)
        
        # Get data for each baseline in the group
        for b, key in enumerate(grp):
            orig_data[:,b,:,:] = uvd.get_data(key, squeeze='none')[:,0,:,:]
            orig_flag[:,b,:,:] = uvd.get_flags(key, squeeze='none')[:,0,:,:]
            orig_nsamp[:,b,:,:] = uvd.get_nsamples(key, squeeze='none')[:,0,:,:]

        # Create array to store shuffled data
        shuf_data = np.zeros(dshape, dtype=dtype)
        shuf_flag = np.zeros(dshape, dtype=ftype)
        shuf_nsamp = np.zeros(dshape, dtype=ntype)
        
        # Loop over times and freqs
        for p in range(uvd.Npols):
            for t in range(uvd.Ntimes):
                for i in range(uvd.Nfreqs):
                    # Randomly shuffle data between bls in redundant group
                    # (for each time, frequency, and polarization)
                    idxs = np.random.permutation(np.arange(len(grp)))
                    shuf_data[t,:,i,p] = orig_data[t,idxs,i,p]
                    shuf_flag[t,:,i,p] = orig_flag[t,idxs,i,p]
                    shuf_nsamp[t,:,i,p] = orig_nsamp[t,idxs,i,p]
        
        # Put shuffled data into new UVData object with the same shape etc.
        for k in range (len(grp)):
            # Get idxs for each bl in the group
            ant1, ant2 = grp[k]
            idxs = uvd.antpair2ind(ant1, ant2)
            uvd_new.data_array[idxs][:,0,:,:] = shuf_data[:,k,:,:]
            uvd_new.flag_array[idxs][:,0,:,:] = shuf_flag[:,k,:,:]
            uvd_new.nsample_array[idxs][:,0,:,:] = shuf_nsamp[:,k,:,:]
        
    return uvd_new
   
