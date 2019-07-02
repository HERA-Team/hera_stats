
import numpy as np
import copy

def shuffle (uvd, redgrps):
    # Get the redgrps
    """
    Shuffle all the baselines which belong to the same reduntant group in the uvdata along each Nfreq and
    Ntimes.
    
    Parameters
    ----------
    uvd : UVData object 
    
    redgrps : list 
        List of redundant baseline (antenna-pair) groups
    
    Returns
    -------
    uvd_new : UVData object 
        UVData object with shuffled baselines 
    """
    # Get data type of array
    dtype = uvd.data_array.dtype
    
    # make a copy of the UVPSpec object
    uvd_new = copy.deepcopy(uvd)
    
    # Shuffle baselines in each redgrp
    for grp in redgrps:
        # Create a 3D zero array for the ushuffled data
        unshuffled_data = np.zeros(((uvd.Ntimes , len(grp) , uvd.Nfreqs)), dtype=dtype)

        # For the specific redgrps, get the data from same baesline grps with same vector and form a 3D array
        for b, key in enumerate(grp):
            unshuffled_data[:,b,:] = uvd.get_data(key)[:,:]

        # Create a zero array for the suffled data
        shuffled_data = np.zeros(unshuffled_data.shape, dtype=dtype)

        for t in range(uvd.Ntimes):
            # loop over Ntimes
            # shuffle the data along the each Nfreqs
            for i in range(uvd.Nfreqs):
                #y = np.random.permutation(x[:,i])
                shuffled_data[t,:,i] = np.random.permutation(unshuffled_data[t,:,i])
        
        # put shuffled data into new UVData object which have the same baseline
        for k in range (len(grp)):
            # Find out the corresponding idxs of the particular antpair in the baselinestimes array 
            ant1, ant2 = grp[k]
            idxs = uvd.antpair2ind(ant1, ant2)
            uvd_new.data_array[idxs][:,0,:,0] = shuffled_data[:,k,:]
    
    # return     
    return uvd_new
   
