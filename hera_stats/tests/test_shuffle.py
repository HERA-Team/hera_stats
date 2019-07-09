import numpy as np
import hera_pspec as hp
import hera_stats as hs
from pyuvdata import UVData
from hera_pspec.data import DATA_PATH as PSPEC_DATA_PATH
import nose.tools as nt
import os, sys
import unittest

def get_data_redgrp(uvd, redgrp, array='data'):
    """
    Get data from all bls in a redundant group and output into ndarray of shape 
    (Ntimes, Nbls, Nfreqs, Npols).
    """
    # Get data type of array
    if array == 'data':
        dtype = uvd.data_array.dtype
        fn = uvd.get_data
    elif array == 'flag':
        dtype = uvd.flag_array.dtype
        fn = uvd.get_flags
    elif array == 'nsamp':
        dtype = uvd.nsample_array.dtype
        fn = uvd.get_nsamples
    else:
        raise ValueError("array '%s' not recognized, must be 'data', 'flag', "
                         "or 'nsamp'." % array)
    
    # Create a 3D zero array
    dshape = (uvd.Ntimes, len(redgrp), uvd.Nfreqs, uvd.Npols)
    redgrp_arr = np.zeros(dshape, dtype=dtype)

    # Get the data for each baseline in the redgrp
    for b, key in enumerate(redgrp):
        redgrp_arr[:,b,:,:] = fn(key, squeeze='none')[:,0,:,:]
    return redgrp_arr


def check_if_sums_are_close(uvd1, uvd2, redgrps, array='data'):
    """
    Check whether the sum of the data, flags, or nsamples in two UVData objects 
    is the same within each redgrp.
    """
    close = []
    for i, grp in enumerate(redgrps):            
        sum_uvd1 = np.sum(get_data_redgrp(uvd1, grp, array=array), axis=1)
        sum_uvd2 = np.sum(get_data_redgrp(uvd2, grp, array=array), axis=1)
        close.append( np.allclose(sum_uvd1, sum_uvd2) )
    close = np.array(close)
    return np.all(close)

def check_if_the_data_are_same(uvd1, uvd2, array='data'):
    """
    Check whether the two data_array in two UVDate object is the same.
    """
    close = np.allclose(uvd1.data_array, uvd2.data_array)
    return close

class test_shuffle():

    def setUp(self):
        # Load example UVData object
        self.filepath = os.path.join(PSPEC_DATA_PATH, 
                                     "zen.even.std.xx.LST.1.28828.uvOCRSA")
        self.uvd = UVData()
        self.uvd.read_miriad(self.filepath)
    
    def tearDown(self):
        pass
    
    def test_shuffle(self):
        
        # Get redundant groups
        redgrps, bl_lens, bl_angs = hp.utils.get_reds(self.uvd, 
                                                      pick_data_ants=True)
        
        # Shuffle samples between bls in each redgrp
        uvd_shuffled = hs.shuffle.shuffle_data_redgrp(self.uvd, redgrps)
        
        # Check if summing the data over bls in each redgrp is invariant after 
        # shuffling (sum should not have changed, as we are doing sampling 
        # without replacement -> all samples conserved)
        for arr in ['data', 'flag', 'nsamp']:
            isclose = check_if_sums_are_close(self.uvd, uvd_shuffled, redgrps, 
                                              array=arr)
            nt.assert_equal(isclose, True)
            
        for arr in ['data', 'flag', 'nsamp']:
            data_isclose = check_if_the_data_are_same(self.uvd, uvd_shuffled, array=arr)
            nt.assert_equal(data_isclose, False)
        
if __name__ == "__main__":
    unittest.main()

