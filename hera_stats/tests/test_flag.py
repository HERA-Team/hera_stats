import numpy as np
import hera_stats as hs
from pyuvdata import UVData
from hera_stats.data import DATA_PATH
import os, sys
import nose.tools as nt
import unittest

class test_flag():

    def setUp(self):
        self.datafile = os.path.join(DATA_PATH, "zen.odd.xx.LST.1.28828.uvOCRSA")
        self.uvd = UVData()
        self.uvd.read_miriad(self.datafile)
    
    def tearDown(self):
        pass
    
    def test_apply_random_flags(self):
        
        # Check basic functionality works
        ffrac = 0.6
        uvd_new = hs.flag.apply_random_flags(self.uvd, ffrac, inplace=False, 
                                             zero_flagged_data=False, seed=10)
        flags_new = np.sum(uvd_new.flag_array.flatten()) # no. flags in uvd_new
        flags_sum = self.uvd.data_array.size*ffrac \
                  + np.sum(self.uvd.flag_array.flatten()) # max flags in uvd_new
        nt.assert_true(flags_sum >= flags_new)
        
        # Try zeroing flagged data (should be zero everywhere under the mask)
        uvd_zeroed = hs.flag.apply_random_flags(self.uvd, 0.1, inplace=False, 
                                                zero_flagged_data=True)
        zeroed = uvd_zeroed.data_array[uvd_zeroed.flag_array]
        np.testing.assert_almost_equal( np.sum(np.abs(zeroed)), 0. )
        
        # Check that random seed is respected
        uvd1 = hs.flag.apply_random_flags(self.uvd, 0.3, seed=10)
        uvd2 = hs.flag.apply_random_flags(self.uvd, 0.3, seed=10)
        np.testing.assert_almost_equal(uvd1.flag_array, uvd2.flag_array)
        
