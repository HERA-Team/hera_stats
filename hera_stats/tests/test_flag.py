import numpy as np
import hera_stats as hs
from pyuvdata import UVData
from hera_stats.data import DATA_PATH
from hera_pspec.data import DATA_PATH as PSPEC_DATA_PATH
import os, sys
import nose.tools as nt
import unittest


class test_flag(unittest.TestCase):

    def setUp(self):
        self.datafile = os.path.join(DATA_PATH, "zen.odd.xx.LST.1.28828.uvOCRSA")
        self.uvd = UVData()
        self.uvd.read_miriad(self.datafile)
        
        self.dfiles = ['zen.even.xx.LST.1.28828.uvOCRSA', 
                       'zen.odd.xx.LST.1.28828.uvOCRSA']
        self.baseline = (38, 68, 'xx')
        
        # Load datafiles into UVData objects
        self.d = []
        for dfile in self.dfiles:
            _d = UVData()
            _d.read_miriad(os.path.join(PSPEC_DATA_PATH, dfile))
            self.d.append(_d)
    
    def tearDown(self):
        pass
    
    def test_apply_random_flags(self):
        
        # Check that basic functionality works
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
        
        # Check that in-place works
        uvd1 = hs.flag.apply_random_flags(self.uvd, 0.3, seed=10)
        hs.flag.apply_random_flags(self.uvd, 0.3, seed=10, inplace=True)
        np.testing.assert_almost_equal(uvd1.flag_array, self.uvd.flag_array)
    
    
    def test_flag_channels(self):
        """
        Test the channel-flagging function
        """
        # Make sure that flagging is occurring
        chans = [(200, 451), (680, 881)]
        col_flagged_uvds = [hs.flag.flag_channels(_d, chans, inplace=False) 
                            for _d in self.d]
        
        for i in range(len(self.d)):
            
            # Check that outside the spw ranges, flags are all equal
            nt.assert_true(np.array_equal( \
                col_flagged_uvds[i].get_flags((38, 68, 'xx'))[:, :200], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, :200]))
            nt.assert_true(np.array_equal( \
                col_flagged_uvds[i].get_flags((38, 68, 'xx'))[:, 451:680], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, 451:680]))
            nt.assert_true(np.array_equal( \
                col_flagged_uvds[i].get_flags((38, 68, 'xx'))[:, 881:], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, 881:]))
            
            # Check that inside the ranges, everything is flagged
            nt.assert_true(np.array_equal( \
                col_flagged_uvds[i].get_flags((38, 68, 'xx'))[:, 200:451], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, 200:451]))
            nt.assert_true(np.array_equal( \
                col_flagged_uvds[i].get_flags((38, 68, 'xx'))[:, 680:881], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, 680:881]))
            
            # Check that inplace objects match in important areas
            nt.assert_true(np.array_equal( \
                col_flagged_uvds[i].get_data((38, 68, 'xx')), \
                                          self.d[i].get_data((38, 68, 'xx'))))
            nt.assert_true(np.array_equal( \
                col_flagged_uvds[i].get_nsamples((38, 68, 'xx')), \
                                          self.d[i].get_nsamples((38, 68, 'xx'))))
