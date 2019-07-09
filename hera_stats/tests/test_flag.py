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

    def test_construct_factorizable_mask(self):
        """
        Test factorizable mask generator function.
        """
        # Test unflagging
        unflagged_uvdlist = hs.flag.construct_factorizable_mask(
                                            self.d, spw_ranges=[(0,1024)], 
                                            unflag=True, inplace=False)
        for uvd in unflagged_uvdlist:
            unflagged_mask = uvd.get_flags((38, 68, 'xx'))
            nt.assert_equal(np.sum(unflagged_mask), 0)
        
        # Ensure that greedy flagging works as expected in extreme cases
        allflagged_uvdlist = hs.flag.construct_factorizable_mask(
                                            self.d, spw_ranges=[(0,1024)], 
                                            greedy_threshold=0.0001, 
                                            first='row', inplace=False)
        for uvd in allflagged_uvdlist:
            flagged_mask = uvd.get_flags((38, 68, 'xx'))
        
        # Everything should be flagged since greedy_threshold is extremely low
            nt.assert_equal(np.sum(flagged_mask), \
                            np.sum(np.ones(flagged_mask.shape)))
        
        # Ensure that n_threshold parameter works as expected in extreme cases
        allflagged_uvdlist2 = hs.flag.construct_factorizable_mask(
                                            self.d, spw_ranges=[(0,1024)], 
                                            n_threshold=35, 
                                            first='row', inplace=False)
        for uvd in allflagged_uvdlist2:
            flagged_mask = uvd.get_flags((38, 68, 'xx'))
            nt.assert_equal(np.sum(flagged_mask), \
                            np.sum(np.ones(flagged_mask.shape)))
        
        # Ensure that greedy flagging is occurring within the intended spw: 
        greedily_flagged_uvdlist = hs.flag.construct_factorizable_mask(
                                            self.d, n_threshold=6, 
                                            greedy_threshold=0.35, 
                                            first='col',
                                            spw_ranges=[(0, 300), (500, 700)], 
                                            inplace=False)
        
        for i in range(len(self.d)):
            # Check that outside the spw range, flags are all equal
            nt.assert_true(np.array_equal( \
                greedily_flagged_uvdlist[i].get_flags((38, 68, 'xx'))[:, 300:500], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, 300:500]))
            nt.assert_true(np.array_equal( \
                greedily_flagged_uvdlist[i].get_flags((38, 68, 'xx'))[:, 700:], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, 700:]))
            
            # Flags are actually retained
            original_flags_ind = np.where(
                                    self.d[i].get_flags((38, 68, 'xx')) == True)
            new_flags = greedily_flagged_uvdlist[i].get_flags((38, 68, 'xx'))
            old_flags = self.d[i].get_flags((38, 68, 'xx'))
            nt.assert_true(np.array_equal( \
                new_flags[original_flags_ind], old_flags[original_flags_ind]))
            
            # Check that inplace objects match in important areas
            nt.assert_true(np.array_equal( \
                greedily_flagged_uvdlist[i].get_data((38, 68, 'xx')), \
                                          self.d[i].get_data((38, 68, 'xx'))))
            nt.assert_true(np.array_equal( \
                greedily_flagged_uvdlist[i].get_nsamples((38, 68, 'xx')), \
                                          self.d[i].get_nsamples((38, 68, 'xx'))))
            
            # Make sure flags are actually independent in each spw
            masks = [new_flags[:, 0:300], new_flags[:, 500:700]]
            for mask in masks:
                Nfreqs = mask.shape[1]
                Ntimes = mask.shape[0]
                N_flagged_rows = np.sum( \
                    1*(np.sum(mask, axis=1)/Nfreqs > 0.999999999))
                N_flagged_cols = np.sum( \
                    1*(np.sum(mask, axis=0)/Ntimes > 0.999999999))
                nt.assert_true(int(np.sum( \
                    mask[np.where(np.sum(mask, axis=1)/Nfreqs < 0.99999999)]) \
                                   /(Ntimes-N_flagged_rows)) == N_flagged_cols)

