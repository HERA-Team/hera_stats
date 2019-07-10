import hera_stats as hs
from hera_pspec.data import DATA_PATH
from pyuvdata import UVData
import numpy as np
import os
import nose.tools as nt
import unittest

class test_utils(unittest.TestCase):
    
    def setUp(self):
        
        self.dfiles = ['zen.even.xx.LST.1.28828.uvOCRSA', 
                       'zen.odd.xx.LST.1.28828.uvOCRSA']
        self.baseline = (38, 68, 'xx')
        
        # Load datafiles into UVData objects
        self.d = []
        for dfile in self.dfiles:
            _d = UVData()
            _d.read_miriad(os.path.join(DATA_PATH, dfile))
            self.d.append(_d)
        
        # data to use when testing the plotting function
        self.data_list = [self.d[0].get_flags(38, 68, 'xx'), 
                          self.d[1].get_flags(38, 68, 'xx')]
        
    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_functions(self):
        hs.utils.plt_layout(10)

        degs = [d - 50 for d in range(100)]
        hs.utils.bin_wrap(degs, 10)
        hs.utils.bin_wrap(degs, 200)

        nt.assert_true(hs.utils.is_in_wrap(350, 10, 0.2))
        nt.assert_false(hs.utils.is_in_wrap(350, 10, 11))
    
    def test_stacked_array(self):
        """
        testing the array stacking function
        """
        key = (38, 68, 'xx')
        flags_list = [uvd.get_flags(key) for uvd in self.d]
        long_array_flags = hs.utils.stacked_array(flags_list)

        # make sure no. rows in output = sum of no. rows in each input array
        nt.assert_equal(long_array_flags.shape[0], sum([flag_array.shape[0] \
                                       for flag_array in flags_list]))
        
        # Ensure that the number of columns is unchanged
        for flag_array in flags_list:

            nt.assert_equal(long_array_flags.shape[1], flag_array.shape[1])
        
        # Ensure that arrays are stacked in order as expected
        nt.assert_true(np.array_equal( \
            long_array_flags[0 : flags_list[0].shape[0], :], flags_list[0]))
        nt.assert_true(np.array_equal( \
            long_array_flags[ flags_list[0].shape[0] : flags_list[0].shape[0] + \
                             flags_list[1].shape[0], :], flags_list[1]))
                             
