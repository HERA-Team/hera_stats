import numpy as np
import hera_pspec as hp
import hera_stats as hs
from pyuvdata import UVData
from hera_pspec.data import DATA_PATH as PSPEC_DATA_PATH
import nose.tools as nt
import os, sys
import unittest

def check_if_sums_are_close(uvd1, uvd2, redgrps):
    # Get the redgrps
    close = []
    for i,grp in enumerate(redgrps):            
        sum_uvd1 = np.sum(get_data_redgrp(uvd1, grp), axis=1)
        sum_uvd2 = np.sum(get_data_redgrp(uvd2, grp), axis=1)
        close.append( np.allclose(sum_uvd1, sum_uvd2) )
    close = np.array(close)
    return np.all(close)


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
        # Check if the sums of all baselines are close before and after shuffle
        redgrps, bl_lens, bl_angs = hp.utils.get_reds(self.uvd, pick_data_ants=True)
        uvd_shuffled = hs.shuffle.shuffle(self.uvd, redgrps)        
        sums_close = check_if_sums_are_close(self.uvd, uvd_shuffled, redgrps)
        nt.assert_equal(sums_close, True)
        
        
if __name__ == "__main__":
    unittest.main()

