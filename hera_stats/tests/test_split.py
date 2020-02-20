import numpy as np
import hera_stats as hs
import hera_pspec as hp
from pyuvdata import UVData
from hera_stats.data import DATA_PATH
import nose.tools as nt
import os
import unittest


class Test_split(unittest.TestCase):

    def setUp(self):
        # Load UVData object
        dfilename = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
        self.uvd = UVData()
        self.uvd.read_miriad(dfilename)
        
        # Load power spectra and container
        self.filepath = os.path.join(DATA_PATH, "uvp_data.h5")
        self.psc = hp.container.PSpecContainer(self.filepath, mode='r')
        self.uvp = self.psc.get_pspec("IDR2_1")[0]
        self.uvp_list = self.psc.get_pspec("IDR2_1")[0]
    
    def test_lst_blocks(self):
        # Test basic functionality
        uvps, lsts = hs.split.lst_blocks(self.uvp, blocks=2, 
                                         lst_range=(0., 2.*np.pi))
        nt.assert_equal(len(lsts), 2+1) # 2 blocks = 3 bin edges
        
        # Check error conditions
        nt.assert_raises(ValueError, hs.split.lst_blocks, self.uvp, 
                         2, (0., 100.)) # invalid LST range
        nt.assert_raises(TypeError, hs.split.lst_blocks, self.psc, 
                         2, (0., np.pi)) # invalid type
    
    def test_lst_stripes(self):
        # Test basic functionality
        # lst_stripes(uvp, stripes=2, width=1, lst_range=(0., 2.*np.pi))
        uvps = hs.split.lst_stripes(self.uvp, stripes=2, width=1, 
                                         lst_range=(0., 2.*np.pi))
        nt.assert_equal(len(uvps), 2)
        
        # Test that a different width also works
        uvps = hs.split.lst_stripes(self.uvp, stripes=2, width=2, 
                                         lst_range=(0., 2.*np.pi))
        nt.assert_equal(len(uvps), 2)
        
        # Check error conditions
        nt.assert_raises(ValueError, hs.split.lst_stripes, self.uvp, 
                         2, 1, (0., 100.)) # invalid LST range
        nt.assert_raises(TypeError, hs.split.lst_stripes, self.psc, 
                         2, 1, (0., np.pi)) # invalid type
    
    def test_split_ants(self):
        np.random.seed(0)
        uvpl = hs.split.split_ants([self.uvp], 1, verbose=True)
        nt.assert_raises(AssertionError, hs.split.split_ants, 20000)
        nt.assert_raises(AssertionError, hs.split.split_ants,
                        [self.uvp]*2, 1, True)

    def test_stripe_times(self):
        np.random.seed(0)
        uvp = hs.split.stripe_times(self.uvp, verbose=True)
        uvp = hs.split.stripe_times(self.uvp, 10.)
        uvp = hs.split.stripe_times(self.uvp, [10., 20.])
        nt.assert_equal(np.array(uvp).shape, (2, 2))
        nt.assert_raises(AssertionError, hs.split.stripe_times, np.array([]))

    def test_split_gha(self):
        np.random.seed(0)
        uvp = hs.split.split_gha([self.uvp], bins_list=[1])
        uvp = hs.split.split_gha(self.uvp, 
                                 bins_list=[np.linspace(228.2, 228.3, 1)], 
                                 specify_bins=True)  
        nt.assert_raises(AttributeError, hs.split.split_gha, self.uvp, 
                         [(0,1,2)], True)

    def test_omit_ants(self):
        np.random.seed(0)
        uvp = hs.split.omit_ants([self.uvp], [37, 38, 39])
        nt.assert_equals(np.asarray(uvp).shape, (1, 3))
        uvp = hs.split.omit_ants([self.uvp], 37)


if __name__ == "__main__":
    unittest.main()

