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
    
    def test_hour_angle(self):
        np.random.seed(0)
        uvp = hs.split.hour_angle([self.uvp], bins_list=[1])
        uvp = hs.split.hour_angle(self.uvp, 
                                 bins_list=[np.linspace(228.2, 228.3, 1)], 
                                 specify_bins=True)  
        nt.assert_raises(AttributeError, hs.split.hour_angle, self.uvp, 
                         [(0,1,2)], True)

    def test_omit_ants(self):
        np.random.seed(0)
        uvp = hs.split.omit_ants([self.uvp], [37, 38, 39])
        nt.assert_equals(np.asarray(uvp).shape, (1, 3))
        uvp = hs.split.omit_ants([self.uvp], 37)
    
    def test_blps_by_antnum(self):
        # Get redundant baseline-pair groups
        blps, lens, angs = self.uvp.get_red_blpairs()
        
        # Split into groups without (A) and with (B) repeated antennas
        blps_a, blps_b = hs.split.blps_by_antnum(blps, split='norepeat')
        for grp in blps_a:
            for blp in grp:
                nt.assert_equal(np.unique(blp).size, 4) # expects 4 unique ants
        for grp in blps_b:
            for blp in grp:
                nt.assert_not_equal(np.unique(blp).size, 4) # expects <4 unique ants
        
        # Split into groups without (A) and with (B) auto-bl pairs
        blps_a, blps_b = hs.split.blps_by_antnum(blps, split='noautos')
        for grp in blps_a:
            for blp in grp:
                nt.assert_not_equal(sorted(blp[0]), sorted(blp[1])) # no autos
        for grp in blps_b:
            for blp in grp:
                nt.assert_equal(sorted(blp[0]), sorted(blp[1])) # expect autos
        
        # Test error checks
        # Invalid split type
        nt.assert_raises(ValueError, hs.split.blps_by_antnum, blps, 'xx')
        
        # blps not passed as list of lists
        nt.assert_raises(TypeError, hs.split.blps_by_antnum, blps[0], 'noautos')
        
if __name__ == "__main__":
    unittest.main()

