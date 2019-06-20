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
    
    def test_estimate_noise_rms(self):
        
        # Get list of unique bls
        bls = [bl for bl in np.unique(self.uvd.baseline_array)]
        
        # Check that basic functionality works
        rms, rms_mod = hs.noise.estimate_noise_rms(self.uvd, bls, 
                                                   fit_poly=True, order=2)
        nt.assert_true(np.all(np.isfinite(rms_mod)))
        
        # Check that not fitting polynomial is sensible too
        rms = hs.noise.estimate_noise_rms(self.uvd, bls, fit_poly=False)
