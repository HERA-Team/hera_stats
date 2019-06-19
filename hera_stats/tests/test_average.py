import numpy as np
import hera_pspec as hp
import hera_stats as hs
from hera_stats.data import DATA_PATH
import nose.tools as nt
import os, sys
import unittest

class test_average():

    def setUp(self):
        # Load pre-computed power spectra
        self.filepath = os.path.join(DATA_PATH, "uvp_data.h5")
        pc = hp.container.PSpecContainer(self.filepath, mode='r')
        self.uvp = pc.get_pspec("IDR2_1")[0]
    
    def tearDown(self):
        pass
    
    def test_average_spectra_cumul(self):
        
        # Get list of unique blps
        blps = [blp for blp in np.unique(self.uvp.blpair_array)]
        
        # Check basic operation of cumulative-in-time mode
        ps, dly, nsamples = hs.average.average_spectra_cumul(
                                self.uvp, blps, spw=0, polpair=('xx', 'xx'), 
                                mode='time', min_samples=1, shuffle=False, 
                                time_avg=True, verbose=False)
        nt.assert_equal(ps.shape, (self.uvp.Ntimes-1, self.uvp.Ndlys))
        np.testing.assert_almost_equal(dly, self.uvp.get_dlys(0))
        
        # Same, but with shuffling and without time averaging (should be 
        # ignored, since mode != 'blpair')
        ps, dly, nsamples = hs.average.average_spectra_cumul(
                                self.uvp, blps, spw=0, polpair=('xx', 'xx'), 
                                mode='time', min_samples=1, shuffle=True, 
                                time_avg=False, verbose=False)
        nt.assert_equal(ps.shape, (self.uvp.Ntimes-1, self.uvp.Ndlys))
        
        # Check basic operation of cumulative-in-blpair mode
        ps, dly, nsamples = hs.average.average_spectra_cumul(
                                self.uvp, blps, spw=0, polpair=('xx', 'xx'), 
                                mode='blpair', min_samples=1, shuffle=False, 
                                time_avg=True, verbose=False)
        nt.assert_equal(ps.shape, (self.uvp.Nblpairs-1, self.uvp.Ndlys))
        
        # Same as above, but without time averaging
        ps, dly, nsamples = hs.average.average_spectra_cumul(
                                self.uvp, blps, spw=0, polpair=('xx', 'xx'), 
                                mode='blpair', min_samples=1, shuffle=False, 
                                time_avg=False, verbose=False)
        nt.assert_equal(ps.shape, 
                        (self.uvp.Nblpairs-1, self.uvp.Ntimes, self.uvp.Ndlys))
        
        # Check that min_samples works
        min_samp = 4
        ps, dly, nsamples = hs.average.average_spectra_cumul(
                                self.uvp, blps, spw=0, polpair=('xx', 'xx'), 
                                mode='time', min_samples=min_samp, shuffle=False, 
                                time_avg=True, verbose=False)
        nt.assert_equal(ps.shape, (self.uvp.Ntimes-min_samp, self.uvp.Ndlys))


if __name__ == "__main__":
    unittest.main()

