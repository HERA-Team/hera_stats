import numpy as np
import hera_pspec as hp
import hera_stats as hs
from hera_stats.data import DATA_PATH
from pyuvdata import UVData
import nose.tools as nt
import os, sys, copy
import unittest

class test_average():

    def setUp(self):
        
        # Load pre-computed power spectra
        self.filepath = os.path.join(DATA_PATH, "uvp_data.h5")
        pc = hp.container.PSpecContainer(self.filepath, mode='r')
        self.uvp = pc.get_pspec("IDR2_1")[0]
        
        # Load example UVData
        self.datafile = os.path.join(DATA_PATH, "zen.odd.xx.LST.1.28828.uvOCRSA")
        self.uvd = UVData()
        self.uvd.read_miriad(self.datafile)
    
    def tearDown(self):
        # Remove output data file
        if os.path.exists("./out.h5"):
            os.remove("./out.h5")
    
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
        
        # Now verbose, and with shuffle
        ps, dly, nsamples = hs.average.average_spectra_cumul(
                                self.uvp, blps, spw=0, polpair=('xx', 'xx'), 
                                mode='blpair', min_samples=1, shuffle=True, 
                                time_avg=True, verbose=True)
        
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
        
        # Check for expected errors
        
        # Invalid mode
        nt.assert_raises(ValueError, hs.average.average_spectra_cumul, 
                         self.uvp, blps, spw=0, polpair=('xx', 'xx'), 
                         mode='something', shuffle=False, time_avg=True, 
                         verbose=False)
        
        # Invalid blps
        nt.assert_raises(TypeError, hs.average.average_spectra_cumul, 
                         self.uvp, [[(1,2)],], spw=0, polpair=('xx', 'xx'), 
                         mode='time', shuffle=False, time_avg=True, 
                         verbose=False)
        
        # If min_samples > no. samples for time and blpair modes
        nt.assert_raises(ValueError, hs.average.average_spectra_cumul, 
                         self.uvp, blps, spw=0, polpair=('xx', 'xx'), 
                         mode='time', min_samples=1000000000, shuffle=False, 
                         time_avg=True, verbose=False)
        nt.assert_raises(ValueError, hs.average.average_spectra_cumul, 
                         self.uvp, blps, spw=0, polpair=('xx', 'xx'), 
                         mode='blpair', min_samples=1000000000, shuffle=False, 
                         time_avg=True, verbose=False)
       
        # Check that spectra with misaligned times can be averaged
        uvd1 = self.uvd.select(times=np.unique(self.uvd.time_array)[::2], 
                               inplace=False)
        uvd2 = self.uvd.select(times=np.unique(self.uvd.time_array)[1::2], 
                               inplace=False)
        if os.path.exists("./out.h5"):
            os.remove("./out.h5")
        blps_toffset = [((38, 68), (37, 38)), ((38, 68), (52, 53))]
        
        fname = "./out.h5"
        ds = hp.pspecdata.pspec_run([uvd1, uvd2],
                                         fname,
                                         blpairs=blps_toffset,
                                         verbose=False, overwrite=True, 
                                         spw_ranges=[(50, 100)], 
                                         rephase_to_dset=0,
                                         broadcast_dset_flags=True, 
                                         time_thresh=0.3)
        
        psc = hp.PSpecContainer(fname, mode='r')
        uvp_toffset = psc.get_pspec(psc.groups()[0])[0]
        
        # Check basic operation of cumulative-in-time mode with offset times
        ps, dly, nsamples = hs.average.average_spectra_cumul(
                                uvp_toffset, blps_toffset, spw=0, 
                                polpair=('xx', 'xx'), 
                                mode='time', min_samples=1, shuffle=False, 
                                time_avg=True, verbose=False)
        
    def test_redundant_diff(self):
        
        # Get list of unique bls
        bls = self.uvd.get_antpairs()
        pol = self.uvd.polarization_array[0]
        
        # Try to run function
        _bls, diffs = hs.average.redundant_diff(self.uvd, bls, pol)
        nt.assert_equal(len(_bls), len(diffs))
        
        # Re-run, no returning the mean
        _bls, diffs, mean = hs.average.redundant_diff(self.uvd, bls, pol, 
                                                      return_mean=True)
        nt.assert_equal(mean.shape, diffs[0].shape)
    
    def test_redundant_diff_summary(self):
        
        # Get redundant baselines contained in test UVData object
        red_bls, lens, angs = hp.utils.get_reds(self.uvd, bl_error_tol=1.0, 
                                                add_autos=False,
                                                bl_len_range=(12., 100.), 
                                                bl_deg_range=(0., 360.),
                                                pick_data_ants=True)
        # Get polarization
        pol = self.uvd.polarization_array[0]
        
        # Try to run function
        op = lambda d, m: np.max(d.real)
        uniq_ants, mat = hs.average.redundant_diff_summary(
                                          self.uvd, red_bls, pol, 
                                          op_upper=op, op_lower=op, 
                                          verbose=True)
        nt.assert_equal(mat.shape[0], uniq_ants.size)
        
if __name__ == "__main__":
    unittest.main()

