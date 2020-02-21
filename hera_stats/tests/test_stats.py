import os
import hera_stats as hs
from hera_stats.data import DATA_PATH
import nose.tools as nt
import numpy as np
import copy
import hera_pspec as hp
from pyuvdata import UVData
import shutil
import unittest
import scipy.stats as stats


class Test_Stats(unittest.TestCase):

    def setUp(self):
        # Load example UVPSpec
        filepath = os.path.join(DATA_PATH, "uvp_data.h5")
        psc = hp.container.PSpecContainer(filepath, mode='r')
        self.uvp = psc.get_pspec("IDR2_1")[0]
    
    
    def test_redgrp_pspec_covariance(self):
        red_grps, red_lens, red_angs = self.uvp.get_red_blpairs()
        grp = red_grps[0]
        if isinstance(grp[0], tuple):
            # FIXME: This would not be necessary if get_red_blpairs() returned 
            # properly-formed blpair integers
            grp = [int("%d%d" % _blp) for _blp in grp]
        
        # Calculate delay spectrum correlation matrix for redundant group and plot
        corr_re, corr_im = hs.stats.redgrp_pspec_covariance(
                                        self.uvp, grp, dly_idx=3, spw=0, 
                                        polpair='xx', mode='corr', verbose=True)
        # Check output
        nt.assert_equal(corr_re.shape, (len(grp), len(grp)))
        
        # Calculate delay spectrum covariance matrix for redundant group and plot
        cov_re, cov_im = hs.stats.redgrp_pspec_covariance(
                                        self.uvp, grp, dly_idx=3, spw=0, 
                                        polpair='xx', mode='cov', verbose=True)
        
        # Check that invalid inputs are caught
        nt.assert_raises(ValueError, hs.stats.redgrp_pspec_covariance, 
                         self.uvp, grp, dly_idx=3, spw=0, polpair='xx', 
                         mode='xxx') # invalid mode
        nt.assert_raises(TypeError, hs.stats.redgrp_pspec_covariance, 
                         self.uvp, grp[0], dly_idx=3, spw=0, polpair='xx', 
                         mode='cov') # invalid blpair spec (must be list)
        
    def test_uvp_zscore(self):
        # Get a gaussian noise UVPSpec
        # Start by loading a beam
        beam = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
        
        # Specify baselines
        bls = [(37, 38), (38, 39), (51, 52), (52, 53), (53, 54), (67, 68)]
        
        # Load a real file into UVData
        dfile = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
        uvd = UVData()
        uvd.read_miriad(dfile)
        
        # Replace data w/ gaussian noise
        np.random.seed(0)
        x = stats.norm.rvs(0, 1./np.sqrt(2.), uvd.data_array.size)
        y = stats.norm.rvs(0, 1./np.sqrt(2.), uvd.data_array.size)
        uvd.data_array = (x + 1.j*y).reshape(uvd.data_array.shape)
        uvd.flag_array[:] = False
        
        # Get baseline groups
        reds, lens, angs = hp.utils.get_reds(uvd, pick_data_ants=True)
        uvp = hp.testing.uvpspec_from_data(uvd, reds[:10], spw_ranges=[(50, 100)], 
                                           beam=beam, verbose=False)

        # Bootstrap and average
        uvp_avg, _, _ = hp.grouping.bootstrap_resampled_error(
                                uvp, blpair_groups=[uvp.get_blpairs()], 
                                time_avg=True, Nsamples=200, seed=0, 
                                normal_std=True, robust_std=False, 
                                cintervals=None)

        # Get zscores
        hs.stats.uvp_zscore(uvp_avg, error_field='bs_std', inplace=True)
        nt.assert_true('bs_std_zscore' in uvp_avg.stats_array.keys())

        # zscore std should be gaussian about 1.0 if errorbars and zscore calc is 
        # correct and N = inf for our case when N != inf, check zscore std is 
        # within 1.0 +/- 1/sqrt(uvp.Nblpairs) this is both a test that bootstrap 
        # errorbars are accurate (given uncorrelated noise) and that the 
        # uvp_zscore function does the right arithmetic
        zs_std_real = np.std(uvp_avg.stats_array['bs_std_zscore'][0].ravel().real)
        zs_std_imag = np.std(uvp_avg.stats_array['bs_std_zscore'][0].ravel().imag)
        nt.assert_true(np.abs(1. - zs_std_real) < 1./np.sqrt(uvp.Nblpairs))
        nt.assert_true(np.abs(1. - zs_std_imag) < 1./np.sqrt(uvp.Nblpairs))

        # Test inplace
        new_uvp = hs.stats.uvp_zscore(uvp_avg, error_field='bs_std', inplace=False)
        new_uvp.history = ''
        uvp_avg.history = ''
        nt.assert_equal(uvp_avg, new_uvp)


if __name__ == "__main__":
    unittest.main()
