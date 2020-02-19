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
        filepath = os.path.join(DATA_PATH, "jack_data.h5")
        pc = hp.container.PSpecContainer(filepath)
        self.jkset = hs.JKSet(pc, "spl_ants")
        
        # Load example UVPSpec
        filepath = os.path.join(DATA_PATH, "uvp_data.h5")
        psc = hp.container.PSpecContainer(filepath, mode='r')
        self.uvp = psc.get_pspec("IDR2_1")[0]

    def test_stats(self):
        hs.stats.weightedsum(self.jkset)
        zs = hs.stats.zscores(self.jkset, axis=1, z_method="varsum")
        hs.stats.anderson(zs, summary=True, verbose=True)
        stat = hs.stats.kstest(zs, summary=True, verbose=True)
        ## this test not working currently: nt.assert_true(stat < 0.8)

        zs = hs.stats.zscores(self.jkset, axis=(0, 1), z_method="weightedsum")
        nt.assert_equal(zs.shape, (5, 2))
        hs.stats.kstest(zs[:, 0])
        hs.stats.anderson(zs.flatten())
        
        nt.assert_raises(NameError, hs.stats.zscores, self.jkset[0], 
                         z_method="invalidmethod")
    
    
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
        
    

def test_uvp_zscore():
    ## Get a gaussian noise UVPSpec ##
    # start w/ a beam
    beam = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    # specify baselines
    bls = [(37, 38), (38, 39), (51, 52), (52, 53), (53, 54), (67, 68)]
    # load a real file into UVData
    dfile = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    uvd = UVData()
    uvd.read_miriad(dfile)
    # replace data w/ gaussian noise
    np.random.seed(0)
    uvd.data_array = stats.norm.rvs(0, 1/np.sqrt(2), uvd.data_array.size).reshape(uvd.data_array.shape) \
                     + 1j * stats.norm.rvs(0, 1/np.sqrt(2), uvd.data_array.size).reshape(uvd.data_array.shape)
    uvd.flag_array[:] = False
    # get baseline groups
    reds, lens, angs = hp.utils.get_reds(uvd, pick_data_ants=True)
    uvp = hp.testing.uvpspec_from_data(uvd, reds[:10], spw_ranges=[(50, 100)], beam=beam, verbose=False)

    # bootstrap and average
    uvp_avg, _, _ = hp.grouping.bootstrap_resampled_error(uvp, blpair_groups=[uvp.get_blpairs()], 
                                                          time_avg=True, Nsamples=200, seed=0, 
                                                          normal_std=True, robust_std=False, cintervals=None)

    # get zscores
    hs.stats.uvp_zscore(uvp_avg, error_field='bs_std', inplace=True)
    nt.assert_true('bs_std_zscore' in uvp_avg.stats_array.keys())

    # zscore std should be gaussian about 1.0 if errorbars and zscore calc is correct and N = inf
    # for our case when N != inf, check zscore std is within 1.0 +/- 1/sqrt(uvp.Nblpairs)
    # this is both a test that bootstrap errorbars are accurate (given uncorrelated noise)
    # and that the uvp_zscore function does the right arithmetic
    zscore_std_real = np.std(uvp_avg.stats_array['bs_std_zscore'][0].ravel().real)
    zscore_std_imag = np.std(uvp_avg.stats_array['bs_std_zscore'][0].ravel().imag)
    nt.assert_true(np.abs(1-zscore_std_real) < 1/np.sqrt(uvp.Nblpairs))
    nt.assert_true(np.abs(1-zscore_std_imag) < 1/np.sqrt(uvp.Nblpairs))

    # test inplace
    new_uvp = hs.stats.uvp_zscore(uvp_avg, error_field='bs_std', inplace=False)
    new_uvp.history = ''
    uvp_avg.history = ''
    nt.assert_equal(uvp_avg, new_uvp)


if __name__ == "__main__":
    unittest.main()
