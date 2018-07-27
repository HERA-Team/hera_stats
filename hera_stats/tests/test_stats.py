import os
import hera_stats as hs
from hera_stats.data import DATA_PATH
import nose.tools as nt
import numpy as np
import copy
import hera_pspec as hp
from pyuvdata import UVData


class Test_Stats():

    def setUp(self):
        filepath = os.path.join(DATA_PATH, "uvp_jackknife.h5")
        pc = hp.container.PSpecContainer(filepath)
        self.jkset = hs.JKSet(pc, "spl_ants")

    def test_stats(self):

        hs.stats.weightedsum(self.jkset[0])
        zs = hs.stats.zscores(self.jkset, axis=1, z_method="varsum")
        hs.stats.anderson(zs, summary=True, verbose=True)
        stat = hs.stats.kstest(zs, summary=True, verbose=True)
        nt.assert_true(stat < 0.8)

        zs = hs.stats.zscores(self.jkset, axis=(0, 1), z_method="weightedsum")
        nt.assert_equal(zs.shape, (40, 2))
        hs.stats.kstest(zs[:, 0])
        hs.stats.anderson(zs.flatten())
        
        nt.assert_raises(NameError, hs.stats.zscores, self.jkset[0], z_method="ahhhhhhhhhhh!!!")




def test_uvp_zscore():
    # get a UVPSpec
    beam = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    ### TODO: replace with a gaussian noise file when its added to repo
    dfile = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    uvp = hp.testing.uvpspec_from_data(dfile, [(37, 38), (38, 39), (51, 52), (52, 53), (53, 54), (67, 68)],
                                       spw_ranges=[(50, 100)], beam=beam, verbose=False)

    # bootstrap and average
    uvp_avg, _, _ = hp.grouping.bootstrap_resampled_error(uvp, time_avg=True, Nsamples=200, seed=0,
                                                          normal_std=True, robust_std=False, cintervals=None)

    # get zscores
    hs.stats.uvp_zscore(uvp_avg, error_field='bs_std', inplace=True)
    nt.assert_true('bs_std_zscore' in uvp_avg.stats_array.keys())

    # test inplace
    new_uvp = hs.stats.uvp_zscore(uvp_avg, error_field='bs_std', inplace=False)
    new_uvp.history = ''
    uvp_avg.history = ''
    nt.assert_equal(uvp_avg, new_uvp)

    ### TODO: check all zscores are -5 < z < 5 and follow gaussian distribution
    ### when dfile is the gaussian noise simulation






