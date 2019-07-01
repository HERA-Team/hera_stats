import hera_stats as hs
import matplotlib.pyplot as plt
import nose.tools as nt
import os
from hera_stats.data import DATA_PATH
import hera_pspec as hp
import matplotlib
import warnings
warnings.simplefilter("ignore", matplotlib.mplDeprecation)
import shutil
import unittest


class Test_Plotting():

    def setUp(self):
        filepath = os.path.join(DATA_PATH, "uvp_data.h5")
        plt.ioff()

        pc = hp.container.PSpecContainer(filepath, mode='r')
        self.uvp = pc.get_pspec("IDR2_1")[0]

    def test_plot_redgrp_corrmat(self):
        
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
        fig = hs.plotting.plot_redgrp_corrmat(corr_re, grp, cmap='RdBu', 
                                              figsize=(30.,20.), line_alpha=0.2)
        
if __name__ == "__main__":
    unittest.main()

