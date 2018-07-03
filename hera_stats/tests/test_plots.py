import hera_stats as hs
import matplotlib.pyplot as plt
import numpy as np
import nose.tools as nt
import os
from hera_stats.data import DATA_PATH
import hera_pspec as hp

import matplotlib
import warnings
warnings.simplefilter("ignore", matplotlib.mplDeprecation)

class Test_Plots():

    def setUp(self):
        self.filepath = os.path.join(DATA_PATH,"onsim.jackknife.spl_ants.Nj20.2018-06-26.17_53_12")
        plt.ioff()

        self.pc = hp.container.PSpecContainer(self.filepath)

    def test_plots(self):

        pc = self.pc

        f = plt.figure()
        ax = f.add_subplot(111,label="h")

        hs.plots.plot_spectra(pc, fig=f)
        nt.assert_raises(IndexError, hs.plots.plot_spectra, pc, 100, fig=f)
        f.clear()

        for mode in ["varsum","weightedsum","raw"]:
            hs.plots.hist_2d(pc, ax=ax, plottype = mode, display_stats=False)
            ax.clear()

        nt.assert_raises(NameError, hs.plots.hist_2d, pc, "nothing")
        hs.plots.plot_kstest(pc, ax=ax)
        hs.plots.plot_anderson(pc, ax=ax)

        hs.plots.scatter(pc, ax=ax, sortby=1)
        ax.clear()
        fig = None
        ax = None

        hs.plots.plot_spectra(pc, fig=f, sortby=1, show_groups=True, with_errors=False)
        hs.plots.hist_2d(pc, ax=ax, sortby=1, normalize=True, vmax=0.2, returned=True)

        hs.plots.plot_kstest(pc, ax=ax, sortby=1)
        hs.plots.plot_anderson(pc, ax=ax, sortby=1)

        hs.plots.plot_zscore_stats(pc)

        hs.plots.split_hist(pc, bins=2)
        hs.plots.split_hist(pc, bins=np.linspace(0,2000,3))
        
        hs.plots.plot_with_and_without(pc, 1)
