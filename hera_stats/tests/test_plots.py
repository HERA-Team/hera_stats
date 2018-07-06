import hera_stats as hs
import matplotlib.pyplot as plt
import nose.tools as nt
import os
from hera_stats.data import DATA_PATH
import hera_pspec as hp

import matplotlib
import warnings
warnings.simplefilter("ignore", matplotlib.mplDeprecation)

class Test_Plots():

    def setUp(self):
        filepath = os.path.join(DATA_PATH, "test_pc_jackknife.h5")
        plt.ioff()

        pc = hp.container.PSpecContainer(filepath)
        self.jkset = hs.JKSet(pc, "spl_ants")
        self.zscores = hs.stats.zscores(self.jkset, method="varsum").T()

    def test_plot_spectra(self):

        jkset = self.jkset
        f, ax = plt.subplots()

        hs.plots.plot_spectra(jkset[0], fig=f)
        nt.assert_raises(AssertionError, hs.plots.plot_spectra, jkset, fig=f)
        hs.plots.plot_spectra(jkset[0, 0], fig=f)
        f.clear()

        hs.plots.plot_spectra(self.zscores, fig=None, logscale=False, with_errors=False, show_groups=True)


    def test_hist_2d(self):

        jkset = self.jkset
        f, ax = plt.subplots()

        hs.plots.hist_2d(jkset[:, 0], logscale=True, ax=ax, display_stats=False)
        ax.clear()

        hs.plots.hist_2d(self.zscores, ylim=(-4,4), logscale=False, ax=None, normalize=True, vmax=0.2)

    def test_stats_plots(self):

        jkset = self.jkset
        f, ax = plt.subplots()

        hs.plots.plot_kstest(jkset[0], ax=ax)
        hs.plots.plot_anderson(jkset[0], ax=ax)

        hs.plots.plot_kstest(jkset[0], ax=None)
        hs.plots.plot_anderson(jkset[0], ax=None)

    def test_scatter(self):
        jkset = self.jkset
        f, ax = plt.subplots()

        hs.plots.scatter(jkset[0], ax=ax)
        hs.plots.scatter(jkset[0], ax=None, compare=False, logscale=False)

