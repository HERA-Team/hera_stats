import os
import hera_stats as hs
from hera_stats.data import DATA_PATH
import nose.tools as nt
import numpy as np
import copy
import hera_pspec as hp

class Test_Stats():

    def setUp(self):
        self.filepath = os.path.join(DATA_PATH, "onsim.jackknife.spl_ants.Nj20.2018-06-26.17_53_12")
        self.pc = hp.container.PSpecContainer(self.filepath)

    def test_stats(self):

        hs.stats.anderson(self.pc, verbose=True)
        hs.stats.kstest(self.pc, verbose=True)

        p = lambda x: x.imag ** 2
        hs.stats.anderson(self.pc, proj=p, asspec=True,verbose=True, method="weightedsum")
        hs.stats.kstest(self.pc, proj=p, asspec=True, bins=10, verbose=True, method="weightedsum")

        hs.stats.avspec_with_and_without(self.pc, 1)
        hs.stats.item_info(self.pc, 1)

        nt.assert_raises(ValueError, hs.stats.get_data, self.pc, sortby=10000)
        nt.assert_raises(AssertionError, hs.stats.get_data, "Thats no moon...")

        nt.assert_raises(AttributeError, hs.stats.standardize, [0,1], [0])
        nt.assert_raises(AttributeError, hs.stats.standardize, [0], [0])
        nt.assert_raises(NameError, hs.stats.standardize, [np.arange(10)]*2,
                         [np.arange(10)]*2, method="It's a space station...")
