import os
import hera_stats as hs
from hera_stats.data import DATA_PATH
import nose.tools as nt
import numpy as np
import copy
import hera_pspec as hp

class Test_Stats():

    def setUp(self):
        filepath = os.path.join(DATA_PATH, "test_pc_jackknife.h5")
        pc = hp.container.PSpecContainer(filepath)
        self.jkset = hs.JKSet(pc, "spl_ants")

    def test_stats(self):

        hs.stats.weightedsum(self.jkset[0])
        zs = hs.stats.zscores(self.jkset, axis=1, method="varsum").T()
        hs.stats.anderson(self.jkset[0], summary=True, verbose=True)
        stat = hs.stats.kstest(zs[0], summary=True, verbose=True)
        nt.assert_true(stat < 0.8)

        zs = hs.stats.zscores(self.jkset.flatten().T(), axis=0, method="weightedsum")
        hs.stats.kstest(zs[:, 0])
        hs.stats.anderson(zs.T())
        
        nt.assert_raises(NameError, hs.stats.zscores, self.jkset[0], method="ahhhhhhhhhhh!!!")