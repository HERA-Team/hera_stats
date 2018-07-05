import os
import hera_stats as hs
from hera_stats.data import DATA_PATH
import nose.tools as nt
import numpy as np
import copy
import hera_pspec as hp

class Test_Stats():

    def setUp(self):
        filepath = os.path.join(DATA_PATH, "test_pc_jackknife")
        pc = hp.container.PSpecContainer(filepath)
        self.jkset = hs.JKSet(pc, "spl_ants")

    def test_stats(self):

        hs.stats.weightedsum(self.jkset[0])
        hs.stats.zscores(self.jkset[0])
        hs.stats.anderson(self.jkset[0], summary=True, verbose=True)
        hs.stats.kstest(self.jkset[0], summary=True, verbose=True)
