import os
import hera_stats as hs
from hera_stats.data import DATA_PATH
class Test_Stats():

    def setUp(self):
        self.st = hs.jkf_stats()

        self.filepath = os.path.join(DATA_PATH,"gauss.spl_ants.Nj40.2018-06-07.04_50_54.jkf")
        self.st.load_file(self.filepath)

    def test_stats(self):
        st = self.st

        st.standardize(st.data.spectra,st.data.errs,method="varsum")
        st.standardize(st.data.spectra,st.data.errs,method="weightedsum")

        st.anderson(showmore=True)
        st.kstest(showmore=True)
