#!/usr/bin/env python2
import numpy as np
import copy, operator, itertools
from collections import OrderedDict as odict
import hera_pspec as hp
import hera_stats as hs
from pyuvdata import UVData
from hera_stats.data import DATA_PATH
import os


def make_uvp_data(uvp_psc_name=None, jack_psc_name=None, overwrite=False):
    """
    This function is used to generate the most up-to-date
    HDF5 data files in hera_stats/hera_stats/data that are
    used in the tests/* testing suite.

    This function can generate two files : 
        1. a PSpecContainer w/ standard uvp objects (uvp_psc_name)
        2. a PSpecContainer w/ a bunch of different jackknife sets (jack_psc_name)

    It will only generate the files that have a specified filename and,
    for the time being, there are some hard-coded parameters that go into
    the generation of the PSpectra, like the weighting type, baseline types,
    spw-ranges etc.

    These files can be committed to the data/ repo under
    uvp_data.h5 and jack_data.h5 for the testing scripts to use.
    """
    assert uvp_psc_name is not None or jack_psc_name is not None, \
           "Neither uvp_psc_name or jack_psc_name is specified, can't run..."

    # get data files
    beam = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    dfile1 = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    dfile2 = os.path.join(DATA_PATH, "zen.odd.xx.LST.1.28828.uvOCRSA")

    # get baseline groups
    uvd = UVData()
    uvd.read_miriad(dfile1)
    reds, lens, angs = hp.utils.get_reds(uvd, bl_error_tol=1.0, pick_data_ants=True)

    # get uvps
    uvp1 = hp.testing.uvpspec_from_data(dfile1, reds[:2],
                                       spw_ranges=[(50, 100)], beam=beam, verbose=False)
    uvp2 = hp.testing.uvpspec_from_data(dfile2, reds[:2],
                                       spw_ranges=[(50, 100)], beam=beam, verbose=False)

    if uvp_psc_name is not None:
        # put into a container
        psc = hp.PSpecContainer(uvp_psc_name, mode="rw")
        psc.set_pspec("IDR2_1", "zen.even.xx.LST.1.28828.uvOCRSA.h5", uvp1, overwrite=overwrite)
        psc.set_pspec("IDR2_1", "zen.odd.xx.LST.1.28828.uvOCRSA.h5", uvp2, overwrite=overwrite)

    if jack_psc_name is not None:
        # make jacknives
        jacks = hp.PSpecContainer(jack_psc_name, mode="rw")
        
        """
        # split ants
        np.random.seed(5)
        uvpl = hs.split.split_ants(uvp1, n_jacks=5, minlen=1, verbose=False)
        for i, uvps in enumerate(uvpl):
            for j, uvp in enumerate(uvps):
                uvp_avg, _, _ = hp.grouping.bootstrap_resampled_error(uvp, time_avg=True, Nsamples=50,
                                                                      seed=0, normal_std=True,
                                                                      robust_std=False,
                                                                      blpair_groups=[uvp.get_blpairs()])
                jacks.set_pspec("jackknives", "spl_ants.{}.{}".format(i, j), uvp_avg, overwrite=overwrite)
        """
        
        # split gha
        np.random.seed(5)
        uvpl = hs.split.hour_angle(uvp1, [3, 2, 3,])
        for i, uvps in enumerate(uvpl):
            for j, uvp in enumerate(uvps):
                uvp_avg, _, _ = hp.grouping.bootstrap_resampled_error(uvp, time_avg=True, Nsamples=50,
                                                                      seed=0, normal_std=True,
                                                                      robust_std=False,
                                                                      blpair_groups=[uvp.get_blpairs()])

                jacks.set_pspec("jackknives", "spl_gha.{}.{}".format(i, j), uvp_avg, overwrite=overwrite)
        
        """
        # stripe times
        np.random.seed(5)
        uvpl = hs.split.stripe_times(uvp1)
        for i, uvps in enumerate(uvpl):
            for j, uvp in enumerate(uvps):
                uvp_avg, _, _ = hp.grouping.bootstrap_resampled_error(uvp, time_avg=True, Nsamples=50,
                                                                      seed=0, normal_std=True,
                                                                      robust_std=False,
                                                                      blpair_groups=[uvp.get_blpairs()])

                jacks.set_pspec("jackknives", "stripe_times.{}.{}".format(i, j), uvp_avg, overwrite=overwrite)
        """

