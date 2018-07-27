#!/usr/bin/env python2
import numpy as np
import copy, operator, itertools
from collections import OrderedDict as odict
import hera_pspec as hp
import hera_stats as hs
from pyuvdata import UVData
from hera_stats.data import DATA_PATH
import os


def make_uvp_data(overwrite=False):
    """ make the UVPSpec and PSpecConatiner data used in tests/* """
    # get data files
    beam = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    dfile1 = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    dfile2 = os.path.join(DATA_PATH, "zen.odd.xx.LST.1.28828.uvOCRSA")

    # get baseline groups
    uvd = UVData()
    uvd.read_miriad(dfile1)
    reds, lens, angs = hp.utils.get_reds(uvd, bl_error_tol=1.0, pick_data_ants=True)

    # get uvps
    uvp1 = hp.testing.uvpspec_from_data(dfile1, reds,
                                       spw_ranges=[(50, 100)], beam=beam, verbose=False)
    uvp1_avg, _, _ = hp.grouping.bootstrap_resampled_error(uvp1, time_avg=False, Nsamples=100, seed=0,
                                                     normal_std=True, robust_std=False)

    uvp2 = hp.testing.uvpspec_from_data(dfile2, reds,
                                       spw_ranges=[(50, 100)], beam=beam, verbose=False)
    uvp2_avg, _, _ = hp.grouping.bootstrap_resampled_error(uvp2, time_avg=False, Nsamples=100, seed=0,
                                                     normal_std=True, robust_std=False)

    # put into a container
    psc = hp.PSpecContainer("uvp_container.h5", mode="rw")
    psc.set_pspec("IDR2_1", "zen.even.xx.LST.1.28828.uvOCRSA.h5", uvp1, overwrite=overwrite)
    psc.set_pspec("IDR2_1", "zen.odd.xx.LST.1.28828.uvOCRSA.h5", uvp2, overwrite=overwrite)

    # make jacknives
    jacks = hp.PSpecContainer("jack_container.h5", mode="rw")

    # split ants
    uvpl = hs.jackknives.split_ants(uvp1_avg, n_jacks=20, minlen=1, verbose=False)
    for i, uvps in enumerate(uvpl):
        for j, uvp in enumerate(uvps):
            jacks.set_pspec("jackknives", "spl_ants.{}.{}".format(i, j), uvp, overwrite=overwrite)

    # split gha
    np.random.seed(5)
    uvpl = hs.jackknives.split_gha(uvp1_avg, [3, 6, 4, 8, 6,])
    for i, uvps in enumerate(uvpl):
        for j, uvp in enumerate(uvps):
            jacks.set_pspec("jackknives", "spl_gha.{}.{}".format(i, j), uvp, overwrite=overwrite)

    # stripe times
    uvpl = hs.jackknives.stripe_times(uvp1_avg)
    for i, uvps in enumerate(uvpl):
        for j, uvp in enumerate(uvps):
            jacks.set_pspec("jackknives", "stripe_times.{}.{}".format(i, j), uvp, overwrite=overwrite)


