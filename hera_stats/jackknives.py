import numpy as np
import hera_pspec as hp
from pyuvdata import UVData
import copy
import os
import utils
import astropy.coordinates as cp
import time

def _bootstrap_single_uvp(uvp, pol, blpairs=None, n_boots=40):
    
    """
    Uses the bootstrap method to estimate the errors for a UVPSpec object.
    Automatically averages over time.

    Parameters:
    ----------
    uvp: hera_pspec.UVPSpec
        Object outputted by pspec, contains power spectrum information.

    pol: str
        The polarization used in pspec calculations.

    blpairs: list, optional
        List of baseline pairs to use in bootstrapping. Should specify only redundant
        baselines. Default: None.

    n_boots: int, optional
        How many bootstraps of the data to to to estimate error bars. Default: 100.

    Returns:
    -------
    uvp_avg: hera_pspec.UVPSpec
        A UVPSpec containing the averaged power spectrum and bootstrapped errors.
    """
    assert isinstance(uvp, hp.UVPSpec), "uvp must be a single UVPSpec."

    # Calculate unique baseline pairs
    if blpairs == None:
        blpairs = uvp.get_blpairs()

    msg = "Expected blpairs to be list of baseline pair tuples, not {}."
    assert isinstance(blpairs[0], tuple), msg.format("list of {}".format(type(blpairs[0]).__name__))
    assert isinstance(blpairs[0][0], tuple), msg.format("list of tuples of {}".format(type(blpairs[0][0]).__name__))
    assert np.asarray(blpairs).shape[1:] == (2, 2), "blpairs shape does not match that of a list of baseline pair tuples."

    uvp_avg = uvp.average_spectra([blpairs], time_avg = True, inplace=False)

    for spw in range(uvp.Nspws):
        allspecs = uvp.data_array[spw]
        nsamples = uvp.nsample_array[spw]
        integs = uvp.integration_array[spw]

        boot_l = []
        for i in range(n_boots):
            # Choose spectra indices at random
            inds = np.random.choice(range(len(allspecs)), len(allspecs), replace=True)

            # Select spectra, nsamples, and integrations in order.
            boot = allspecs[inds]
            nsamps = nsamples[inds]
            ints = integs[inds]

            boot_uvp = copy.deepcopy(uvp)

            # Replace old arrays with new ones.
            boot_uvp.data_array[spw] = boot
            boot_uvp.nsample_array[spw] = nsamps
            boot_uvp.integration_array[spw] = ints

            # Average spectra
            boot_uvp.average_spectra([blpairs], time_avg=True)

            # Get data and appendd
            boot_spec = boot_uvp.get_data((spw, uvp.get_blpairs()[0], pol))[0]
            boot_l.append(boot_spec)

        # Calculate error bar and save to stats_array
        boot_l = np.vstack(boot_l)
        errs = np.std(boot_l.real, axis=0) + 1j * np.std(boot_l.imag, axis=0)
        uvp_avg.set_stats("bootstrap_errs", (spw, uvp.get_blpairs()[0], pol), errs[None])

    return uvp_avg

def bootstrap_jackknife(uvp_list, pol, blpairs=None, n_boots=50):
    """
    Bootstraps a list of jackknife pairs.

    Parameters
    ----------
    uvp_list: list of UVPSpec pairs
        The jackknived data, with UVPSpec.labels as the group and UVPSpec.jktype as 
        the jackknife type, shape=(n, 2).

    pol: string
        Polarization to use for bootstrapping.

    blpairs: list, optional
        List of baseline pairs to use in bootstrapping. Should specify only redundant
        baselines. Default: None.

    n_boots: int, optional
        Number of times to bootstrap to create a sample. Default: 50.

    Returns
    -------
    boot_uvp_list: list
        List containing UVPSpecs that have been bootstrapped.
    """
    msg = "Expected uvp to be list of UVPSpec lists, not {}."
    assert isinstance(uvp_list, (list, tuple, np.ndarray)), msg.format(type(uvp_list).__name__)
    assert isinstance(uvp_list[0], (list, tuple, np.ndarray)), msg.format("list of {}".format(type(uvp_list[0]).__name__))
    assert isinstance(uvp_list[0][0], hp.UVPSpec), msg.format("list of {} lists".format(type(uvp_list[0][0]).__name__))

    uvp_boot_list = []
    for uv in uvp_list:
        bspair = []
        for u in uv:
            uc = copy.deepcopy(u)
            bspair.append(_bootstrap_single_uvp(uc, blpairs=blpairs, pol=pol, n_boots=n_boots))
        uvp_boot_list.append(bspair)
    return uvp_boot_list

def save_jackknife(pc, uvp_list, set_jktype=None, overwrite=False):
    """
    Saves a bootstrapped jackknife pair list to a PSpecContainer.
    Each jackknife has a type, specified in the jackknife function and
    saved as uvp.jktype, which is used to create the group name {jktype}.{n},
    where n is the location in uvp_list. then, each UVPSpec in uvp_list[n] is
    saved as grp{i}, where i is the index within uvp_list[n].

    Parameters
    ----------
    pc: PSpecContainer
        PSpecContainer in which to store the jackknife.

    uvp_list: list of UVPSpecs
        The list to save. Note: will save jackknives with no bootstrap,
        so make sure to run uvp_list through bootstrap_jackknife first.
    """
    msg = "Expected {} to be {}, not {}"
    assert isinstance(pc, hp.container.PSpecContainer), msg.format("pc", "PSpecContainer",
                                                                   type(pc).__name__)
    assert isinstance(uvp_list, (list, tuple, np.ndarray)), msg.format("uvp_list", "list of UVPSpecs",
                                                                       type(uvp_list).__name__)
    assert isinstance(uvp_list[0][0], hp.UVPSpec), "entries of uvp_list must be UVPSpecs."

    if set_jktype is None:
        assert all([hasattr(u, "jktype") for uvp in uvp_list for u in uvp]), "If not all uvps have attribute 'jktype', one must be specified via parameter 'set_jktype'"
        # Check if all jackknives are the same type
        jktypes = [u.jktype for uvp in uvp_list for u in uvp]
        if all([j == jktypes[0] for j in jktypes]):
            jktype = jktypes[0]
        else:
            raise AttributeError("All jackknifes must be of the same type")
    else:
        assert isinstance(set_jktype, str), "set_jktype must be a string."
        jktype = set_jktype

    for i, uvp_pair in enumerate(uvp_list):
        jkf = "jackknives"
        for k, uvp in enumerate(uvp_pair):
            # Save pspec to group 
            uvp.label_1_array = np.array([0])
            uvp.label_2_array = np.array([0])
            name = "{}.{}.grp{}".format(jktype, i, k)
            pc.set_pspec(jkf, psname=name, pspec=uvp, overwrite=overwrite)

def split_ants(uvp, n_jacks=40, verbose=False):
    """
    Splits available antenna into two groups randomly, and returns the
    UVPSpec of each.

    Parameters
    ----------
    uvp: list or single UVPSpec
        The data to use for jackkniving.

    n_jacks: int, optional
        Number of times to jackknife the data. Default: 40.

    verbose: boolean, optional
        If true, prints actions.

    Returns
    -------
    uvp: list or single UVPSpec
        List of hera_pspec.UVPSpec objects that have been split
        accordingly.
    """
    if isinstance(uvp, (list, tuple, np.ndarray)):
        if len(uvp) != 1:
            uvp = hp.uvpspec.combine_uvpspec(uvp)
        else:
            uvp = uvp[0]
    else:
        assert isinstance(uvp, hp.UVPSpec), "Expected uvp to be list or UVPSpec, not {}".format(type(uvp).__name__)

    # Load all baselines in uvp
    blns = map(uvp.bl_to_antnums, uvp.bl_array)
    ants = np.unique(blns)

    groups = []
    uvpl = []
    for i in range(n_jacks):

        if verbose:
            print "Splitting pspecs for %i/%i jackknives" % (i+1, n_jacks)
        c = 0
        minlen = 0
        while minlen <= len(blns)//6 or minlen <= 3:
            # Split antenna into two equal groups
            grps = np.random.choice(ants, len(ants)//2*2,
                                    replace=False).reshape(2, -1)

            # Find baselines for which both antenna are in a group            
            blg = [[], []]
            for bl in blns:
                if bl[0] in grps[0] and bl[1] in grps[0]:
                    blg[0].append(bl)
                elif bl[0] in grps[1] and bl[1] in grps[1]:
                    blg[1].append(bl)

            # Find minimum baseline length
            minlen = min([len(b) for b in blg])

            # If fails to find big enough group 50 times, raise excption
            c += 1
            if c == 50:
                raise AttributeError("Not enough antenna provided")

        blgroups = []
        for b in blg:
            inds = np.random.choice(range(len(b)), minlen, replace=False)
            blgroups.append([uvp.antnums_to_bl(b[i]) for i in inds])

        # Split uvp by groups
        [uvp1,uvp2] = [uvp.select(bls=b, inplace=False) for b in blgroups]

        # Set metadata for saving
        uvp1.labels = np.array(list(grps[0]))
        uvp2.labels = np.array(list(grps[1]))
        uvp1.jktype = "spl_ants"
        uvp2.jktype = "spl_ants"

        uvpl.append([uvp1,uvp2])

    return uvpl

def stripe_times(uvp, period=None, verbose=False):
    """
    Jackknife that splits the UVPSpecData into groups based on
    the period.

    The period is given in seconds, and data is split based on
    alternating bins, that is, if the time array is [1,2,3,4,5,6] and the
    period is 1, then the data will be split into [1,3,5] and [2,4,6].
    If period is 2, then data will be split into [1,2,5,6] and [3,4].
    Finally, if period is 3, you get [1,2,3] and [4,5,6]. If no period
    is profided, it will run a jackknife for every valid and unique
    period.

    Parameters
    ----------
    period: float or list, optional
        If float, jackknifes a single time using one period. If list,
        jackknives for every period provided.

    Returns
    -------
    uvpl: list
        List of UVPSpecData objects split accordingly.
    """
    if isinstance(uvp, list):
        if len(uvp) > 1:
            uvp = hp.uvpspec.combine_uvpspec(uvp)
        else:
            uvp = uvp[0]

    assert isinstance(uvp, hp.UVPSpec), "Expected uvp to be list or single UVPSpec, not {}".format(type(uvp).__name__)

    if isinstance(period, (int, float, np.float)):
        period = [period]

    # Convert all times to seconds after first recorded time.
    times = np.array(sorted(np.unique(uvp.time_avg_array)))
    secs = (times-times[0])*24*3600

    # If no binsize provided, use every binsize possible and unique
    if period == None:
        minperiod = secs[2] - secs[0]
        allperiods = np.array([len(secs)/n for n in range(2,len(secs))])
        period = np.unique(allperiods//1) * minperiod

    uvpl = []
    for per in period:
        # Phase randomly to broaden search
        phase = np.random.uniform(0, per)
        select = np.sin(2*np.pi*(secs + phase)/per) >= 0

        # Split times into bins
        minlen = min([sum(select), sum(~select)])
        t1 = np.random.choice(times[select], minlen, replace=False)
        t2 = np.random.choice(times[~select], minlen, replace=False)

        [uvp1, uvp2] = [uvp.select(times=t, inplace=False) for t in [t1, t2]]

        # Set metadata
        uvp1.labels = "Period %.2f sec Even" % per
        uvp2.labels = "Period %.2f sec Odd" % per
        uvp1.jktype = "stripe_times"
        uvp2.jktype = "stripe_times"
        uvpl.append([uvp1,uvp2])

    return uvpl

def split_gha(uvp, bins_list, specify_bins=False):
    """
    Splits based on the galactic hour-angle at the time of measurement.

    Parameters
    ----------
    uvp: list or UVPSpec
        List or single hera_pspec.UVPSpec object, containing data to use.

    bins_list: list
        One entry for each bin layout, default is that it must be an integer,
        where min and max values for hourangle values will automatically be
        set as limits. If specify_bins is True, then the inpud must be a list of
        ndarrays.

    specify_bins: boolean
        If true, allows bins_list to be specified as a list of the bins themselves.
        Default: False

    Returns
    -------
    uvpl: list of UVPSpec pairs
        The resulting data, one list per jackknife.
    """
    if isinstance(uvp, list):
        if len(uvp) > 1:
            uvp = hp.uvpspec.combine_uvpspec(uvp)
        else:
            uvp = uvp[0]
    assert isinstance(uvp, hp.UVPSpec), "Expected uvp to be list or UVPSpec, not {}".format(type(uvp).__name__)

    if specify_bins:
        assert np.asarray(bins_list).ndim == 2, "Expected bins to be a list of lists."
    else:
        assert np.asarray(bins_list).ndim == 1, "Expected bins to be a list of antenna numbers."

    # Create reference lst -> time_avg dictionary (for sorting).
    ref = dict(zip(uvp.lst_avg_array, uvp.time_avg_array))
    rads = np.unique(uvp.lst_avg_array)

    # Get telescope location information
    R = np.sqrt(sum(uvp.telescope_location**2))
    lat = np.arcsin(uvp.telescope_location[2]/R) * 180. / np.pi

    # Convert lst to gha
    norms = cp.SkyCoord(rads, lat, unit=["rad", "deg"])
    gha = norms.transform_to(cp.builtin_frames.Galactic)

    uvpl = []
    for bins in bins_list:
        # If bins is an integer, use bin wrapping function
        if isinstance(bins, int):
            bins = utils.bin_wrap(gha.l.deg, bins)

        inrange=[]
        for i in range(len(bins) - 1):
            # Check if 
            gha_range = (bins[i], bins[i + 1])
            val = np.array([utils.is_in_wrap(bins[i], bins[i + 1], deg)
                            for deg in gha.l.deg])
            jdays = [ref[rads[i]] for i in range(len(rads)) if val[i] == True]

            if sum(jdays) == 0:
                raise AttributeError("No times found in one or more of the bins specified.")

            _uvp = uvp.select(times=jdays, inplace=False)

            # Set metadata
            _uvp.jktype = "spl_gha"
            _uvp.labels = np.array([np.average(gha.l.deg[val])])
            inrange.append(_uvp)

        uvpl.append(inrange)
    return uvpl

def omit_ants(uvp, ant_nums=None):
    """
    Splits UVPSpecs into groups, omitting one antenna from each.

    uvp: UVPSpec or list
        Single UVPSpec or list of UVPSpecs to use in splitting.

    ant_nums: list
        A list containing integers, each entry will generate one UVPSpec
        which does not contain the antenna specified.

    Returns
    -------
    uvp_list: list of UVPSpecs
        A list containing one list of UVPSpecs, with one for every ant_num
        specified.
    """
    # Check if uvp is valid and combine list.
    if isinstance(uvp, (list, tuple, np.ndarray)):
        if len(uvp) != 1:
            uvp = hp.uvpspec.combine_uvpspec(uvp)
        else:
            uvp = uvp[0]

    # Set up ant_nums
    if isinstance(ant_nums, (list, tuple, np.ndarray)):
        ant_nums = list(ant_nums)
    elif isinstance(ant_nums, int):
        ant_nums = [ant_nums]
    elif ant_nums == None:
        ant_nums = np.unique([uvp.bl_to_antnums(b) for b in uvp.bl_array])
    else:
        raise AssertionError("Expected ant_nums to be list or int, not {}".format(type(ant_nums).__name__))

    assert isinstance(uvp, hp.UVPSpec), "Expected uvp to be hera_pspec.UVPSpec, not {}".format(type(uvp).__name__)

    blpairs = uvp.get_blpairs()

    bls = []
    [[bls.append(b) for b in blp if b not in bls] for blp in blpairs]

    unique_ants = np.unique(bls)
    bl_list = []
    for ant in ant_nums:
        if ant not in unique_ants:
            raise AttributeError("No data for antenna {} found.".format(ant))
        valid = [bl for bl in bls if ant not in bl]
        bl_list.append(valid)

    minlen = min([len(bll) for bll in bl_list])

    uvp_list = []
    for i, bl in enumerate(bl_list):
        inds = np.random.choice(range(len(bl)), minlen, replace=False)
        bl_i = np.array(map(uvp.antnums_to_bl, bl))[inds]
        uvp1 = uvp.select(bls=bl_i, inplace=False)
        uvp1.labels = np.array([ant_nums[i]])
        uvp1.jktype = "omit_ants"
        uvp_list.append(uvp1)

    return [uvp_list]

def sep_files(uvp, filenames):
    """
    Calculates pspec on individual files.
    """
    if isinstance(uvp, hp.UVPSpec):
        uvp = [uvp]
    assert isinstance(uvp, (list, tuple, np.ndarray)), "uvp must be a list."
    assert isinstance(uvp[0], hp.UVPSpec), "uvp items must be UVPSpecs."
    assert isinstance(filenames, (list, tuple, np.ndarray)), "filenames must be a list."
    assert isinstance(filenames[0], str), "filnames must be strings."

    uvp_list = []
    for i, u in enumerate(uvp):
        u.labels = np.array([filenames[i]])
        u.jktype = "sep_files"
        uvp_list.append([u])

    return uvp_list
