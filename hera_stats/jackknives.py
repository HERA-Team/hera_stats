import numpy as np
import hera_pspec as hp
from pyuvdata import UVData
import copy
import os
import utils
import astropy.coordinates as cp


def split_ants(uvp, n_jacks=40, minlen=3, verbose=False):
    """
    Splits available antenna into two groups randomly, and returns the
    UVPSpec of each.

    Parameters
    ----------
    uvp: list or single UVPSpec
        The data to use for jackkniving.

    n_jacks: int, optional
        Number of times to jackknife the data. Default: 40.

    minlen : int, optional
        Minimum number of baselines needed in final group.
        Default: 3.

    verbose: boolean, optional
        If true, prints actions.

    Returns
    -------
    uvp: list or single UVPSpec
        List of hera_pspec.UVPSpec objects that have been split
        accordingly.
    """
    uvp = copy.deepcopy(uvp)

    if isinstance(uvp, (list, tuple, np.ndarray)):
        if len(uvp) != 1:
            uvp = hp.uvpspec.combine_uvpspec(uvp)
        else:
            uvp = uvp[0]
    else:
        assert isinstance(uvp, hp.UVPSpec), "Expected uvp to be list or UVPSpec, not {}".format(type(uvp).__name__)

    # Load all baselines in uvp
    bl_array = np.unique([uvp.antnums_to_bl(bl) for blp in uvp.get_blpairs() for bl in blp])
    blns = map(uvp.bl_to_antnums, bl_array)
    ants = np.unique(blns)

    groups = []
    uvpl = []
    for i in range(n_jacks):

        if verbose:
            print "Splitting pspecs for %i/%i jackknives" % (i+1, n_jacks)
        c = 0
        blglen = 0
        while blglen <= len(blns)//6 or blglen <= minlen:
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

            # Find minimum baseline group length
            blglen = min([len(b) for b in blg])

            # If fails to find big enough group 50 times, raise excption
            c += 1
            if c == 50:
                raise AttributeError("Not enough antenna provided")

        blgroups = []
        for b in blg:
            inds = np.random.choice(range(len(b)), blglen, replace=False)
            blgroups.append([uvp.antnums_to_bl(b[i]) for i in inds])

        # Split uvp by groups
        [uvp1,uvp2] = [uvp.select(bls=b, only_pairs_in_bls=False, inplace=False) for b in blgroups]

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
    uvp = copy.deepcopy(uvp)

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
        assert select.any(), "No times selected in random search..."

        # Split times into bins
        minlen = min([sum(select), sum(~select)])
        t1 = np.random.choice(times[select], minlen, replace=False)
        t2 = np.random.choice(times[~select], minlen, replace=False)

        [uvp1, uvp2] = [uvp.select(times=t, inplace=False) for t in [t1, t2]]

        # Set metadata
        uvp1.labels = np.array(["period_%0.2f_even" % per])
        uvp2.labels = np.array(["period_%0.2f_odd" % per])
        uvp1.jktype = "stripe_times"
        uvp2.jktype = "stripe_times"
        uvpl.append([uvp1,uvp2])

    return uvpl


def split_gha(uvp, bins_list, specify_bins=False, bls=None):
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

    bls: list of tuples, optional
        The baselines to use in in omitting antenna. If None, uses all
        baselines. Default: None.

    Returns
    -------
    uvpl: list of UVPSpec pairs
        The resulting data, one list per jackknife.
    """
    uvp = copy.deepcopy(uvp)

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

    if bls is not None:
        uvp.select(bls=bls, inplace=True)

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

        inrange = []
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
            angs = gha.l.deg[val]
            angs_180 = (angs + 180) % 360
            if np.std(angs) <= np.std(angs_180):
                _uvp.labels = np.array([np.average(angs)])
            else:
                _uvp.labels = np.array([(np.average(angs_180) - 180) % 360])

            inrange.append(_uvp)

        uvpl.append(inrange)
        
    return uvpl


def omit_ants(uvp, ant_nums=None, bls=None):
    """
    Splits UVPSpecs into groups, omitting one antenna from each.

    uvp: UVPSpec or list
        Single UVPSpec or list of UVPSpecs to use in splitting.

    ant_nums: list, optional
        A list containing integers, each entry will generate one UVPSpec
        which does not contain the antenna specified.

    bls: list of tuples, optional
        The baselines to use in in omitting antenna. If None, uses all
        baselines. Default: None.

    Returns
    -------
    uvp_list: list of UVPSpecs
        A list containing one list of UVPSpecs, with one for every ant_num
        specified.
    """
    uvp = copy.deepcopy(uvp)

    # Check if uvp is individual UVPSpec. If not, combine list.
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
        ant_nums = np.unique(uvp.get_blpairs())
    else:
        raise AssertionError("Expected ant_nums to be list or int, not {}".format(type(ant_nums).__name__))

    assert isinstance(uvp, hp.UVPSpec), "Expected uvp to be hera_pspec.UVPSpec, not {}".format(type(uvp).__name__)

    if bls is None:
        # Get all baseline pairs
        blpairs = uvp.get_blpairs()

        # Find unique baselines
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
    Keeps files separate, each one having it's own UVPSpec, but formats them for jkset.

    Parameters
    ----------
    uvp: UVPSpec or list of UVPSpecs
        One uvpspec loaded from each individual file

    filenames: list of strings
        The filenames corresponding to UVPSpecs in the same position.

    Returns
    -------
    uvp_list: list of UVPSpecs
        2D list of UVPSpecs that can be submitted to bootstrap or save functions.
    """
    uvp = copy.deepcopy(uvp)

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

