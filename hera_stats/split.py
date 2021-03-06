import numpy as np
import hera_pspec as hp
from pyuvdata import UVData
import copy
import astropy.coordinates as coords
from . import utils


def _stripe_array(x, stripes, width=1):
    """
    Split 1D array into a number of stripes of given width.
    
    Parameters
    ----------
    x : array_like
        Input array of values to be striped.
    
    stripes : int
        Number of stripes.
    
    width : int, optional
        Width of each stripe. Default: 1.
    
    Returns
    -------
    x_stripes : list of array_like
        List containing each stripe of the input array.
    """
    blocks = []
    # Loop over no. of stripes
    for i in range(stripes): 
        x_stripe = []
        for j in range(width): # loop over width of stripe 
            x_stripe = np.concatenate((x_stripe, 
                                       x[i*(width)+j::stripes*width]))
        blocks.append( np.sort(x_stripe) )
    return blocks


def lst_blocks(uvp, blocks=2, lst_range=(0., 2.*np.pi)):
    """
    Split a UVPSpec object into multiple objects, each containing spectra 
    within different contiguous LST ranges. There is no guarantee that each 
    block will contain the same number of spectra or samples.
    
    N.B. This function uses the `lst_avg_array` property of an input UVPSpec 
    object to split the LSTs (and not the LSTs of the individual visibilities 
    that went into creating each delay spectrum).
    
    Parameters
    ----------
    uvp : UVPSpec
        Object containing delay spectra.
    
    blocks : int, optional
        How many blocks to return. Default: 2.
    
    lst_range : tuple, optional
        Tuple containing the minimum and maximum LST to retain. This is the 
        range that will be split up into blocks. Default: (0., 2*pi)
    
    Returns
    -------
    uvp_list : list of UVPSpec
        List of UVPSpec objects, one for each LST range. Empty blocks will 
        appear as None in the list.
    
    lst_bins : array_like
        Array of LST bin edges. This has dimension (blocks+1,).
    """
    # Check validity of inputs
    if not isinstance(uvp, hp.UVPSpec):
        raise TypeError("uvp must be a single UVPSpec object.")
    if not (lst_range[0] >= 0. and lst_range[1] <= 2.*np.pi):
        raise ValueError("lst_range must be in the interval (0, 2*pi)")
    if not isinstance(blocks, (int, np.int, np.integer)):
        raise TypeError("'blocks' must be an integer")
    if not blocks > 0:
        raise ValueError("Must have blocks >= 1")
    
    # Get LSTs
    lsts = np.unique(uvp.lst_avg_array)
    
    # Define bin edges
    lst_bins = np.linspace(lst_range[0], lst_range[1], blocks+1)
    
    # Loop over bins and select() the LST ranges required
    uvp_list = []
    for i in range(lst_bins.size - 1):
        idxs = np.where( np.logical_and(lsts >= lst_bins[i], 
                                        lsts < lst_bins[i+1]) )[0]
        _uvp = None
        if idxs.size > 0:
            # Select LSTs in this range
            _uvp = uvp.select(lsts=lsts[idxs], inplace=False)
        uvp_list.append(_uvp)
    
    return uvp_list, lst_bins


def lst_stripes(uvp, stripes=2, width=1, lst_range=(0., 2.*np.pi)):
    """
    Split a UVPSpec object into multiple UVPSpec objects, each containing 
    spectra within alternating stripes of LST.
    
    N.B. Gaps in LST are ignored; this function stripes based on the ordered 
    list of available LSTs in `uvp` only.
    
    N.B. This function uses the `lst_avg_array` property of the input UVPSpec 
    object to split the LSTs (and not the LSTs of the individual visibilities 
    that went into creating each delay spectrum).
    
    Parameters
    ----------
    uvp : UVPSpec
        Object containing delay spectra.
    
    stripes : int, optional
        How many stripes to return. Default: 2.
    
    width : int, optional
        Width of each stripe, in number of LST bins. Default: 1.
    
    lst_range : tuple, optional
        Tuple containing the minimum and maximum LST to retain. This is the 
        range that will be split up into blocks. Default: (0., 2*pi)
    
    Returns
    -------
    uvp_list : list of UVPSpec
        List of UVPSpec objects, one for each LST range.
    """
    # Check validity of inputs
    if not isinstance(uvp, hp.UVPSpec):
        raise TypeError("uvp must be a single UVPSpec object.")
    if not (lst_range[0] >= 0. and lst_range[1] <= 2.*np.pi):
        raise ValueError("lst_range must be in the interval (0, 2*pi)")
    if not isinstance(stripes, (int, np.int, np.integer)):
        raise TypeError("'stripes' must be an integer")
    assert stripes > 0, "Must have stripes >= 1"
    assert width > 0, "Must have width >= 1"
    
    # Get sorted list of LSTs within the specified range
    lsts = np.unique(uvp.lst_avg_array)
    idxs = np.where( np.logical_and(lsts >= lst_range[0], 
                                    lsts < lst_range[1]) )
    lsts = np.sort(lsts[idxs])
    
    # Loop over bins and select() the LST ranges required 
    uvp_list = []
    lst_stripes = _stripe_array(lsts, stripes=stripes, width=width)
    for lstripe in lst_stripes:
        _uvp = uvp.select(lsts=lstripe, inplace=False)
        uvp_list.append(_uvp)
    
    return uvp_list



def hour_angle(uvp, bins_list, specify_bins=False, bls=None):
    """
    Splits UVPSpec based on the galactic hour-angle at the time of measurement.

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
    assert isinstance(uvp, hp.UVPSpec), \
        "Expected uvp to be list or UVPSpec, not {}".format(type(uvp).__name__)

    if specify_bins:
        assert np.asarray(bins_list).ndim == 2, \
            "Expected bins to be a list of lists."
    else:
        assert np.asarray(bins_list).ndim == 1, \
            "Expected bins to be a list of antenna numbers."

    if bls is not None:
        uvp.select(bls=bls, inplace=True)

    # Create reference lst -> time_avg dictionary (for sorting)
    ref = dict(list(zip(uvp.lst_avg_array, uvp.time_avg_array)))
    rads = np.unique(uvp.lst_avg_array)

    # Get telescope location information
    R = np.sqrt(sum(uvp.telescope_location**2))
    lat = np.arcsin(uvp.telescope_location[2]/R) * 180. / np.pi

    # Convert lst to gha
    norms = coords.SkyCoord(rads, lat, unit=["rad", "deg"])
    gha = norms.transform_to(coords.builtin_frames.Galactic)

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
                raise AttributeError("No times found in one or more of the "
                                     "bins specified.")
            _uvp = uvp.select(times=jdays, inplace=False)

            # Set metadata
            angs = gha.l.deg[val]
            angs_180 = (angs + 180) % 360
            if np.std(angs) <= np.std(angs_180):
                _uvp.labels = np.array([np.average(angs)])
            else:
                _uvp.labels = np.array([(np.average(angs_180) - 180) % 360])

            inrange.append(_uvp)

        uvpl.append(inrange)
        
    return uvpl


#*********#
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

    assert isinstance(uvp, hp.UVPSpec), \
        "Expected uvp to be hera_pspec.UVPSpec, not {}".format(type(uvp).__name__)

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
        inds = np.random.choice(list(range(len(bl))), minlen, replace=False)
        bl_i = np.array(list(map(uvp.antnums_to_bl, bl)))[inds]
        uvp1 = uvp.select(bls=bl_i, inplace=False)
        uvp1.labels = np.array([ant_nums[i]])
        uvp1.jktype = "omit_ants"
        uvp_list.append(uvp1)

    return [uvp_list]


def blps_by_antnum(blps, split='norepeat'):
    """
    Split a list of redundant groups of baseline-pairs into two, depending on 
    whether there are repeated antennas in the baseline pair (`norepeat`), or 
    whether there are auto baselines in the pair (`noautos`).
    
    Parameters
    ----------
    blps : list of list of blpairs
        List of redundant groups of baseline-pairs. The blps can be either 
        tuples of tuples, or blpair integers.
    
    split : str, optional
        Type of split to perform on each baseline group. Available options are:
        
        - 'norepeat':
            Split into one group where antennas are used at most once per 
            blpair, and another where they are used more than once.
            
        - 'noautos': 
            Split into one group with auto-blpairs and one group with non-autos 
            (but antennas can be used more than once per blpair).
         
    Returns
    -------
    blps_a, blps_b : list of list of blpairs
        List of redundant groups of baseline-pairs.
        
         - For 'norepeat', group A contains the blps *without* repeated 
           antennas, while group B contains the blps *with* repeated antennas.
         
         - For 'noautos', group A contains the blps *without* auto-baselines, 
           while group B contains the blps *with* auto-baselines.
    """
    # Check that input blps is list of lists
    if not isinstance(blps[0], list):
        raise TypeError("blps must be a list of lists of baseline-pairs")
    
    # Check for valid split type
    if split not in ['norepeat', 'noautos']:
        raise ValueError("Split type '%s' not recognized." % split)
    
    # Loop over redundant groups
    blps_a, blps_b = [], []
    for grp in blps:
        
        # Loop over baseline-pairs
        grp_a, grp_b = [], []
        for blp in grp:
            
            # Convert into antnum pairs if not already
            if isinstance(blp, int):
                blp = hp.uvputils._blpair_to_antnums(blp)
            
            # Split according to whether antenna numbers are repeated
            if split == 'norepeat':
                # Split out 
                if np.unique(blp).size == 4:
                    grp_a.append(blp) # without repeated antennas
                else:
                    grp_b.append(blp) # with repeated antennas
            elif split == 'noautos':
                # Split out blpairs being multiplied by themselves
                if sorted(blp[0]) == sorted(blp[1]):
                    grp_b.append(blp) # with repeated baselines
                else:
                    grp_a.append(blp) # without repeated baselines
            
        blps_a.append(grp_a)
        blps_b.append(grp_b)
        
    return blps_a, blps_b
    
