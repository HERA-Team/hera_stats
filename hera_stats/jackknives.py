import numpy as np
import hera_pspec as hp
from pyuvdata import UVData
import copy
import os
import utils
import astropy.coordinates as cp

def _bootstrap_uvp_once(uvp, pol="xx", n_boots=40):
    
    """
    Uses the bootstrap method to estimate the errors for a PSpecData object. 

    Parameters:
    ----------
    uvp: hera_pspec.UVPSpec
        Object outputted by pspec, contains power spectrum information.

    pol: str, optional
        The polarization used in pspec calculations. Default: "xx".

    n_boots: int, optional
        How many bootstraps of the data to to to estimate error bars. Default: 100.

    Returns:
    -------
    uvp_avg: hera_pspec.UVPSpec
        A UVPSpec containing the averaged power spectrum and bootstrapped errors.
    """
    # Calculate unique baseline pairs
    blpairs = uvp.get_blpairs()
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

            # Get data and append
            boot_spec = boot_uvp.get_data((spw, uvp.get_blpairs()[0], pol))[0]
            boot_l.append(boot_spec)

        # Calculate error bar and save to stats_array
        boot_l = np.vstack(boot_l)
        errs = np.std(boot_l.real, axis=0) + 1j * np.std(boot_l.imag, axis=0)
        uvp_avg.set_stats("bootstrap_errs", (spw, uvp.get_blpairs()[0], pol), errs[None])

    return uvp_avg

def bootstrap_jackknife(uvpl, pol="xx", n_boots=50):
    """
    Bootstraps a list of jackknife pairs.

    Parameters
    ----------
    uvpl: list of UVPSpec pairs
        The jackknived data, with UVPSpec.labels as the group and UVPSpec.jkftype as 
        the jackknife type, shape=(n, 2).

    pol: string, optional
        Polarization to use for bootstrapping. Default="xx".

    n_boots: int, optional
        Number of times to bootstrap to create a sample. Default: 50.
    """
    data = []
    for uv in uvpl:
        bspair = []
        for u in uv:
            uc = copy.deepcopy(u)
            bspair.append(_bootstrap_uvp_once(uc, pol=pol, n_boots=n_boots))
        data.append(bspair)
    return data

def save_jackknife(uvpl, savename=None):
    """
    Saves a bootstrapped jackknife pair list.

    uvpl: list of UVPSpecs
        The list to save. Note: will save jackknives with no bootstrap,
        so make sure to run uvpl through bootstrap_jackknife first.

    savename: string, optional
        Outfile name. Will append with a bunch of information.
        Default: None
    """
    if not os.path.exists("./data"):
        os.mkdir("./data")

    # Check if all jackknives are the same type
    jkftypes = [u.jkftype  for uvp in uvpl for u in uvp]
    if all([j == jkftypes[0] for j in jkftypes]):
        jkftype = jkftypes[0]
    else:
        raise AttributeError("All jackknifes must be of the same type")

    fname = "jackknife." + jkftype + ".Nj%i." % len(uvpl) + utils.timestamp()
    if savename is not None:
        fname = savename + "." + fname

    # Open pspec container
    pc = hp.PSpecContainer("./data/" + fname, "rw")

    for i, uvp_pair in enumerate(uvpl):
        jkf = "%s.%i" % (uvp_pair[0].jkftype, i)
        for k, uvp in enumerate(uvp_pair):
            # Save pspec to group {jackknife_type}.{jackknife_num} and 
            # pspec grp{groupnumber}
            name = "grp%i" % k
            pc.set_pspec(jkf, psname=name, pspec=uvp)

def split_ants(uvp, n_jacks=40, verbose=False):
    """
    Splits available antenna into two groups randomly, and returns the
    UVPspec of each.

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
    uvpl: list
        List of hera_pspec.UVPSpecData objects that have been split
        accordingly.
    """
    if isinstance(uvp, list):
        uv = uvp[0]
        for u in uvp[1:]:
            uv += u
        uvp = uv

    # Load all baselines in uvp
    blns = [uvp.bl_to_antnums(bl) for bl in uvp.bl_array]
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

        uvp1.labels = np.array(list(grps[0]))
        uvp2.labels = np.array(list(grps[1]))
        uvp1.jkftype = "spl_ants"
        uvp2.jkftype = "spl_ants"

        uvpl.append([uvp1,uvp2])

    return uvpl


def split_files(uvp, files, identifier=None, filepairs=None, verbose=False):
    """
    Splits the files into two groups, using one of two methods. One of
    these must be provided. Providing both will error out.

    If an identifier is specified, the jackknife finds the files that
    have the identifier and compares them to those which don't. It
    automatically does every combination but only up to a maximum of
    n_jacks (not yet tho).

    If pairs are specified, which is a list of len-2 lists, the pairs are
    selected using two indices provided. This is essentially a list of
    pairs of indices to use for selecting files.

    Parameters
    ----------
    uvp: list of UVPSpec objects
        One entry to this list corresponds to one file, to be named in the
        files kwarg.

    files: list of strings
        The filepaths of each corresponding UVPSpec in uvp.

    identifier: string, optional
        String segment to use as a filter for grouping files.

    pairs: list, optional
        List of index pairs, for a more manual selection of file groupings.

    Returns
    -------
    uvpl: list
        List of UVPSpecData objects split accordingly.
    """
    # Sanity checks
    if type(uvp) != list:
        raise AttributeError("Split files needs a list of uvp objects.")
    if len(uvp) < 2:
        raise AttributeError("Fewer than two files supplied. Make sure "
                             "combine = False for load_uvd.")
    if identifier is not None and filepairs is not None:
        raise AttributeError("Please only specify either identifier or "
                             "file pairs.")
    if identifier is None and filepairs is None:
        raise AttributeError("No identifier or file pair list specified.")

    uvp = np.array(uvp)
    files = np.array(files)
    if type(identifier) == str:

        infile = np.array([identifier in f for f in files])
        grp1, uvp1 = files[infile], uvp[infile]
        grp2, uvp2 = files[~infile], uvp[~infile]
        minlen = min([len(uvp1), len(uvp2)])
        for i in range(minlen):
            uvp1[i].jkftype = "spl_files"
            uvp1[i].labels = np.array([grp1[i]])
            uvp2[i].jkftype = "spl_files"
            uvp2[i].labels = np.array([grp2[i]])

        if len(grp1) == 0:
            raise AttributeError("Identifier not found in any filenames "
                                 "loaded.")
        if len(grp2) == 0:
            raise AttributeError("Identifier found in all filenames... "
                                 "Be more strict!")

        print uvp1, uvp2
        uvpl = [[uvp1[i], uvp2[i]] for i in range(len(uvp1))]
        grps = [[grp1[i], grp2[i]] for i in range(len(grp1))]

        if verbose:
            print "Found %i pairs of files" % len(uvpl)

    # Split according to index pairs provided
    elif type(filepairs) == list:
        uvpl = [[uvp[i], uvp[j]] for [i,j] in filepairs]
        grps = [[files[i], files[j]] for [i,j] in filepairs]

    return uvpl

def split_times(uvp, periods=None, verbose=False):
    """
    Jackknife that splits the UVPSpecData into groups based on
    the binsize.

    The binsize is given in seconds, and data is split based on
    alternating bins, that is, if the time array is [1,2,3,4,5,6] and the
    binsize is 1, then the data will be split into [1,3,5] and [2,4,6].
    If binsize is 2, then data will be split into [1,2,5,6] and [3,4].
    Finally, if binsize is 3, you get [1,2,3] and [4,5,6]. If no binsize
    is profided, it will run a jackknife for every valid and unique time
    binsize.

    Parameters
    ----------
    binsizes: float or list, optional
        If float, jackknifes a single time using one binsize. If list,
        jackknives for every binsize provided.

    Returns
    -------
    uvpl: list
        List of UVPSpecData objects split accordingly.

    grps: list
        Groups used. Each contains two filepaths.

    n_pairs: list
        Number of baseline pairs used in each jackknife.
    """
    if isinstance(uvp, list):
        uvp = uvp[0]

    if not isinstance(periods, list):
        periods = [periods]

    times = np.array(sorted(np.unique(uvp.time_avg_array)))
    secs = (times-times[0])*24*3600

    # If no binsize provided, use every binsize possible and unique
    if periods == [None]:
        minperiod = secs[2] - secs[0]
        allperiods = np.array([len(secs)/n for n in range(2,len(secs))])
        periods = np.unique(allperiods//1) * minperiod

    uvpl = []
    for per in periods:
        # Phase randomly to broaden search
        phase = np.random.uniform(0, per)
        select = np.sin(2*np.pi*(secs + phase)/per) >= 0

        # Split times into bins
        minlen = min([sum(select), sum(~select)])
        t1 = np.random.choice(times[select], minlen, replace=False)
        t2 = np.random.choice(times[~select], minlen, replace=False)

        [uvp1, uvp2] = [uvp.select(times=t, inplace=False) for t in [t1, t2]]
        uvp1.labels = ["spl_ants", "Period %.2f sec Even" % per]
        uvp2.labels = ["spl_ants", "Period %.2f sec Odd" % per]
        uvpl.append([uvp1,uvp2])

    return uvpl

def split_gha(uvp, bins_list):
    """
    Splits based on the galactic hour-angle at the time of measurement.

    Parameters
    ----------
    uvp: list or UVPSpec
        List or single hera_pspec.UVPSpec object, containing data to use.

    bins_list: list
        One entry for each bin layout, can either be a integer, where min and max
        hourangle values will automatically be set as limits, or ndarray, where
        bins are specified.

    Returns
    -------
    uvpl: list of UVPSpec pairs
        The resulting data, one list per jackknife.
    """
    if isinstance(uvp, list):
        uvp = hp.uvpspec.combine_uvpspec(uvp)

    ref = dict(zip(uvp.lst_avg_array, uvp.time_avg_array))
    rads = np.unique(uvp.lst_avg_array)

    norms = cp.SkyCoord(rads, -23, unit=["rad", "deg"])
    gha = norms.transform_to(cp.builtin_frames.Galactic)

    uvpl = []
    for bins in bins_list:
        if isinstance(bins, int):
            bins = utils.bin_wrap(gha.l.deg, bins)
        inrange=[]
        for i in range(len(bins) - 1):
            gha_range = (bins[i], bins[i + 1])
            val = np.array([utils.is_in_wrap(bins[i], bins[i + 1], deg)
                            for deg in gha.l.deg])
            jdays = [ref[rads[i]] for i in range(len(rads)) if val[i] == True]

            if sum(jdays) == 0:
                raise AttributeError("No times found in one or more of the bins specified.")

            _uvp = uvp.select(times=jdays, inplace=False)
            _uvp.jkftype = "spl_gha"
            _uvp.labels = np.array([np.average(gha.l.deg[val])])
            inrange.append(_uvp)
        uvpl.append(inrange)
    return uvpl