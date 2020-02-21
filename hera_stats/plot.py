import numpy as np
from . import stats, utils
from pyuvdata import UVData
import hera_pspec as hp
import matplotlib.pyplot as plt
from matplotlib import gridspec


def long_waterfall(uvd_list, bl, pol, title=None, cmap='gray', starting_lst=[], 
                   mode='nsamples', operator='abs', file_type='uvh5', 
                   figsize=(20, 80)):
    """    
    Generates a waterfall plot of flags or nsamples with axis sums from an
    input array.
    
    Parameters
    ----------
    uvd_list : list of UVData objects or list of str
        List of UVData objects to be stacked and displayed. If a list of 
        strings is specified, each UVData object will be loaded one at a time 
        (reduces peak memory consumption).
    
    bl : int or tuple
        Baseline integer or antenna pair tuple of the baseline to plot.
    
    pol : str or int
        Polarization string or integer.
    
    title : str, optional
        Title of the plot. Default: none.
    
    cmap : str, optional
        Colormap parameter for the waterfall plot. Default: 'gray'.
        
    starting_lst : list, optional
        list of starting lst to display in the plot
    
    mode : str, optional
        Which array to plot from the UVData objects. Options: 'data', 'flags', 
        'nsamples'. Default: 'nsamples'. 
    
    operator : str, optional
        If mode='data', the operator to apply when plotting the data. Can be 
        'real', 'imag', 'abs', 'phase'. Default: 'abs'.
    
    file_type : str, optional
        If `uvd_list` is passed as a list of strings, specifies the file type 
        of the data files to assume when loading them. Default: 'uvh5'.
    
    figsize : tuple, optional
        The size of the figure, in inches. Default: (20, 80).
    
    Returns
    -------
    main_waterfall : matplotlib.axes
        Matplotlib Axes instance of the main plot
        
    freq_histogram : matplotlib.axes
        Matplotlib Axes instance of the sum across times
        
    time_histogram : matplotlib.axes
        Matplotlib Axes instance of the sum across freqs
        
    data : numpy.ndarray
        A copy of the stacked_array output that is being displayed
    """
    # Check data operator is valid (if specified)
    if mode == 'data':
        if operator == 'abs':
            op = np.abs
        elif operator == 'real':
            op = np.real
        elif operator == 'imag':
            op = np.imag
        elif operator == 'phase':
            op = np.angle
        else:
            raise ValueError("'%s' is not a valid value for the operator kwarg. "
                             "Valid options: ['real', 'imag', 'abs', 'phase']" \
                             % operator)
    
    arr_list = []
    for _uvd in uvd_list:
        
        # Try to load UVData from file
        if isinstance(_uvd, str):
            try:
                uvd = UVData()
                if file_type == "uvh5":
                    
                    # Do partial load
                    if isinstance(bl, (int, np.int)):
                        raise TypeError("Baseline 'bl' must be specified as an "
                                        "antenna pair to use the partial load "
                                        "feature.")
                    uvd.read_uvh5(_uvd, bls=[bl,], polarizations=[pol,])
                else:
                    
                    # Load the whole file
                    uvd.read(_uvd, file_type=file_type)
            except OSError:
                # Common issue is that wrong file_type is being used
                import sys
                print("long_waterfall: file_type = '%s'" % file_type, 
                      file=sys.stderr)
                raise
                
        elif isinstance(_uvd, UVData):
            # Already loaded into UVData object
            uvd = _uvd
        else:
            raise TypeError("uvd_list must contain either filename strings or "
                            "UVData objects")
    
        # Construct key to access data
        if isinstance(bl, (int, np.int)):
            bl = uvd.baseline_to_antnums(bl)
        key = (bl[0], bl[1], pol)
    
        # Get requested data
        if mode == 'data':
            arr_list.append(op(uvd.get_data(key)))
        elif mode == 'flags':
            arr_list.append(uvd.get_flags(key))
        elif mode == 'nsamples':
            arr_list.append(uvd.get_nsamples(key))
        else:
            raise ValueError("mode '%s' not recognized." % mode)
    
    # Stack arrays into one big array
    data = utils.stacked_array(arr_list)
    
    # Set up the figure and grid
    fig = plt.figure()
    grid = gridspec.GridSpec(ncols=10, nrows=32)
    
    # Create main components of figure
    main_waterfall = fig.add_subplot(grid[1:30, 1:8])
    freq_histogram = fig.add_subplot(grid[30:32, 1:8], sharex=main_waterfall)
    time_histogram = fig.add_subplot(grid[1:30, 8:10], sharey=main_waterfall)
    
    # Set sizes
    fig.set_size_inches(figsize)
    fig.suptitle(title, fontsize=30, y=0.984) #, horizontalalignment='center')
    grid.tight_layout(fig)
    counter = data.shape[0] // 60
    
    # Waterfall plot
    main_waterfall.imshow(data, aspect='auto', cmap=cmap, 
                          interpolation='none')
    main_waterfall.set_ylabel('Integration Number')
    main_waterfall.set_yticks(np.arange(0, counter*60 + 1, 30))
    main_waterfall.set_ylim(60*(counter+1), 0)
    
    # Red lines separating files
    for i in range(counter+1):
        main_waterfall.plot(np.arange(data.shape[1]),
                            60*i*np.ones(data.shape[1]), '-r')
    for i in range(len(starting_lst)):
        if not isinstance(starting_lst[i], str):
            raise TypeError("starting_lst must be a list of strings")
    
    # Add text of filenames
    if len(starting_lst) > 0:
        for i in range(counter):
            short_name = 'first\nintegration LST:\n'+starting_lst[i]
            plt.text(-20, 26 + i*60, short_name, rotation=-90, size='small',
                     horizontalalignment='center')
    main_waterfall.set_xlim(0, data.shape[1])
    
    # Frequency sum plot
    counts_freq = np.sum(data, axis=0)
    max_counts_freq = max(np.amax(counts_freq), data.shape[0])
    normalized_freq = 100 * counts_freq/max_counts_freq
    freq_histogram.set_xticks(np.arange(0, data.shape[1], 50))
    freq_histogram.set_yticks(np.arange(0, 101, 5))
    freq_histogram.set_xlabel('Channel Number (Frequency)')
    freq_histogram.set_ylabel('Occupancy %')
    freq_histogram.grid()
    freq_histogram.plot(np.arange(0, data.shape[1]), normalized_freq, 'r-')
    
    # Time sum plot
    counts_times = np.sum(data, axis=1)
    max_counts_times = max(np.amax(counts_times), data.shape[1])
    normalized_times = 100 * counts_times/max_counts_times
    time_histogram.plot(normalized_times, np.arange(data.shape[0]), 'k-',
                        label='all channels')
    time_histogram.set_xticks(np.arange(0, 101, 10))
    time_histogram.set_xlabel('Flag %')
    time_histogram.autoscale(False)
    time_histogram.grid()
    
    # Returning the axes
    return main_waterfall, freq_histogram, time_histogram, data


def scatter_bandpowers(uvp, x_dly, y_dly, keys, operator='abs', label=None, 
                       ax=None):
    """
    Scatter plot of delay spectrum bandpowers from two different delay bins. 
    The bandpowers can be taken from multiple spws / blpairs / polpairs at the 
    same time (see the `keys` argument).
    
    Example of `keys` argument:
    .. highlight:: python
    .. code-block:: python
        red_grps, red_lens, red_angs = uvp.get_red_blpairs()
        keys = {
            'spws':     0,
            'blpairs':  red_grps[0],
            'polpairs': uvp.get_polpairs(),
        }
    
    Parameters
    ----------
    uvp : UVPSpec
        Input delay spectrum object.
    
    x_dly, y_dly : int
        Index of delay bins to use as the x and y values respectively.
    
    keys : dict
        Dictionary with keys 'spws', 'blpairs', 'polpairs', which can be lists 
        or single values. This allows bandpowers from multiple spws / blpairs / 
        polpairs to be plotted simultaneously.
    
    operator : str, optional
        If mode='data', the operator to apply when plotting the data. Can be 
        'real', 'imag', 'abs', 'phase'. Default: 'abs'. Default: 'abs'.
    
    label : str, optional
        Label to use for this set of data in the plot legend. Default: None.
    
    ax : matplotlib.axes, optional
        Default: None.
    
    Returns
    -------
    ax : matplotlib.axes
        Matplotlib Axes instance.
    """
    if 'spws' not in keys:
        raise KeyError("keys dict has missing 'spws' key")
    if 'blpairs' not in keys:
        raise KeyError("keys dict has missing 'blpairs' key")
    if 'polpairs' not in keys:
        raise KeyError("keys dict has missing 'polpairs' key")
    
    # Check that dict items are all lists
    for key in ['spws', 'blpairs', 'polpairs']:
        if not isinstance(keys[key], (list, np.ndarray)):
            keys[key] = [keys[key],]
    
    # Loop over all dims and fetch bandpowers for requested delay indices
    x = []; y = []
    for spw in keys['spws']:
        for polpair in keys['polpairs']:
            for blp in keys['blpairs']:
                ps = uvp.get_data((spw, blp, polpair))
                x = np.concatenate( (x, ps[:,x_dly].flatten()) )
                y = np.concatenate( (y, ps[:,y_dly].flatten()) )
    
    # Apply transform corresponding to chosen mode
    ops = {'abs': np.abs, 'phase': np.angle, 'real': np.real, 'imag': np.imag}
    if operator not in ops.keys():
        raise KeyError("Operator '%s' not recognized; must be one of %s" \
                        % (operator, list(ops.keys())) )
    x = ops[operator](x)
    y = ops[operator](y)
    
    # Create plot if needed
    if ax is None:
        ax = plt.subplot(111)
    
    # Plot points
    ax.plot(x, y, marker='.', ls='none', label=label)
    ax.legend()
    return ax


def redgrp_corrmat(corr, red_grp, cmap='RdBu', figsize=(30.,20.), 
                   line_alpha=0.2):
    """
    Plot the correlation matrix for a set of delay spectra in a redundant 
    group. See `hera_stats.stats.redgrp_pspec_covariance()` for a function to 
    calculate the correlation matrix.
    
    The elements of the correlation matrix are assumed to be in the same order 
    as the `red_grp` list. Furthermore, the `red_grp` list is assumed to be 
    ordered by blpair integer. Blocks of elements that have the same first bl 
    in common are marked in the matrix.
    
    Parameters
    ----------
    corr : ndarray
        Covariance or correlation matrix.
    
    red_grp : list
        List of baseline-pairs in the redundant group (one for each row/column 
        of the `corr` matrix).
    
    cmap : str, optional
        Matplotlib colormap to use for the correlation matrix plot. 
        Default: 'RdBu'.
    
    vmin, vmax : float, optional
        Minimum and maximum values of the 
        
    figsize : tuple, optional
        Size of the figure, in inches. Default: (30, 20).
    
    line_alpha : float, optional
        Alpha value of the lines used to draw blocks in the correlation 
        matrix. Default: 0.2.
    
    Returns
    -------
    fig : matplotlib.Figure
        Figure containing the correlation matrix.
    """
    # Plot matrix
    fig = plt.figure()
    mat = plt.matshow(corr, cmap=cmap, vmin=-1., vmax=1.)
    plt.colorbar()
    ax = plt.gca()
    
    # Add label for each block of baseline-pairs
    # (items in each block have the same first bl in blpair)
    blps = [hp.uvpspec_utils._blpair_to_bls(_blp) for _blp in red_grp]
    bl1, bl2 = zip(*blps)
    unique_bls = np.unique(bl1)
    block_start_idx = [np.argmax(bl1 == _uniq) for _uniq in unique_bls]
    
    # Add ticks for each block
    ticks = ax.set_xticks(block_start_idx)
    ticks = ax.set_yticks(block_start_idx)
    ticklbls = ax.set_xticklabels(["%d" % _bl for _bl in unique_bls], 
                                  rotation='vertical')
    ticklbls = ax.set_yticklabels(["%d" % _bl for _bl in unique_bls])
    plt.gcf().set_size_inches(figsize)
    
    # Draw block dividers
    for idx in block_start_idx:
        plt.axhline(idx, color='k', alpha=line_alpha)
        plt.axvline(idx, color='k', alpha=line_alpha)
    
    return fig


