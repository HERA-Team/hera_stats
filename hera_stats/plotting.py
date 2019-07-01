import numpy as np
import matplotlib.pyplot as plt
import hera_pspec as hp
from . import stats


def plot_delay_scatter(uvp, x_dly, y_dly, keys, mode='abs', label=None, ax=None):
    """
    
    
    Parameters
    ----------
    uvp : UVPSpec
        Input delay spectrum object.
    
    x_dly, y_dly : int
        Index of delay bins to use as the x and y values respectively.
    
    keys : dict
        Dictionary with keys 'spws', 'blpairs', 'polpairs', which can be lists 
        or single values.
    
    Returns
    -------
    
    """
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
    modes = {'abs': np.abs, 'phase': np.angle, 'real': np.real, 'imag': np.imag}
    if mode not in modes.keys():
        raise KeyError("mode '%s' not recognized; must be one of %s" \
                        % (mode, list(modes.keys())) )
    x = modes[mode](x)
    y = modes[mode](y)
    
    # Create plot
    if ax is None:
        ax = plt.subplot(111)
    
    ax.plot(x, y, marker='.', ls='none', label=label)
    return ax


def plot_redgrp_corrmat(corr, red_grp, cmap='RdBu', figsize=(30.,20.), 
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

