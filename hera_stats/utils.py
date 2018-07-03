# Extra utilities for hera_stats

import ephem
import os
import copy
import numpy as np

def find_files(direc, endstring, remove=[]):
    """
    Given a directory path and a search string, finds files. Useful for looking for hera miriad files
    with the right suffixes.

    Parameters:
    ----------
    direc: string
        Directory in which to search for the files.

    endstring: string:
        Search parameter, the ending of the files requested.
    """
    if direc[-1] != "/":
        direc += "/"

    if not isinstance(remove, list):
        remove = [remove]

    st = endstring

    allf = os.listdir(direc)
    files = []
    for f in allf:
        if f[-len(st):] == st and sum([r in f for r in remove]) == 0:
            files += [f]

    files = sorted(files)

    return [direc + f for f in files]

def unique_items(grps):
    """
    Finds all unique items in a jackknife group list. Use this as opposed to np.unique because
    np.unique cannot do baseline pairs

    Parameters
    ----------
    grps: list
        Groups directly from jaccknife data.

    Returns
    -------
    items: list
        Unique items found in grps.
    """
    unique = []
    [[[unique.append(item)
       for item in g if (item not in unique)]
      for g in glist]
     for glist in grps]

    return sorted(unique)

def plt_layout(req):
    """Helper function for figuring out best layout of histogram table"""
    # req is the requested number of plots. All numbers below it are potential divisors
    divisors = np.arange(1,req)

    # How many empty plots would be left
    left = [req%a for a in divisors]

    # difference between axis lengths
    howsquare = [np.abs(req//a - a) for a in divisors]

    lenientness = 0
    while True:
        for a in divisors-1:
            # Look for smallest number of empty plots left and close to equal 
            # side lengths
            if left[a] <= lenientness and howsquare[a] <= lenientness:
                return (divisors[a],int(np.ceil(float(req)/divisors[a])))
        # If nothing found, decrease strictness
        lenientness += 1
def bin_wrap(angles, n):

    degs = np.linspace(-360, 360, 720/10+1)
    ha = np.hstack([angles-360, angles])

    isindeg = [bool(sum((ha > degs[i]) * (ha < degs[i+1]))) for i in range(len(degs)-1)]
    longest = None
    rng = None
    for i, val in enumerate(isindeg):
        if val == True:
            seq = isindeg[i: ]
            try:
                seq = seq[:seq.index(False)]
            except ValueError:
                pass
            if len(seq) > longest:
                longest = len(seq)
                rng = (i, i + len(seq) + 1)

    wrapped = ha[(ha > degs[rng[0]]) * (ha <= degs[rng[1]])]
    bins = np.linspace(min(wrapped), max(wrapped), n+1)
    bins -= bins//360 * 360
    return bins

def is_in_wrap(low, hi, angle):

    if angle >= low and angle < hi:
        return True
    angle += 180
    low += 180
    hi += 180
    if angle % 360 >= low % 360 and angle % 360 < hi % 360:
        return True

    return False

def timestamp():
    """
    Returns a timestamp string.
    """
    t = ephem.now()
    timestamp = str(t.datetime()).replace(" ",".").replace(":","_")[:19]
    return timestamp