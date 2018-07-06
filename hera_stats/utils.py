# Extra utilities for hera_stats

import ephem
import os
import copy
import numpy as np

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

    angles = np.array(angles)
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
