#! bin/python3
"""
Created on Thu Aug 11 08:36:00 2022

@author: ZacharyMcKenzie

Recreating the histdiff function from the c code in histdiff for matlab from Nick's code 
set'

time_stamps = spikeTimes of interest should be nSpikes x1
reference_points = scalar or vector to references in Nick's code he puts it in as a s
calar'
bin_borders =scalar or vector for our bins. 

To see original C code to  cortex-lab/spikes/analyis/helpers/histdiff.c That 
source code has a bunch of Mex functions to allow the code to play with Matlab. I 
changed those to be inputs into python and I also deleted all the warning messages
since for my implementation it should be pre-processed by this point and should 
basically work.

With the numba jit boost it runs extremely well and shouldn't slow anything down.
numpy.histogram works well for sorting spike times by events, but for two vectors
with the bins this works, but I didn't get numpy.histogram working

INPUTS: time_stamps = spikeTimes (sorted in previous function), but basically an 
                       nTimes vectors
        reference_points = 1 event from eventTime (ie a scalar)
        bin_borders = vector of our bins
OUTPUTS: cnts = the number of counts per bin [nbins]
         ctrs = the bins centered rather than on the edges [nbins]
         
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def histdiff(time_stamps: np.array, reference_points: np.array, bin_borders: np.array):
    """Function to organize for the numba jit. It takes in three np.arrays all of floats
    time_stamps should be the values to be assessed with reference to `reference_points`
    based on `bin_borders`"""
    data1: np.array = time_stamps
    ndata1: int = len(time_stamps)
    data2: np.array = reference_points
    ndata2: int = len(reference_points)
    nbins: int = len(bin_borders)

    """Now we check if nbins is scalar or vector"""
    if nbins == 1:
        minV, maxV = findext(data1, ndata1, data2, ndata2)  # find min and max
        size = (maxV - minV) / nbins  # our bins needed to be regularly spaced
    else:  # if a vector subtract one to account for the extra "end point"
        nbins = len(bin_borders) - 1
        bins = bin_borders  # all of our bins are listed in bin_borders
        size = bins[1] - bins[0]
        for count in range(1, nbins):
            if (
                abs(bins[count + 1] - bins[count] - size) > 1e-3 * size
            ):  # regular spacing check
                size = 0
                break
        if size:
            minV = bins[0]
    """Here we check for which algorithm to use. If size means regularly spaced 
     and we can use faster if ordered and fast if unordered. Should be ordered 
     for us"""
    cnts: np.array = np.zeros((nbins))
    ctrs: np.array = np.zeros((nbins))
    if size:  # checking for regular spacing
        if chckord(data1, ndata1, data2, ndata2):  # check if ordered
            cnts, ctrs = ordhist(
                data1, ndata1, data2, ndata2, minV, size, nbins, cnts, ctrs
            )

        else:  # if not then run slightly slower algo.
            cnts, ctrs = reghist(
                data1, ndata1, data2, ndata2, minV, size, nbins, cnts, ctrs
            )

    else:  # if unordered, not-regular do slow algorithm
        cnts, ctrs = binhist(data1, ndata1, data2, ndata2, bins, nbins, cnts, ctrs)

    return cnts, ctrs


"""Sorting algorithms below"""

# for regularly spaced, but unordered data faster than binhist
@jit(nopython=True)
def reghist(
    data1: np.array,
    ndata1: int,
    data2: np.array,
    ndata2: int,
    minV: float,
    size: float,
    nbins: float,
    cnts: np.array,
    ctrs: np.array,
) -> tuple[np.array, np.array]:

    maxV = minV + size * nbins
    for counta in range(nbins):
        cnts[counta] = 0
        ctrs[counta] = minV + counta * size + size / 2

    for counts in range(ndata1):
        for counts2 in range(ndata2):
            diff: float = data1[counts] - data2[counts2]
            if diff < minV or diff >= maxV:
                continue
            idx = (diff - minV) / size
            cnts[int(idx)] += 1

    return cnts, ctrs


# for regularly spaced, ordered data. Is able to skip values based on order. Fastest
@jit(nopython=True)
def ordhist(
    data1: np.array,
    ndata1: int,
    data2: np.array,
    ndata2: int,
    minV: float,
    size: float,
    nbins: int,
    cnts: np.array,
    ctrs: np.array,
) -> tuple[np.array, np.array]:

    maxV: float = minV + size * nbins
    for counta in range(nbins):
        # cnts[counta] = 0
        ctrs[counta] = minV + counta * size + size / 2
    jmin = 0
    for count_data1 in range(ndata1):
        for counts2 in range(jmin, ndata2):
            if (data1[count_data1] - data2[counts2]) <= maxV:
                jmin = counts2
                break
        for counts_2 in range(jmin, ndata2):
            diff = data1[count_data1] - data2[counts_2]
            if diff >= minV:
                cnts[int((diff - minV) / size)] += 1

    return cnts, ctrs


# for irregular spacing un ordered data. Slowest.
@jit(nopython=True)
def binhist(
    data1: np.array,
    ndata1: int,
    data2: np.array,
    ndata2: int,
    bins: np.array,
    nbins: int,
    cnts: np.array,
    ctrs: np.array,
) -> tuple[np.array, np.array]:

    for counts in range(nbins):
        # cnts[counts] = 0
        ctrs[counts] = (bins[counts] + bins[counts + 1]) / 2
    for counts2 in range(ndata1):
        for counts3 in range(ndata2):
            for counts in range(nbins):
                if (data1[counts2] - data2[counts3]) >= bins[counts] and (
                    data1[counts2] - data2[counts3]
                ) < bins[counts + 1]:
                    cnts[counts] += 1
    return cnts, ctrs


"""Auxiliary functions for the finding min and max if unordered data"""


@jit(nopython=True)
def findext(
    data1: np.array, ndata1: int, data2: np.array, ndata2: int
) -> tuple[float, float]:
    minV = data1[0] - data2[0]
    maxV = data1[0] - data2[0]
    for counts in range(1, ndata1):
        for counts2 in range(1, ndata2):
            diff = data1[counts] - data2[counts2]
            if diff < minV:
                minV = diff
            if diff > maxV:
                maxV = diff
    return minV, maxV


"""Auxiliary function for checking if data is ordered or unordered"""


@jit(nopython=True)
def chckord(data1: np.array, ndata1: int, data2: np.array, ndata2: int) -> int:
    for counts in range(1, ndata1):
        if data1[counts] < data1[counts - 1]:
            return 0
    for counts2 in range(1, ndata2):
        if data2[counts2] < data2[counts2 - 1]:
            return 0
    return 1
