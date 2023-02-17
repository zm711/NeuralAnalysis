#! bin/python
"""
Created on Thu Aug 11 08:36:00 2022

NOTE: 22JAN23--I've implemented the same in a numpy using the c layer so this
code really doesn't need to be used. I deprecated the wrapper for this code so it 
should automatically just use the numpy implementation. '

@author: ZacharyMcKenzie

Recreating the histdiff function from the c code in histdiff for matlab from Nick's code 
set'

timeStamps = spikeTimes of interest should be nSpikes x1
referencePoints = scalar or vector to references in Nick's code he puts it in as a s
calar'
binBorders =scalar or vector for our bins. Nick preprocesses this as a vector.

To see original C code good to our github/analyis/helpers/histdiff.c. Can be opened with
any text editor or if using Linux can be viewed with cat or less pretty easily. That 
source code has a bunch of Mex functions to allow the code to play with Matlab. I 
changed those to be inputs into python and I also deleted all the warning messages
since for my implementation it should be pre-processed by this point and should 
basically work.

INPUTS: timeStamps = spikeTimes (sorted in previous function), but basically an 
                       nTimes vectors
        referencePoints = 1 event from eventTime (ie a scalar)
        binBorders = vector of our bins
OUTPUTS: cnts = the number of counts per bin [nbins]
         ctrs = the bins [nbins]
         
         
NOW I have rust based code to run things faster. For rust it takes in timeStamps, 
referencePoints as an array, and a pointer to the totalBins array. This will be mutated 
in place with no returns. So should go fast. If ordhist.pyd or ordhist.so is not present
on the local computer or searchable on github it defaults to a python implmentation
"""

import numpy as np
from numba import jit

try:
    from ordhist import reghistpy
except ModuleNotFoundError:
    print("no rust code")


def rusthist(timeStamps, RefPts, binBorders):
    reghistpy(timeStamps, RefPts, binBorders)


def histdiff(timeStamps: np.array, referencePoints: np.array, binBorders: np.array):
    """First we pull in our data"""
    data1 = timeStamps
    ndata1 = np.size(timeStamps)
    data2 = referencePoints
    ndata2 = np.size(referencePoints)
    nbins = len(binBorders)
    """Now we check if nbins is scalar or vector"""
    if nbins == 1:
        minV, maxV = findext(data1, ndata1, data2, ndata2)
        size = (maxV - minV) / nbins
    else:  # if a vector subtract one to account for the extra "end point"
        nbins = len(binBorders) - 1
        bins = binBorders  # all of our bins are listsed in binBorders
        size = bins[1] - bins[0]
        for count in range(1, nbins):
            if (
                abs(bins[count + 1] - bins[count] - size) > 1e-3 * size
            ):  # regular spacing check
                size = 0
                break
        if size:
            minV = bins[0]
    """Here we check for which algorithm to use. If size means regularly spaced and we 
       can use faster if ordered and fast if unordered.
       Should be ordered for us"""
    if size:
        if chckord(data1, ndata1, data2, ndata2):  # check if ordered
            cnts, ctrs = ordhist(data1, ndata1, data2, ndata2, minV, size, nbins)
            return cnts, ctrs, 1
        else:
            cnts, ctrs = reghist(data1, ndata1, data2, ndata2, minV, size, nbins)
            return cnts, ctrs, 2
    else:  # if unordered do slow algorithm
        cnts, ctrs = binhist(data1, ndata1, data2, ndata2, bins, nbins)
        return cnts, ctrs, 3


"""Sorting algorithms below"""


@jit(nopython=True, cache=True)
def reghist(
    data1: np.array,
    ndata1: int,
    data2: np.array,
    ndata2: int,
    minV: float,
    size: float,
    nbins: float,
) -> tuple[np.array, np.array]:
    cnts = np.zeros((nbins))
    ctrs = np.zeros((nbins))
    maxV = minV + size * nbins
    for counta in range(nbins):
        cnts[counta] = 0
        ctrs[counta] = minV + counta * size + size / 2

    for counts in range(ndata1):
        if ndata2 == 1:
            diff: float = data1[counts] - data2
            if diff < minV or diff >= maxV:
                continue
            idx = (diff - minV) / size
            cnts[int(idx)] += 1
        else:
            for counts2 in range(ndata2):
                diff = data1[counts] - data2[counts2]
                if diff < minV or diff >= maxV:
                    continue

                idx = diff - minV
                idx = idx / size
                print(type(idx))
                cnts[int(idx)] += 1
    return cnts, ctrs


@jit(nopython=True, cache=True)
def ordhist(
    data1: np.array,
    ndata1: int,
    data2: np.array,
    ndata2: int,
    minV: float,
    size: float,
    nbins: int,
):
    cnts = np.zeros((nbins))
    ctrs = np.zeros((nbins))
    maxV: float = minV + size * nbins
    for counta in range(nbins):
        cnts[counta] = 0
        ctrs[counta] = minV + counta * size + size / 2
    jmin = 0
    for count_data1 in range(ndata1):
        if ndata2 == 1:
            if data1[count_data1] - data2 >= maxV:
                continue
            else:
                diff = data1[count_data1] - data2
                if diff >= minV:
                    cnts[int((diff - minV) / size)] += 1
        else:
            for counts2 in range(jmin, ndata2):
                if (data1[count_data1] - data2[counts2]) >= maxV:
                    jmin = counts2
                for counts3 in range(jmin, ndata2):
                    diff = data1[count_data1] - data2[counts3]
                    if diff >= minV:
                        cnts[((diff - minV) / size).astype(int)] += 1
    return cnts, ctrs


@jit(nopython=True, cache=True)
def binhist(
    data1: np.array,
    ndata1: int,
    data2: np.array,
    ndata2: int,
    bins: np.array,
    nbins: int,
) -> tuple[np.array, np.array]:
    cnts = np.zeros((nbins))
    ctrs = np.zeros((nbins))
    for counts in range(nbins):
        cnts[counts] = 0
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


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
def chckord(data1: np.array, ndata1: int, data2: np.array, ndata2: int) -> int:
    for counts in range(1, ndata1):
        if data1[counts] < data1[counts - 1]:
            return 0
    for counts2 in range(1, ndata2):
        if data2[counts2] < data2[counts2 - 1]:
            return 0
    return 1
