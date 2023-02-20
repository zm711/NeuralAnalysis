# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 09:41:39 2022

@author: ZacharyMcKenzie


INPUTS: st: spike times nSpikes
        eventTimes: list of eventtimes
        window are list with [start, stop]
        psthBinSize: float of time bin in seconds
"""
import numpy as np
import numpy.matlib

#import analysis.histdiff as hd
from numba import jit

""" collected the psth, rasters, bins, spikeCounts"""


def psthAndBA(
    st: np.array, event_times: np.array, window: list, psthBinSize: float
) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    spike_times = st
    spike_time_index = (spike_times > np.min(event_times + window[0])) & (
        spike_times < np.max(event_times + window[1])
    )
    spike_times_sorted = spike_times[spike_time_index]

    # pull out values and ensure typing for jit
    start = float(window[0])
    end = float(window[1])

    binned_array, bins = time_stamps_to_bins(
        spike_times_sorted,
        event_times,
        psthBinSize,
        start,
        end,
    )
    spike_counts = np.sum(binned_array, axis=1)
    psth = np.mean(np.divide(binned_array, psthBinSize), axis=0)
    rasterX = 0
    rasterY = 0
    binned_index = np.transpose(np.nonzero(binned_array))
    tr = binned_index[:, 0]
    b = binned_index[:, 1]
    bins = bins.T
    rasterX, yy = rasterize(bins[b])
    rasterY = yy + np.reshape(np.matlib.repmat(tr.T, 3, 1), (1, len(tr.T) * 3))

    return psth, bins, rasterX, rasterY, spike_counts, binned_array


"""creates rasters in x and y dimension in case we need to do python raster plots"""


@jit(nopython=True, cache=True)
def rasterize(time_stamps: np.array) -> tuple[np.array, np.array]:
    min_val: int = 0
    max_val: int = 1
    x_out = np.empty((len(time_stamps) * 3))
    x_out[:] = np.NaN
    for value in range(len(x_out)):
        if value % 3 == 0:
            x_out[value] = time_stamps[(2 * value) % len(time_stamps)]
        if value % 3 == 1:
            x_out[value] = time_stamps[2 * (value - 1) % len(time_stamps)]
    y_out = np.zeros((len(time_stamps) * 3)) + min_val
    for value in range(len(y_out)):
        if value % 3 == 1:
            y_out[value] = max_val
    xx = x_out.reshape(1, len(x_out))
    yy = y_out.reshape(1, len(y_out))

    return xx, yy


@jit(nopython=True, cache=True)  # speeds up the numpy processes. My testing showed 7x
def time_stamps_to_bins(
    time_stamps: np.array,
    reference_pts: np.array,
    bin_size: float,
    start: float,
    end: float,
) -> tuple[np.array, np.array]:
    step_number = int(abs((end - start) / bin_size) + 1)
    bin_borders = np.linspace(start, end, step_number)
    bin_number = len(bin_borders) - 1
    bin_array = np.zeros((len(reference_pts), bin_number))

    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

    """
    if len(reference_pts) == 0:
        bin_array = list()

        return bin_array, bin_centers
    """
    if len(time_stamps) == 0:
        return bin_array, bin_centers

    for n in range(len(reference_pts)):
        bin_array[n], _ = np.histogram(time_stamps, bin_borders + reference_pts[n])

    return bin_array, bin_centers


"""sp is are master dictionary of the neural data. eventTimes here is our dictionary
of all our stimuli with all events and their lengths. timeBinSize will be binsize in 
seconds"""


def rasterPSTH(sp: dict, eventTimes: dict, timeBinSize: float) -> tuple[dict, list]:
    spikeTimes = np.squeeze(sp["spikeTimes"])
    clu = np.squeeze(sp["clu"])
    clusterIDs = list(sp["cids"])
    psthvalues = {}
    windowlst = list()
    for stim in eventTimes.keys():
        if len(eventTimes[stim]["EventTime"]) == 0:
            continue
        else:
            psthvalues[eventTimes[stim]["Stim"]] = {}
            windowIn = input(
                "Enter stimulus window to be analyzed for each event in format x.y for stimulus {stim}".format(
                    stim=eventTimes[stim]["Stim"]
                )
            )
            windowStr = windowIn.split(",")
            window = [float(windowStr[0]), float(windowStr[-1])]
            windowlst.append(window)
            eventTimesOnset = eventTimes[stim]["EventTime"]
            # psthvalues[eventTimes[stim]['Stim']]['Window'] = window[-1]-window[0]

            for cluster in clusterIDs:
                psthvalues[eventTimes[stim]["Stim"]][str(cluster)] = {}
                print("Processing cluster {clu}".format(clu=cluster))
                _, bins, _, _, _, ba = psthAndBA(
                    spikeTimes[clu == cluster], eventTimesOnset, window, timeBinSize
                )
                psthvalues[eventTimes[stim]["Stim"]][str(cluster)]["BinnedArray"] = ba
                if np.shape(ba)[1] != len(bins):
                    bins = psthvalues[eventTimes[stim]["Stim"]][str(clusterIDs[0])][
                        "Bins"
                    ]
                    psthvalues[eventTimes[stim]["Stim"]][str(cluster)]["Bins"] = bins
                else:
                    psthvalues[eventTimes[stim]["Stim"]][str(cluster)]["Bins"] = bins

    return psthvalues, windowlst


""" sets up the data to run the histdiff--DEPRECATING


def timestampsToBinned(
    timeStamps: np.array, referencePoints: np.array, binSize: float, window: list
):
    stepNumber = int(abs((window[1] - window[0]) / binSize + 1))
    binBorders = np.linspace(window[0], window[1], num=stepNumber)
    binNumber = len(binBorders) - 1
    we make two deep copies of our bins. Since we will be mutating things with rust we need these copies to
    have their own pointers so we can use them without just changing the same data over and over
    totalBins = np.broadcast_to(
        binBorders, (len(referencePoints),) + binBorders.shape
    ).copy()
    binCenters = binBorders.copy()

    if len(referencePoints) == 0:
        binArray = list()
        binCenters = binBorders[0:-2] + binSize / 2
        return binArray, binCenters
    binArray = np.zeros((len(referencePoints), binNumber))
    if len(timeStamps) == 0:
        binCenters = binBorders[0:-2] + binSize / 2
        return binArray, binCenters

    try:
        bincenterpy(binCenters)
    except NameError:
        print("Rust error")
    finally:
        for r in range(len(referencePoints)):
            try:  # if ordhist module generate our binCenters once. Remove last point
                hd.rusthist(
                    timeStamps, np.array(referencePoints[r]), totalBins[r]
                )  # then we run our rust histo alogrithm, which mutates in place in totalBins
            except (
                NameError
            ):  # if we can't use rust, we default back to the slower, but functional python-native algorithm
                n, binCenters, test = hd.histdiff(
                    timeStamps, referencePoints[r], totalBins[r]
                )
                binArray[r] = n

        if len(binCenters) > binNumber:  # if rust code check for extra collumn
            binCenters = binCenters[:-1]  # delete the extra column
            binArray = totalBins[:, :-1]  #

        return binArray, binCenters
"""
