# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 09:41:39 2022

@author: ZacharyMcKenzie

"""
import numpy as np
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
    rasterY = yy + np.reshape(np.tile(tr.T, (3, 1)), (1, len(tr.T) * 3))

    return psth, bins, rasterX, rasterY, spike_counts, binned_array


@jit(nopython=True, cache=True)
def rasterize(time_stamps: np.array) -> tuple[np.array, np.array]:
    """creates raster values for generating raster plots, based on 
    counts in time bins"""
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



def rasterPSTH(sp: dict, eventTimes: dict, time_bins: list,window_list: list) -> tuple[dict, list]:
    """takes in `sp` neural data, `eventTimes` the stimulus dict, `time_bins` a list of time bin
    sizes, and `window_list` a list of windows to analyze. Function returns psthvalues and 
    the windows used."""
    spikeTimes = np.squeeze(sp["spikeTimes"])
    clu = np.squeeze(sp["clu"])
    clusterIDs = list(sp["cids"])
    psthvalues = {}
    windowlst = list()
    if len(eventTimes.keys())>1 and len(time_bins) == 1:
        time_bins *= len(eventTimes.keys())
    for (index, stim) in enumerate(eventTimes.keys()):
        if len(eventTimes[stim]["EventTime"]) == 0:
            continue
        else:
            psthvalues[eventTimes[stim]["Stim"]] = {}
            if window_list:
                sub_window = window_list[index]
                window = [float(sub_window[0]), float(sub_window[1])]
                windowlst.append(window)
            else:
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
            time_bin_size = time_bins[index]
            for cluster in clusterIDs:
                psthvalues[eventTimes[stim]["Stim"]][str(cluster)] = {}
                print("Processing cluster {clu}".format(clu=cluster))
                _, bins, _, _, _, ba = psthAndBA(
                    spikeTimes[clu == cluster], eventTimesOnset, window, time_bin_size
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
