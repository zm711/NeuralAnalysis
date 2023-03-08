#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 13:00:09 2022

@author: zacharymckenzie

This is a function set that creates a drift map based on spike_amplitudes and 
spike_depths. It then  goes through a subset of bins and looks for changes in
depth using the `detectDriftEvents` nested function. It returns a scatter with 
grayscale dots based on spikes. It marks spikes which have drifted in red

INPUTS: spike_times: nSpikes (times in seconds of when spikes occurred)
        spike_amps: nSpikes (amps of each spike)
        spike_depths: nSpikes (depths of each spike-either referenced to probe
                               or reference to start of tissue)
        
OUTPUTS: scatter of spikes with time on x, depth on y, and blackness based on 
         amps.
         
         
Functions are adapted from the Matlab code by Nick Steinmetz
grayfn is my version of the matlab gray() function to generate a gray scale
map
"""

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns


def plotDriftmap(
    spike_times: np.array, spike_amps: np.array, spike_depths: np.array
) -> None:

    nColorBins = 20  # how many shades of gray--20 is fine

    amp_range = np.quantile(
        spike_amps, [0.1, 0.9]
    )  # figure out how many colors we need
    color_bins = np.linspace(amp_range[0], amp_range[1], num=nColorBins)

    colors = gray_fn(nColorBins)  # creates grayscale based on # of divisions
    colors = colors[::-1]  # invert so that big spikes are darker

    """if I apply my depth correction to get absolute depths my depth range
    should go from 0-1000 to something like 400-1400. If I do this correction
    I need to flip my axis so do a correction of -1. Otherwise just keep the 
    axis the same"""

    if np.min(spike_depths) < 100:
        y_ax_corr = 1
    else:
        y_ax_corr = -1

    fig, ax = plt.subplots(figsize=(10, 8))

    """plot each set of spikes based on amplitudes"""
    for b in range(nColorBins - 1):
        these_spikes = np.squeeze(
            np.logical_and(spike_amps >= color_bins[b], spike_amps <= color_bins[b + 1])
        )
        ax.scatter(
            spike_times[these_spikes],
            y_ax_corr * spike_depths[these_spikes],
            color=colors[b, :],
            marker=".",
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Depth (µm)")

    y_depth_bin = 500  # Nick used 800 I use 500. Can be changed.

    for d in range(0, int(np.max(spike_depths)), y_depth_bin):
        tmp = spike_amps[
            np.logical_and(spike_depths >= d, spike_depths < d + y_depth_bin)
        ]
        I = np.squeeze(
            np.logical_and(
                spike_amps > np.mean(tmp) + 1.5 * np.std(tmp),
                (np.logical_and(spike_depths >= d, spike_depths < d + y_depth_bin)),
            )
        )
        drift_events = detect_drift_events(spike_times[I], spike_depths[I])
        try:  # if there are drift events plot them
            ax.scatter(drift_events[:, 0], y_ax_corr * drift_events[:, 1], color="r")
        except TypeError:
            print("No drift")

    sns.despine()
    plt.figure(dpi=1200)
    plt.show()


"""this function takes spike_times and the spike_depths. We create bins and count
with histogram. Then we look for local maxima (or minima) with the 
scipy.signal.find_peaks functions. We iterate over each extremum and assess
conditioanls to decide if it counts as drift or not. Then we return a matrix
nx3 of drift events with columns [time, depth, drift_size]"""


def detect_drift_events(spikeTimes: np.array, spikeDepths: np.array) -> np.array:

    drift_events = list()
    if len(spikeTimes) == 0:
        return None

    D = 2  # cut into 2 um pieces--Nick's default so I'm keeping it

    # bins = np.linspace(int(np.min(spikeDepths)-D), int(np.max(spikeDepths)+2*D), num= int((np.max(spikeDepths)+2*D-np.min(spikeDepths)-D)/D))
    bins = np.arange(
        int(np.min(spikeDepths) - D), int(np.max(spikeDepths) + D), D
    )  # not many negatives so arange should be stable enough
    h = np.histogram(spikeDepths, bins)[0]  # get the histogram
    # h = h[:-1] # originally histc doesn't delete the last bin MATLAB histcounts and numpy both autodelete the last
    bins = bins[:-1] + D / 2  # center the bins
    locs = find_peaks(h)[0]  # scipyfn-different bins ~potential drift events

    """in generate we iterate through potential drift events. We find the beginning
    posBegin and the end posEnd --with defaults if we can't find the exact location
    Nick uses the 0.05 correction so I'm keeping it. Then we skip this as a drift
    if it fails the if statement logical. Once we have our start and end we then
    find the depths we need and the times we need. Then we slowly chop things up
    and look to see if any of the depths are at least 6 away from the median of
    the spike depths and that we have at least ten spikes like this so that we
    don't count a drift event as a few randomly misplaced spikes, ie real drift
    should be a meaningful change in spike depths. If all these conditions are met
    we put in the time (t) in which they occurred. Which depth bins this occured
    in for plotting purposes, and then we keep track of the actual drift size"""
    for p in range(len(locs)):
        if h[locs[p]] < 0.3 * spikeTimes[-1]:

            continue
        try:
            posBegin = np.max(np.where(h[: locs[p]] < 0.05 * h[locs[p]]))
        except ValueError:
            posBegin = 0
        try:
            posEnd = np.min(np.where(h[locs[p] :] < 0.05 * h[locs[p]])) + locs[p] - 1
        except ValueError:
            posEnd = len(bins) - 1

        """I try bitwise so that if p < len(locs)-2 it fails before we reaches the
        impossible locs[p+1]. I'm not sure how matlab event evaluates this, but
        the shortcircuting although not elegant is functional"""
        if np.logical_and(p >= 1, posBegin < locs[p - 1]) or (
            p < len(locs) - 2 and posEnd > locs[p + 1]  # boolean rather than bitwise
        ):
            continue

        sub_spikes = np.squeeze(
            np.logical_and(spikeDepths > bins[posBegin], spikeDepths < bins[posEnd])
        )
        currentspikeDepths = spikeDepths[sub_spikes]
        currentspikeTimes = spikeTimes[sub_spikes]
        for t in np.arange(0, int(spikeTimes[-1]), 10):
            I = np.logical_and(currentspikeTimes >= t, currentspikeTimes <= t + 10)
            drift_size = bins[locs[p]] - np.median(currentspikeDepths[I])
            if np.abs(drift_size) > 6 and np.sum(I) > 10:
                drift_events += [
                    t + 5,
                    bins[locs[p]],
                    drift_size,
                ]  # append then will reshape
    drift_events = np.array(drift_events).reshape(
        (int(len(drift_events) / 3), 3)
    )  # just reshape to be a nx3 matrix [time, depth, drift_size]
    return drift_events


"""Makes a gray scale for color mapping"""


def gray_fn(m: int) -> np.array:
    g = np.linspace(0, m - 1, num=m - 1) / np.max(
        [m - 1, 1]
    )  # creates grayscale vector
    t = np.matlib.repmat(g, 1, 3).reshape(
        3, m - 1
    )  # repamt switched from vector to 3xm-1
    tT = t.T  # transpose the matrix to be m-1 x 3
    return tT
