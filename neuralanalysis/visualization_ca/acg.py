#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 09:25:54 2022

@author: zacharymckenzie

This creates an autocorrelogram for each cluster---> ie a way to represent refractory 
period violationsvisusally. In general we use the psth function suite to create a 
histogram of data with respect to itself

INPUTS: sp our dict of our Kilosort/Phy data

OUTPUTS: graphs of ACGs for each cluster in sp
"""

import numpy as np
from ..analysis.histdiff import histdiff
import matplotlib.pyplot as plt
import seaborn as sns


def plotACGs(sp: dict, refract_time=0.002) -> None:

    """start by pulling out our data from sp. squeeze creates nX data rather
    than nX x 1 data which requires two values to specify"""
    spikeTimes = np.squeeze(sp["spikeTimes"])
    clu = np.squeeze(sp["clu"])
    cluster_IDs = list(sp["cids"])
    sample_rate = sp["sampleRate"]

    for cluster in cluster_IDs:  # we go through each cluster
        print(f"Analyzing cluster {cluster}")
        these_spikes = spikeTimes[clu == cluster]
        acg(
            these_spikes, sample_rate=sample_rate, cluster=cluster, ref_per=refract_time
        )


def acg(st: np.array, sample_rate: float, cluster: int, ref_per: float) -> None:

    # n_spikes = len(st)
    bin_size = 0.00025  # do small bins to get accurate sizes
    # n_bins = int(np.ceil((0.2- (1/(2*sample_rate)))/bin_size))
    # acg_bins = np.linspace(1/(2*sample_rate), 0.2, n_bins)
    # need a couple samples after 0 to 200 milliseconds with small bins for time
    acg_bins = np.arange(1 / (sample_rate * 2), 0.2, bin_size)  # generate the bins

    spike_counts, bin_centers = histdiff(st, st, acg_bins)

    """We check if we have enough spikes in our small window to actually show up
    if not we use the whole 200 ms. If we have enough we look at only 20 ms of data.
    People are allowed to put in their own refractory period. I default to 2 ms."""

    if sum(spike_counts[:81]) < 20:
        bin_centers_vals = np.concatenate((-np.flip(bin_centers), bin_centers))
        stairs_val = np.concatenate((np.flip(spike_counts), spike_counts))
    else:
        bin_centers_vals = np.concatenate(
            (-np.flip(bin_centers[:81]), bin_centers[:81])
        )
        stairs_val = np.concatenate((np.flip(spike_counts[:81]), spike_counts[:81]))

    decimal_points = len(
        str(ref_per).split(".")[1]
    )  # how many decimal places needed to compare to refractory period
    bin_centers_vals = np.array(
        [float(f"%.{decimal_points}f" % x) for x in bin_centers_vals]
    )  # convert x values to appropriate decimal places

    bin_centers_val_len = int(
        len(bin_centers_vals) / 8
    )  # divide to a small number of values for tick labels
    line2 = np.argwhere(
        abs(bin_centers_vals) == ref_per
    )  # put our lines at refractory period line

    bin_centers_vals = np.array(
        [float("%.3f" % x) for x in bin_centers_vals]
    )  # convert x-values to 3 decimal points for viusalization

    """Make our figure with refractory period lines and stairs. High resolution"""

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.stairs(stairs_val, color="black")
    ax1.plot(
        [line2[0], line2[0]], [0, np.max(stairs_val) + 5], color="red", linestyle=":"
    )  # refractory period lines
    ax1.plot(
        [line2[-1], line2[-1]], [0, np.max(stairs_val) + 5], color="red", linestyle=":"
    )  # refract lines
    ax1.set(
        xlim=(np.min(bin_centers_vals), np.max(bin_centers_vals)),
        xticks=np.linspace(0, len(bin_centers_vals), 9),
        ylabel="Spike Counts",
        xlabel="Time (s)",
    )
    ax1.set_xticklabels(bin_centers_vals[0:-1:bin_centers_val_len])  # labels a subset
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=12)
    plt.tight_layout()
    sns.despine()  # I like despined figures
    plt.title("ACG Neuron: {}".format(cluster), fontsize=8, weight="bold")
    plt.figure(dpi=1200)
    plt.show()
