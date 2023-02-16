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
import histdiff as hd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import ordhist
except ModuleNotFoundError:
    print("no rust code available")


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

    n_spikes = len(st)
    bin_size = 0.00002  # do small bins to get accurate sizes
    acg_bins = np.arange(1 / sample_rate / 2, 0.2, bin_size)  # generate the bins
    bins2 = acg_bins.copy()  # copy for using the rust version of code
    try:
        ordhist.bincenterpy(bins2)  # generates bin centers
        hd.rusthist(st, st, acg_bins)  # generates the counts
        x2 = bins2[:-1]  # set our x-axis delete extra column
        number2 = acg_bins[:-1]  # set our counts and delte one extra column
        number2 = number2 / bin_size / n_spikes  # conversion from Nick's code
    except NameError:  # if no rust default back to python implementation
        print("Using python algo")
        n2, x2 = hd.histdiff(st, st, acg_bins)
        number2 = n2 / bin_size / n_spikes

    """We check if we have enough spikes in our small window to actually show up
    if not we use the whole 200 ms. If we have enough we look at only 20 ms of data.
    People are allowed to put in their own refractory period. I default to 2 ms."""

    if sum(number2[:1001]) < 200:
        x2_vals = np.concatenate((-np.flip(x2), x2))
        stairs_val = np.concatenate((np.flip(number2), number2))
    else:
        x2_vals = np.concatenate((-np.flip(x2[:1001]), x2[:1001]))
        stairs_val = np.concatenate((np.flip(number2[:1001]), number2[:1001]))

    decimal_points = len(
        str(ref_per).split(".")[1]
    )  # how many decimal places needed to compare to refractory period
    x2_vals = np.array(
        [float(f"%.{decimal_points}f" % x) for x in x2_vals]
    )  # convert x values to appropriate decimal places

    x2_val_len = int(
        len(x2_vals) / 8
    )  # divide to a small number of values for tick labels
    line2 = np.argwhere(
        abs(x2_vals) == ref_per
    )  # put our lines at refractory period line

    x2_vals = np.array(
        [float("%.3f" % x) for x in x2_vals]
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
        xlim=(np.min(x2_vals), np.max(x2_vals)),
        xticks=np.linspace(0, len(x2_vals), 9),
        ylabel="Spike Counts",
        xlabel="Time (s)",
    )
    ax1.set_xticklabels(x2_vals[0:-1:x2_val_len])  # labels a subset
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=12)
    plt.tight_layout()
    sns.despine()  # I like despined figures
    plt.title("ACG Neuron: {}".format(cluster), fontsize=18, weight="bold")
    plt.figure(dpi=1200)
    plt.show()
