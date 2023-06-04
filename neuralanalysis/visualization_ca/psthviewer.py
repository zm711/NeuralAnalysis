#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 15:15:53 2022

@author: zacharymckenzie
"""

import matplotlib.pyplot as plt

# from matplotlib import cm
import numpy as np
from scipy import signal
import seaborn as sns
from ..analysis import psthfunctions as psfn


"""This is my function to create a smoothed firing rate PSTH along with a raster plot of the same data
Inputs are psthvalues. Can be calculated from the Cluster Analysis class. It needs the general eventTimes
Dictionary. And the window over which one wants to graph. Though ideally the window should be whatever was
used during psthvalues calculation. There's an optional groupSep set to True, which analyzes trial Groups
separately. It can be set to false if someone prefers not to look at the data split that way."""


def plot_psth(
    psthvalues: dict,
    eventTimes: dict,
    labels: dict,
    raster_window: list[list[float, float]],
    groupSep:bool=True,
    eb:bool=False,
) -> None:
    """
    plots a smoothed firing rate plot on top of a raster plot

    Parameters
    ----------
    psthvalues : dict
        dictionary of psth values
    eventTimes : dict
        dictionary of stimulus values
    labels : dict
        dictionary to translate from int trial groups to str trial groups
    raster_window : list[list[float, float]]
        list of lists for each stimulus giving the time window for plotting
    groupSep : bool, optional
        Where to separate trial groups or to plot together. True is separate. The default is True.
    eb : bool, optional
        Whether to include error bar shading. The default is False.

    Returns
    -------
    None
        Plots of smoothed firing rates and raster plots

    """
    eventLst = list()
    for stimE in eventTimes.keys():
        eventLst.append(stimE)

    for index, stim in enumerate(psthvalues.keys()):
        """smoothing in general applies a gaussian to average adjacent bins. It should help make
        baseline right around 0 and the firing peaks should stay relatively. Based on my reading
        smoothing factor is more of an empiric decision. Too smooth would destory all peaks. No
        smoothing should jsut be a lineplot of a histogram data set"""
        param = int(input("input the smoothing factor\n"))
        gw = signal.windows.gaussian(
            round(param * 6), (round(param * 6) - 1) / 6
        )  # this takes std vs alpha for matlab version
        # std = (L-1)/2alpha (matlab alpha = 3)

        window = raster_window[index]
        windowS = window[0]
        windowE = (
            window[-1] + (window[-1] - window[0]) / 20
        )  # add 1/20 of stimulus length for spacing
        trial_groups = np.array(eventTimes[eventLst[index]]["TrialGroup"])
        event_lengths = eventTimes[eventLst[index]]["Lengths"]
        eventLength = np.mean(
            eventTimes[eventLst[index]]["Lengths"]
        )  # events should be same use mean d/t natural variation
        n_events = len(eventTimes[eventLst[index]]["EventTime"])
        raster_scale = np.floor(n_events / 100)  # correct raster sizing
        # windowLength = qcvalues[stim]['Window']
        # qcValues = {k: qcvalues[stim][k] for k in qcvalues[stim] if k not in ["Window"]}
        for cluster in psthvalues[stim].keys():
            ba = psthvalues[stim][cluster]["BinnedArray"]  # BinnedArray are the counts
            if np.sum(ba) == 0:
                continue
            bins = psthvalues[stim][cluster]["Bins"]  # these are the centers of bins
            tg = list(np.unique(trial_groups))
            n_groups = len(tg)
            n_bins = len(bins)
            smWin = gw / np.sum(gw)
            # baT = np.reshape(ba, (np.shape(ba)[1], np.shape(ba)[0]))
            baSm = np.zeros((np.shape(ba)[0], np.shape(ba)[1]))  # memory allocation

            for row in range(np.shape(ba)[0]):
                baSm[row] = signal.convolve(ba[row], smWin, mode="same") / (
                    bins[1] - bins[0]
                )  # convolution to apply gaussians

            if groupSep:
                psthSm = np.zeros((n_groups, n_bins))
                stderr = np.zeros((n_groups, n_bins))
                event_len = np.zeros((n_groups,))

                for group in range(n_groups):
                    psthSm[group] = np.mean(baSm[trial_groups == tg[group]], axis=0)
                    stderr[group] = np.std(baSm) / np.sqrt(np.shape(baSm)[0])
                    event_len[group] = np.mean(event_lengths[trial_groups == tg[group]])
            else:
                psthSm = np.mean(baSm, axis=0)
                stderr = np.std(baSm) / np.sqrt(np.shape(baSm)[0])

            """if we wnt our groups separated we need to organize by trial group 
            rather than by event number. (ie if we do random stimuli trial groups)
            then we need to reorganize our events for raster"""
            if groupSep:
                inds = np.argsort(trial_groups)
                BinIndex = np.transpose(np.nonzero(ba[inds, :]))
                tr = BinIndex[:, 0]
                b = BinIndex[:, 1]
                indices = np.argsort(b)
                b = b[indices]
                tr = tr[indices]
            else:
                BinIndex = np.transpose(np.nonzero(ba))
                tr = BinIndex[:, 0]
                b = BinIndex[:, 1]
                indices = np.argsort(b)
                b = b[indices]
                tr = tr[indices]

            raster_x, yy = psfn.rasterize(bins[b])
            raster_x = np.squeeze(raster_x)
            raster_y = yy + np.reshape(np.tile(tr.T, (3, 1)).T, (1, len(tr.T) * 3))
            raster_y = np.squeeze(raster_y)
            raster_y[1:-1:3] = raster_y[1:-1:3] + raster_scale  # apply scale

            minV = np.min(
                psthSm
            )  # get our minimum, but if > 0 . Then we want our ymin to be 0

            if minV > 0:
                minV = 0

            # just my color scheme. Could be anything really
            if groupSep == True:
                colorlist = [
                    "blue",
                    "green",
                    "#ff8c00",
                    "red",
                    "yellow",
                    "blue",
                    "black",
                    "blue",
                    "green",
                    "red",
                    "yellow",
                    "blue",
                    "green",
                    "black",
                    "red",
                    "pink",
                    "blue",
                    "orange",
                    "magenta",
                    "cyan",
                    "fuchsia",
                ]

                color1 = list()
                sns.set(rc={"legend.frameon": False})
                sns.set_style("white")
                plt.rc("axes", labelsize=50)
                plt.rc("axes", titlesize=50)
                for color in range(np.shape(psthSm)[0]):
                    color1.append(colorlist[color])

                fig1, (ax1, ax2) = plt.subplots(2, figsize=(20, 16), sharex=True)

                all_plots = list()
                for value in range(np.shape(psthSm)[0]):
                    if eb:
                        err_minus = psthSm[value] - stderr[value]
                        err_plus = psthSm[value] + stderr[value]
                    plots = ax1.plot(
                        bins,
                        psthSm[value],
                        color=color1[value],
                        linewidth=0.75,
                    )  # plot the psth values as lineplot
                    all_plots.append(plots[0])  # get handles for creating a legend
                    if eb:
                        ax1.plot(
                            bins,
                            err_minus,
                            color=color1[value],
                            linewidth=0.25,
                        )  # error line
                        ax1.plot(
                            bins, err_plus, color=color1[value], linewidth=0.25
                        )  # error line
                        ax1.fill_between(
                            bins,
                            err_minus,
                            err_plus,
                            color=color1[value],
                            alpha=0.2,
                        )  # faint fill between errors
                    ax1.plot(
                        [0, 0],
                        [minV, np.max(psthSm) + 1],
                        color="red",
                        linestyle=":",
                    )
                    ax1.plot(
                        [event_len[value], event_len[value]],
                        [minV, np.max(psthSm) + 1],
                        color=color1[value],
                        linestyle=":",
                    )  # mark out our stimulus
                    ax1.set(
                        xlim=(windowS, windowE),
                    )

                if eb:
                    ax1.set(ylim=(0, np.max(psthSm) + np.max(stderr) + 1))
                else:
                    ax1.set(ylim=(0, np.max(psthSm) + 1))

                if labels:  # if we have appropriate labels for the figure then add them
                    sub_label = labels[stim]
                    legend_list = list()
                    keys_list = sorted([float(key) for key in sub_label.keys()])
                    keys_list_str = [str(item) for item in keys_list]
                    for key in keys_list_str:
                        legend_list.append(sub_label[key])

                    ax1.legend(all_plots, legend_list)
                ax1.set_ylabel("Firing Rate (Hz)", fontsize=30)
                ax1.tick_params(axis="y", which="major", labelsize=30)
                ax2.plot(raster_x, raster_y, color="black")
                ax2.plot(
                    [0, 0],
                    [0, np.max(raster_y) + 1],
                    color="red",
                    linestyle=":",
                )

                for value in range(np.shape(psthSm)[0]):
                    ax2.plot(
                        [event_len[value], event_len[value]],
                        [0, np.max(raster_y) + 1],
                        color=color1[value],
                        linestyle=":",
                    )

                ax2.set(
                    xlim=(windowS, windowE),
                    xlabel="Time (s)",
                    ylim=(0, np.max(raster_y) + 1),
                )
                # ax2.set_yticks(np.arange(1, np.shape(raster_x)[0]+1), labels=eventLabels)
                ax2.set_ylabel("Event", fontsize=30)
                ax2.tick_params(axis="both", which="major", labelsize=30)
                ax2.set_xlabel("Time (s)", fontsize=30)
            else:
                fig1, (ax1, ax2) = plt.subplots(2, figsize=(20, 16), sharex=True)

                if eb:
                    err_minus = psthSm - stderr
                    err_plus = psthSm + stderr
                ax1.plot(bins, psthSm, color="k", linewidth=0.75)
                if eb:
                    ax1.plot(bins, err_minus, color="k", linewidth=0.25)
                    ax1.plot(bins, err_plus, color="k", linewidth=0.25)
                    ax1.fill_between(bins, err_minus, err_plus, color="k", alpha=0.2)
                ax1.plot(
                    [0, 0],
                    [minV, np.max(psthSm) + 1],
                    color="red",
                    linestyle=":",
                )
                ax1.plot(
                    [eventLength, eventLength],
                    [minV, np.max(psthSm) + 1],
                    color="red",
                    linestyle=":",
                )
                ax1.set(
                    xlim=(windowS, windowE),
                )
                if eb:
                    ax1.set(ylim=(0, np.max(psthSm) + np.max(stderr) + 1))
                else:
                    ax1.set(ylim=(0, np.max(psthSm) + 1))

                ax1.set_ylabel("Firing Rate (Hz)", fontsize=30)
                ax1.tick_params(axis="y", which="major", labelsize=30)

                ax2.plot(raster_x, raster_y, color="black")
                ax2.plot(
                    [0, 0],
                    [0, np.max(raster_y) + 1],
                    color="red",
                    linestyle=":",
                )
                ax2.plot(
                    [eventLength, eventLength],
                    [0, np.max(raster_y) + 1],
                    color="red",
                    linestyle=":",
                )

                ax2.set(
                    xlim=(windowS, windowE),
                    xlabel="Time (s)",
                    ylim=(0, np.max(raster_y) + 1),
                )
                # ax2.set_yticks(np.arange(1, np.shape(raster_x)[0]+1), labels=eventLabels)]
                ax2.set_ylabel("Event", fontsize=30)
                ax2.tick_params(axis="both", which="major", labelsize=30)
                ax2.set_xlabel("Time (s)", fontsize=30)

            """regardless of figure type following are the same"""

            fig1.suptitle(
                "Cluster Number {clu} Stim: {stim}".format(clu=cluster, stim=stim),
                fontsize=8,
                weight="bold",
            )

            plt.grid(False)

            # plt.rc("xtick", labelsize=20)
            # plt.rc("ytick", labelsize=20)
            # plt.rc("figure", labelsize=20)
            # plt.yticks(label="Event", fontsize=20)
            # plt.xticks(fontsize=20)
            # plt.xlabel("Time (s)", fontsize=20)
            plt.tight_layout()
            sns.despine()
            plt.figure(dpi=600)
            plt.show()
