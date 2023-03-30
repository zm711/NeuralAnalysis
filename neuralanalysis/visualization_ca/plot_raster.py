# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:30:48 2023

@author: ZacharyMcKenzie
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ..analysis import psthfunctions as psfn


def plot_raster(
    psthvalues: dict,
    eventTimes: dict,
    labels: dict,
    raster_window: list[list[float, float]],
    groupSep=True,
    eb=False,
) -> None:
    eventLst = list()
    for stimE in eventTimes.keys():
        eventLst.append(stimE)

    for index, stim in enumerate(psthvalues.keys()):
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

            if groupSep:
                event_len = np.zeros((n_groups,))
                for group in range(n_groups):
                    event_len[group] = np.mean(event_lengths[trial_groups == tg[group]])

            """if we wnt our groups separated we need to organize by trial group 
            rather than by event number. (ie if we do random stimuli trial groups)
            then we need to reorganize our events for raster"""
            if groupSep:
                inds = np.argsort(trial_groups)
                BinIndex = np.transpose(np.nonzero(ba[inds, :]))
                tr = BinIndex[:, 0]
                b = BinIndex[:, 1]
            else:
                BinIndex = np.transpose(np.nonzero(ba))
                tr = BinIndex[:, 0]
                b = BinIndex[:, 1]

            raster_x, yy = psfn.rasterize(bins[b])
            raster_x = np.squeeze(raster_x)
            raster_y = yy + np.reshape(np.tile(tr.T, (3, 1)), (1, len(tr.T) * 3))
            raster_y = np.squeeze(raster_y)
            raster_y[1:-1:3] = raster_y[1:-1:3] + raster_scale  # apply scale

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

                sns.set(rc={"legend.frameon": False})
                sns.set_style("white")

                fig1, ax2 = plt.subplots(2, figsize=(20, 16), sharex=True)

                ax2.plot(raster_x, raster_y, color="black")
                ax2.plot(
                    [0, 0],
                    [0, np.max(raster_y) + 1],
                    color="red",
                    linestyle=":",
                )

                for value in range(len(tg)):
                    ax2.plot(
                        [event_len[value], event_len[value]],
                        [0, np.max(raster_y) + 1],
                        color=colorlist[value],
                        linestyle=":",
                    )

                ax2.set(
                    xlim=(windowS, windowE),
                    xlabel="Time (s)",
                    ylim=(0, np.max(raster_y) + 1),
                )

                ax2.set_ylabel("Event", fontsize=30)
                ax2.tick_params(axis="both", which="major", labelsize=30)
                ax2.set_xlabel("Time (s)", fontsize=30)
            else:
                fig1, ax2 = plt.subplots(2, figsize=(20, 16), sharex=True)

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

            plt.tight_layout()
            sns.despine()
            plt.figure(dpi=600)
            plt.show()
