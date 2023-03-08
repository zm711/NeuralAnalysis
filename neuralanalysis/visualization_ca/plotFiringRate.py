#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 15:32:04 2022

@author: zacharymckenzie

This function takes the giant firingRate dictionary and breaks it up into a dataFrame, which
is easy to store as an excel--csv. Once in dataFrame it is easy to use seaborn for graphing.

INPUTS: firingRate dict 
        fid: optional name of file
        
OUTPUTS: firingRateDF a pandas dataframe that can be used for plotting
         it also saves a csv which can be loaded later with pandas.
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plotFiringRate(
    firing_rate_df,
    graph="v",
    response_df=None,
    labels=None,
) -> None:

    if labels is not None:
        for key in labels.keys():

            firing_rate_df.replace({"Trial Group": {key: labels[key]}}, inplace=True)

    plotCoreFR(firing_rate_df, graph)

    if response_df is not None:

        plot_by_window(firing_rate_df, response_df)


def plotCoreFR(firing_rate_df, graph) -> None:

    sns.set(
        rc={
            "legend.frameon": False,
        }
    )
    sns.set_style("white")

    plt.subplots(figsize=(10, 8))

    if graph == "v":
        ax = sns.violinplot(
            data=firing_rate_df,
            x="Trial Group",
            y="Spikes/sec",
            hue="Window",
            hue_order=["Rest", "Onset", "Sustained", "Offset", "Relief"],
        )
    else:
        ax = sns.lineplot(
            data=firing_rate_df,
            x="Trial Group",
            y="Spikes/sec",
            hue="Window",
            hue_order=["Rest", "Onset", "Sustained", "Offset", "Relief"],
        )
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)

    # sns.stripplot(data=firing_rate_df, x='Trial Group',y='Spikes/sec', hue='Window')

    plt.ylabel("Firing Rate (spikes/s)")

    sns.despine()
    plt.figure(dpi=1200)
    ax.legend(loc="upper left")
    plt.show()


def plot_by_window(firing_rate_df, response_neurons) -> None:

    for window in firing_rate_df["Window"].unique():
        sub_df = firing_rate_df.loc[firing_rate_df["Window"] == window]

        final_df = pd.merge(
            sub_df,
            response_neurons,
            how="right",
            on=["IDs", "File Hash", "Trial Group"],
        )
        final_df.dropna(axis=0, inplace=True)

        sns.set(
            rc={
                "legend.frameon": False,
            }
        )

        sns.set_style("white")

        plt.subplots(figsize=(10, 8))
        ax = sns.violinplot(
            data=final_df, x="Trial Group", y="Spikes/sec", hue="Sorter"
        )
        # sns.stripplot(data=final_df, x="Trial Group", y="Spikes/sec", hue='Sorter')
        sns.despine()
        plt.title(f"{window.title()}", fontsize=8)
        plt.figure(dpi=1200)
        ax.legend(loc="upper left")

        plt.show()
