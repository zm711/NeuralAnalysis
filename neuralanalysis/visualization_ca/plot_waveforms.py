# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:57:34 2022

@author: ZacharyMcKenzie

function to plot the true, raw waveforms of the spike data as compared to the templates provided by Phy.

INPUTS: wf: a dict of the raw waveforms organized by c-order and f-order reading of the data. Wf has both
            individual spikes nWf as well as mean waveforms all organized by clusters and channels, ie
            wf[order][waveFormsMean] = nClu x nChan x nSample
            wf[order][waveForms] = nClu x nWf x nChan x nSample
            
        order: either c style or f style data reading. Default is fortran style, but if graphs look weird try
               running function with 'C'
               
        Ind: is just a flag whether you want the indiviual waves plotted over each other (True) or you want
             just the max (False). Default is True.
             
             
OUTPUTS: graphs of either individual waveforms or maximum waveform.

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_waveforms(wf: dict, order="F", Ind=True) -> None:

    """first we pull our data out of the dict"""

    waveFormsMean = wf[order]["waveFormsMean"]
    waveForms = wf[order]["waveForms"]
    clusterIDs = wf[order]["ClusterIDs"]

    for clu in range(
        len(clusterIDs)
    ):  # iterate through each cluster. grab the max channel and plot it's waveforms
        max_val = np.argwhere(waveFormsMean[clu] == np.min(waveFormsMean[clu]))[0]
        max_channel = max_val[0]

        if Ind == True:  # do all waveforms
            curr_waves = waveForms[clu, :, max_channel, :]
            curr_mean = waveFormsMean[clu, max_channel, :]

            plotWaveformsInd(curr_waves, curr_mean, clusterIDs[clu])

        else:  # plot just the mean of the max
            waveformMax = waveFormsMean[clu, max_channel, :]
            plotWaveformsMax(waveformMax, clusterIDs[clu])


def plotWaveformsInd(waveForms: np.array, meanwaveform: np.array, cluster: int) -> None:

    fig, ax = plt.subplots(figsize=(10, 8))
    if np.shape(waveForms)[0] > 300:
        wave_range = 300
    else:
        wave_range = np.shape(waveForms)[0]

    for wave in range(wave_range):  # changed this to 500 since 2000 was way too much
        ax.plot(np.linspace(-40, 41, num=82), waveForms[wave], color="gray")
    ax.plot(np.linspace(-40, 41, num=82), meanwaveform, color="k")

    ax.set(xlabel="Sample Numbers", ylabel="Voltage (μV)")
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=12)
    plt.tight_layout()
    plt.title(f"Cluster Number {cluster} Waveforms", fontsize=8, weight="bold")
    sns.despine()
    plt.figure(dpi=1200)
    plt.show()


def plotWaveformsMax(waveFormsMax: np.array, cluster: int) -> None:

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(np.linspace(-40, 41, num=82), waveFormsMax, color="black")
    ax.set(xlabel="Sample Numbers", ylabel="Voltage (μV)")
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=12)
    plt.tight_layout()
    plt.title(f"Cluster Number {cluster} Waveform", fontsize=8, weight="bold")
    sns.despine()
    plt.figure(dpi=1200)
    plt.show()
