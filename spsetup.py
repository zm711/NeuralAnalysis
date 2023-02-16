# -*- coding: utf-8 -*-
"""
Python Loading of Numpy Files

@author: ZacharyMcKenzie
"""
import numpy as np
import os
from zmgenhelpers import readCGSfile, savefile, getdirzm
from stimulushelperszm import paramread


def loadsp() -> dict:

    """First I grab the directory where are numpy files are open that directory and save
    the title in case we decide to save things later"""

    old_dir, open_file, fileName = getdirzm()

    sp = {}  # Our final dict structure that we will return

    """First we read out our spikeTimes get our sample rate and load sample rate to sp"""

    spikeTimesRaw = np.load("spike_times.npy")  # spike times in sample numbers
    sampleRate: float = paramread()  # this sample rate
    spikeTimes: np.array = (
        spikeTimesRaw / sampleRate
    )  # in time(s) rather than sample number
    sp["sampleRate"] = sampleRate

    spikeTemplates: np.array = np.load("spike_templates.npy")

    """spike_clusters are post curation ie merges/splits. So we check for merges
       and splits and load those values as clu if they exist. Otherwise we just take
       the template data which are what phy originally sees from kilosort"""

    if os.path.isfile("spike_clusters.npy"):
        clu: np.array = np.load("spike_clusters.npy")
    else:
        clu: np.array = spikeTemplates

    tempScalingAmps: np.array = np.load("amplitudes.npy")

    """Collect the PC features for isolation analyses etc later in code"""

    pcFeat: np.array = np.load(
        "pc_features.npy"
    )  # nSpikes x nFeatures x nLocalChannels
    pcFeatInd: np.array = np.load("pc_feature_ind.npy")  # nTemplates x nLocalChannels

    """Now I get the cluster group labels mua, good, unsorted, noise. If not curated 
    then everything is unsorted, but if we have labeled noise then this will exclude 
    those values from being loaded into sp"""

    cgsfile = ""
    if os.path.isfile("cluster_groups.csv"):
        with open("cluster_groups.csv", "r") as c:
            cgsfile = c.readlines()
    elif os.path.isfile("cluster_group.tsv"):
        with open("cluster_group.tsv", "r") as c:
            cgsfile = c.readlines()
    else:
        print("no cgs information provided")

    """Now if we have curated our data in phy we need to account for this curation
    by only looking at the curated data (ie don't look at noise)"""

    if len(cgsfile) != 0:
        cids, cgs = readCGSfile(cgsfile)

        NoiseCluster = []

        for index, label in enumerate(cgs):
            if label == 0:
                NoiseCluster.append(cids[index])

        NoiseCluster = np.array(NoiseCluster)

        if len(set(cgs)) == 1 and 0 in set(cgs):
            cgs = 3 * np.ones(len(set(np.squeeze(clu))))
            cids = np.array(list(set(np.squeeze(clu))))
        else:
            cgs = np.array(cgs)
            cids = np.array(cids)

        """need to account for if kilosort does merges or splits that phy doesn't 
        account for. So I look at some sets of what values could be and if they don't 
        exist I remove them from the pcFeatInd since those guys must've been split or 
        merged"""

        """
        clu_set = set(cids) # our list of current cluster ids
        possible_clu_set = set(range(np.shape(pcFeatInd)[0])) # range of the pc feat space

        if len(clu_set) != len(possible_clu_set):
            merge_splits = list()
            for value in possible_clu_set:
                if value in clu_set:
                    merge_splits.append(value) # append values still present so we can index and delete values lost

            pcFeatInd = pcFeatInd[merge_splits,:] # this lets us get rid of any pc spaces not in use
        """

        """np.isin is equivalent to matlab ismember(). invert this to take all our values
        which are not within our noise dataset"""

        st = spikeTimes[np.isin(clu, NoiseCluster, invert=True)]
        tempScalingAmps = tempScalingAmps[np.isin(clu, NoiseCluster, invert=True)]
        spikeTemplates = spikeTemplates[np.isin(clu, NoiseCluster, invert=True)]

        pcFeat = pcFeat[np.squeeze(np.isin(clu, NoiseCluster, invert=True))]
        # pcFeatInd = pcFeatInd[np.squeeze(np.isin(cids, NoiseCluster, invert=True))] Nick's code doesn't do this...

        clu = clu[np.isin(clu, NoiseCluster, invert=True)]

        cgs = cgs[np.isin(cids, NoiseCluster, invert=True)]
        noise = np.isin(cids, NoiseCluster)
        cids = cids[np.isin(cids, NoiseCluster, invert=True)]

    else:
        clu = spikeTemplates
        cids = np.unique(spikeTemplates)
        cgs = 3 * np.ones((len(cids),))
        st = spikeTimes
        noise = np.nan

    """reading in more npy data"""

    coords: np.array = np.load("channel_positions.npy")
    ycoords = coords[:, 1]  # convert coords into x positions and y positions
    xcoords = coords[:, 0]

    temps = np.load("templates.npy")  # kilosort/phy action potential templates

    winv = np.load("whitening_mat_inv.npy")

    """Finally loading on the data into the sp dict for easy return and access"""

    sp["spikeTimes"] = st
    sp["spikeTemplates"] = spikeTemplates
    sp["clu"] = clu
    sp["tempScalingAmps"] = tempScalingAmps
    sp["cgs"] = cgs
    sp["cids"] = cids
    sp["xcoords"] = xcoords
    sp["ycoords"] = ycoords
    sp["temps"] = temps
    sp["winv"] = winv
    sp["pcFeat"] = pcFeat
    sp["pcFeatInd"] = pcFeatInd
    sp["filename"] = fileName
    sp["noise"] = noise

    savefile(fileName + "sp.npy", sp)
    return sp
