#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 12:49:03 2022

@author: zacharymckenzie

This is a python version of the code from Nick Steinmetz's matlab respository 
sortingQuality. The general idea is to find the mahal distance between each spike in a 
cluster and other spikes within and without of the cluster to determine the purity of 
that cluster. The first two functions are translations of Nick's code to python
from the matlab. The last set of functions are based on the core ideas given by
Haris et al 2001 (also translated from the matlab implementation as given by Nick).

inputs are just the filepath, ie the directory with numpy files. From there is loads the
data and returns two values Isolation Distance, which is the mahalanhobis distance at 
which 50% of the spikes are contamination spikes. Intuition here is the farther out we 
go (increasing distance) the more isolated a spike must be

Then we do contamination Rate which estimates how many spikes in a cluster actually come
from other clusters

unitQuality = [nClu x 1]
contamination Rat
"""

import numpy as np
from scipy.spatial.distance import cdist
import os
from ..misc_helpers.genhelpers import getdir, savefile, findindex
from numba import jit
from .silhouette_score import silhouette_score

""" This is the initialization function. We pull in our PC features and then organize 
the current cluster features compared to the nearest other cluster features. We are 
limiting the overal number of PCs and looking for nearer neighbors for computational 
reasons. I'm just  translating Nick's code into python, so I'm not sure of specifize 
size reasons. The  original papers for these metrics were working with tetrodes and so 
were using 2D dimensionsx 4 channels for this first function we are seeing if the data 
has been curated into clusters vs the kilosort/phy templates that were found. If so we 
need to get the cluster data instead of the template data"""


def maskedClusterQuality(sp=None) -> tuple[np.array, np.array, dict]:
    if sp is not None:
        filename = sp["filename"]
        if filename in os.getcwd():
            if "pyanalysis" in os.getcwd():
                os.chdir("..")
        else:
            _, curr_dir, filename = getdir()
            os.chdir(curr_dir)

    qcvalues = {}
    print("loading data\n")

    try:
        pc_features = np.load("pc_features.npy")
    except FileNotFoundError:
        print("PC Features Loading Failed. File not Found. Verify Directory")

    try:
        pc_features_ind = np.load("pc_feature_ind.npy")
    except FileNotFoundError:
        print("Individual Features not found. Ensure file exists")

    print("Building features matrix from clusters/templates\n")
    try:
        spike_clusters = np.squeeze(
            np.load("spike_clusters.npy")
        )  # load spike clusters which are curated clusters (=templates if no curation)
        spike_templates = np.squeeze(
            np.load("spike_templates.npy")
        )  # templates are the phy values sorted out by Kilosort
        clusterIDs = list(set(np.squeeze(spike_clusters)))
        nClusters = len(clusterIDs)
        nSpikes = np.size(spike_clusters)
        nFet = 4
        nFetPerChan = np.shape(pc_features)[1]

        newFet = np.zeros((nSpikes, nFetPerChan, nFet))
        newFetInds = np.zeros((nClusters, nFet))

        """Iterate through each cluster so that we can compare each cluster to other 
        clusters"""

        for cluster in range(len(clusterIDs)):
            thisID = clusterIDs[cluster]
            theseSpikes = (
                spike_clusters == thisID
            )  # spikes only for our current cluster
            theseTemplates = spike_templates[
                theseSpikes
            ]  # templates for current cluster
            inclTemps, inst = count_unique(theseTemplates)  # use my countUnique below

            thisTemplate = inclTemps[inst == max(inst)]
            theseChans = pc_features_ind[thisTemplate, :nFet]
            newFetInds[cluster] = theseChans

            """Now we set up the features that we will use, ie features for the cluster 
            instead of the template"""

            for feat in range(nFet):
                thisChanInds = pc_features_ind == theseChans[feat]
                tempsWithThisChan, chanInds = findindex(thisChanInds)

                inclTempsWithThisFet = np.where(np.isin(inclTemps, tempsWithThisChan))[
                    0
                ]
                for temp in range(np.size(inclTempsWithThisFet)):
                    thisSubTemp = inclTemps[inclTempsWithThisFet[temp]]
                    tempschannindex = tempsWithThisChan == thisSubTemp
                    thisTfetInd = chanInds[tempschannindex]
                    theseSpikeTemplates = spike_templates == thisSubTemp
                    thSpikeTemps = np.squeeze(
                        np.logical_and(theseSpikes, theseSpikeTemplates)
                    )
                    newFet[thSpikeTemps, :, feat] = pc_features[
                        thSpikeTemps, :, thisTfetInd
                    ]

        pc_features = newFet  # load curated or non-curated values into variable
        pc_features_ind = newFetInds

    except FileNotFoundError:
        print("Spike clusters does not exist using spike templates instead")
        spike_clusters = np.load("spike_templates.npy")

    if np.shape(pc_features)[1] != 3:
        print("Error generating pc features. Problem in code fix please.")
        return

    else:
        print("computing cluster qualities....\n")
        unitQuality, contaminationRate, sil_score = masked_cluster_quality_sparse(
            spike_clusters, pc_features, pc_features_ind
        )
        print("Finalizing output")
        qcvalues["uQ"] = unitQuality
        qcvalues["cR"] = contaminationRate
        qcvalues["sil"] = sil_score
        savefile(filename + "qcvalues.npy", qcvalues)

        return unitQuality, contaminationRate, qcvalues


"""function to generate the unique values in a list and how many times those
values appear. Needed for matlab code, but ZM made a different pythonic 
implementation"""


@jit(nopython=True, cache=True)
def count_unique(x: np.array) -> tuple[int, int]:
    x = [np.int32(val) for val in x]
    values = [np.int32(x_val) for x_val in set(x)]
    instance = [np.int32(ins) for ins in range(0)]
    for val in values:
        instance.append(x.count(val))
    return values, instance


"""Now we set up our function to organize the thisCluster vs otherClusters. We have 
either brought into template data with pc features or the cluster data with pc features.
With these features we will now take each cluster/template and compare it to others 
using the PC space/features that we have pulled"""


def masked_cluster_quality_sparse(
    clu: np.array, fet: np.array, fetInds: np.array, fetNchans=0
) -> tuple[float, float]:
    """fet is a nSpike x 3 x 4 matrix and fetInd is a nClu x 4 matrix
    This is where Nick decided to stick with 4 features, so I left the default to be
    4"""

    if fetNchans == 0:
        fetNchans = min(4, np.shape(fetInds)[1])

    nFetPerChan: int = np.shape(fet)[1]
    fetN: int = fetNchans * nFetPerChan  # number of features total

    N: int = len(clu)

    clusterIDs = list(set(clu))

    unitQuality = np.zeros(
        len(clusterIDs),
    )  # memory allocation
    contaminationRate = np.zeros(
        len(clusterIDs),
    )  # memory allocation
    sil_score = np.zeros(len(clusterIDs))
    """Iterate through the clusters now to get the this vs other we start by 
    geting all the this feature space"""
    print("Quality metrics values being calculated")
    print("Cluster Number/ Isolation Distance/ Contamination Rate/ Silhouette Score")
    for cluster in range(len(clusterIDs)):
        theseSpikes = clu == clusterIDs[cluster]
        n = np.sum(theseSpikes)
        if n < fetN or n >= N / 2:
            unitQuality[cluster] = 0
            contaminationRate[cluster] = np.NaN
            sil_score[cluster] = np.NaN
            continue

        fetThisCluster = np.reshape(fet[theseSpikes, :, :fetNchans], (n, -1))

        theseChans = fetInds[cluster, :fetNchans]
        # nInd = 0
        # otherSpikes = np.size(clu) - np.size(np.where(theseSpikes)[0])
        fetOtherClusters = np.empty((0, np.shape(fet)[1], fetNchans))  #

        """And now we do the other feature space generation"""

        for nonCluster in range(len(clusterIDs)):
            if nonCluster != cluster:
                chansC2Has = fetInds[nonCluster]
                theseOtherSpikes = clu == clusterIDs[nonCluster]
                clusterList = list()
                for f in range(len(theseChans)):
                    if np.isin(theseChans[f], chansC2Has):
                        thisCfetInd = np.where(chansC2Has == theseChans[f])[0]
                        if f == 0:
                            myfetone = fet[theseOtherSpikes, :, thisCfetInd]
                            clusterList.append(myfetone)
                        elif f == 1:
                            myfettwo = fet[theseOtherSpikes, :, thisCfetInd]
                            clusterList.append(myfettwo)
                        elif f == 2:
                            myfetthree = fet[theseOtherSpikes, :, thisCfetInd]
                            clusterList.append(myfetthree)
                        else:
                            myfetfour = fet[theseOtherSpikes, :, thisCfetInd]
                            clusterList.append(myfetfour)
                    """else:
                        if f==0:
                            myfetone = np.zeros((np.shape(fet[theseOtherSpikes])[0], 3))
                        elif f==1:
                            myfettwo = np.zeros((np.shape(fet[theseOtherSpikes])[0], 3))
                        elif f==2:
                            myfetthree = np.zeros((np.shape(fet[theseOtherSpikes])[0], 3))
                        else:
                            myfetfour = np.zeros((np.shape(fet[theseOtherSpikes])[0], 3))"""
                if len(clusterList) != 0:
                    thisClusterOtherFet = np.stack((clusterList), axis=2)
                    if np.shape(thisClusterOtherFet)[2] != 4:
                        thisClusterOtherFet = np.concatenate(
                            (
                                thisClusterOtherFet,
                                np.zeros(
                                    (
                                        np.shape(thisClusterOtherFet)[0],
                                        3,
                                        (4 - np.shape(thisClusterOtherFet)[2]),
                                    )
                                ),
                            ),
                            axis=2,
                        )
                # spikesAlready = np.isin(thisClusterOtherFet, fetOtherClusters)
                # & np.all(spikesAlready,where=[False])
                if sum(np.isin(chansC2Has, theseChans) != 0):
                    fetOtherClusters = np.vstack(
                        (fetOtherClusters, thisClusterOtherFet)
                    )
                """
                for f in range(len(theseChans)):
                    if np.isin(theseChans[f], chansC2Has):
                        theseOtherSpikes = clu==clusterIDs[nonCluster]
                        thisCfetInd = np.argwhere(chansC2Has==theseChans[f])[0]
                        fetOtherClusters[nInd:nInd+sum(theseOtherSpikes),:,f] = fet[theseOtherSpikes,:,thisCfetInd]
                        #fetOtherClusters = np.append(fetOtherClusters, )
                if sum(np.isin(chansC2Has, theseChans))!=0:
                    nInd = nInd+sum(theseOtherSpikes)
                            
        fetOtherClusters = np.squeeze(fetOtherClusters) # new line to test"""
        if len(fetOtherClusters) != 0:
            fetOtherClusters = np.reshape(
                fetOtherClusters, (np.shape(fetOtherClusters)[0], -1)
            )
        else:  # this puts in a small array which allows for the Core function to fail.
            fetOtherClusters = np.array([0])
        uQ, cR, sil = masked_cluster_quality_core(
            fetThisCluster, fetOtherClusters
        )  # dist fn

        unitQuality[cluster] = uQ  # load each isolation distance into our final output
        contaminationRate[cluster] = cR * 100.00  # load the contaimnation rate
        sil_score[cluster] = sil

        print(
            "      {cluster}                 {uQ:.2f}               {cR:.2f}%        {sil:.2f}".format(
                cluster=clusterIDs[cluster], uQ=uQ, cR=cR * 100, sil=sil
            )
        )

    return unitQuality, contaminationRate, sil_score


"""Finally we are at the core where the data is ready to be processed. We bring in 
thisFeature space vs the otherFeatures space and compare the distances between these
spaces to see the distance between spikes. We calculate the mahal distance internally
followed by externally and compare

Explanations of this calculation are given in Haris et al. 2001 Cell Press
Along with Schmitzer-Torber et al. 2005 Neuroscience. For explanation of mahal
distance etc see those papers"""


def masked_cluster_quality_core(
    fetThisCluster: np.array, fetOtherClusters: np.array
) -> tuple[float, float]:
    n: int = np.shape(fetThisCluster)[0]
    nOther: int = np.shape(fetOtherClusters)[0]
    nfet: int = np.shape(fetThisCluster)[1]

    """First we make sure that the data is reasbonable to do this analysis. If our 
    current cluster is way too big (n>nOther) or we have more features than samples we 
    can't do this type of analysis"""

    if nOther > n and n > nfet:
        mean_sil_score = silhouette_score(fetThisCluster, fetOtherClusters)
        cov_fetThisCluster = np.linalg.inv(np.cov(fetThisCluster, rowvar=False))
        mean_fetThisCluster = np.reshape(np.mean(fetThisCluster, axis=0), (1, -1))
        md = cdist(
            fetOtherClusters, mean_fetThisCluster, "mahalanobis", VI=cov_fetThisCluster
        )
        md = np.sort(np.squeeze(md))
        md = md**2

        mdSelf = cdist(
            fetThisCluster, mean_fetThisCluster, "mahalanobis", VI=cov_fetThisCluster
        )  # this is internal distance

        mdSelf = np.sort(np.squeeze(mdSelf))
        mdSelf = mdSelf**2

        mdselfCount = len(mdSelf)  # for proportion below

        unit_quality = md[n]  # We take the raw isolation distance
        contamination_rate = 1 - (
            tipping_point(mdSelf, md) / mdselfCount
        )  # we calculate the point at which we are more contamination then not

    else:  # if we fail our conditions above we report that for this cluster
        unit_quality = 0
        contamination_rate = np.NaN
        mean_sil_score = np.NaN

    return unit_quality, contamination_rate, mean_sil_score


"""This function determines the point at which we take a ball of good spikes and bad 
spikes how many are bad spikes. We return this above and use it to get contamination 
percentage"""


@jit(nopython=True, cache=True)
def tipping_point(x: np.array, y: np.array) -> int:
    nX: int = len(x)

    ind = np.argsort(np.concatenate((x, y)))
    inds2 = np.argsort(ind)

    xInds = inds2[:nX]
    countdown = list(np.arange(nX, 0, -1))
    # countup = np.zeros((nX, len(countdown)))
    # for value in range(nX):
    #    countup[:, value] = xInds - value

    for count in range(nX):
        min_index = countdown[count] < (xInds - count)
        if len(np.nonzero(min_index)[0]) != 0:
            final_index = np.nonzero(min_index)[0]
            pos = final_index[0]
            break
        else:
            pos = nX

    return pos
