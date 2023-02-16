# -*- coding: utf-8 -*-

"""
Created on Tue Sep 27 17:09:07 2022

@author: ZacharyMcKenzie

This function generates a sparse PC matrix to allow us to compare spikes from 
'this cluster' compared to other nearbyclusters. It takes in sp and then number
 of chans and features in the pc space. I don't know if it can handle not 
defaults currently
"""


import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt


def plotPCs(sp: dict, nPCsPerChan=4, nPCchans=15) -> None:
    clu = np.squeeze(sp["clu"])
    clusterIDs = list(sorted(set(clu)))
    pc_feat = sp["pcFeat"].copy()
    pc_feat_ind = sp["pcFeatInd"].copy()
    spike_templates = sp["spikeTemplates"]

    sparsePCfeat = sparsePCs(
        pc_feat, pc_feat_ind, spike_templates, nPCsPerChan, nPCchans
    )  # local fn below

    """We iterate through each cluster and figure out its most important two 
    PCs to plot then we find the nearby other spikes"""

    for cluster in clusterIDs:
        thesePCs = sparsePCfeat[clu == cluster]
        meanPC = np.mean(thesePCs, axis=0)
        topChans = np.argsort(-abs(meanPC))[:2]
        otherSpikesIncl = (
            (sparsePCfeat[:, topChans[0]] != 0) == (sparsePCfeat[:, topChans[1]] != 0)
        ) == (clu != cluster)
        otherSpikesPCtemp = sparsePCfeat[otherSpikesIncl]
        otherSpikesPCs = otherSpikesPCtemp[:, topChans]
        otherPCsToPlotInds = np.random.permutation(np.shape(otherSpikesPCs)[0])
        otherPCsToPlot = otherSpikesPCs[otherPCsToPlotInds, :]
        thesePCsToPlot = thesePCs[:, topChans]

        plotPCsCore(thesePCsToPlot, otherPCsToPlot, cluster)


"""Generates sparsePC matrix from the pcFeat and pcFeatInd. It relies on scipy
sparse.csr_matrix function which is different than matlab's function. The inputs
are a tuple of (data, indices) where indices are a tuple of (row, column). data
row, and column all need to be size nData. This will follow a relationship of 
row(k), column(k) = data(k). Then we need to put in a shape as a tuple where we 
have r x s. In our function we take the nSpikes and the nChannels x nPCs per Channel
to give our shape. We generate rowinds by just iterating over the nSpikes
Our coulmns are based on the numbers of pCs we need to pull outfinally pcFeatRS will be 
our data that we feed into the sparse matrix

key point--since this based on a matlab function we need to reshape with a 
'fortran' order rather than 'c' order. so during reshape I use order='f'"""


def sparsePCs(
    pcFeat: np.array,
    pcFeatInd: np.array,
    spikeTemplates: np.array,
    nPCsPerChan: int,
    nPCchans: int,
) -> np.array:

    # pcFeat = sp["pcFeat"].copy()  # need copy since we mutate for the analysis
    # pcFeatInd = sp["pcFeatInd"].copy()
    # spikeTemplates = sp["spikeTemplates"]

    nPCchans = min(nPCchans, np.shape(pcFeat)[2])

    if nPCchans < np.shape(pcFeat)[2]:
        pcFeat = pcFeat[:, :, :nPCchans]
        pcFeatInd = pcFeatInd[:, :nPCchans]

    nPCsPerChan = min(nPCsPerChan, np.shape(pcFeat)[1])

    if nPCsPerChan < np.shape(pcFeat)[1]:
        pcFeat = pcFeat[:, :nPCsPerChan, :]

    nSpikes = np.shape(pcFeat)[0]

    nChannels = float(np.max(pcFeatInd[:]) + 1)

    rowInds = np.tile(np.linspace(0, nSpikes - 1, num=nSpikes), nPCchans * nPCsPerChan)
    # np.repeat(np.linspace(0, nSpikes-1, nSpikes), nPCchans*nPCsPerChan).reshape(-1, nPCchans*nPCsPerChan).T.flatten()
    colIndsTemp = np.zeros((nSpikes * nPCchans,))

    for q in range(nPCchans):
        colIndsTemp[(q) * nSpikes : (q + 1) * nSpikes] = np.squeeze(
            pcFeatInd[spikeTemplates, q]
        )

    colInds = np.zeros((nSpikes * nPCchans * nPCsPerChan,))

    for thisFeat in range(nPCsPerChan):
        colInds[thisFeat * nSpikes * nPCchans : (thisFeat + 1) * nSpikes * nPCchans] = (
            colIndsTemp * nPCsPerChan + thisFeat
        )

    pcFeatRS = np.zeros((nSpikes * nPCchans * nPCsPerChan,))

    for thisFeat in range(nPCsPerChan):
        pcFeatRS[
            thisFeat * nSpikes * nPCchans : (thisFeat + 1) * nSpikes * nPCchans
        ] = np.reshape(
            np.squeeze(pcFeat[:, thisFeat, :]), nSpikes * nPCchans, order="F"
        )
    S = scipy.sparse.csr_matrix(
        (pcFeatRS, (rowInds, colInds)),
        shape=(nSpikes, int(nChannels * nPCsPerChan)),
        dtype="float",
    )
    sparsePCfeat = S.toarray()

    return sparsePCfeat


"""Pretty simple plotting function which I abstracted from the main plot function
just for ease of editing below the others. We take the top two PC spaces for our
current spikes and plot these spikes vs other spikes"""


def plotPCsCore(
    thesePCsToPlot: np.array, otherPCsToPlot: np.array, cluster: str
) -> None:

    plt.subplots(figsize=(10, 8))
    plt.scatter(otherPCsToPlot[:, 0], otherPCsToPlot[:, 1], color="black", alpha=0.6)
    plt.scatter(thesePCsToPlot[:, 0], thesePCsToPlot[:, 1], color="red", alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(["Other Spikes", "Cluster Spikes"], fontsize=6, loc="upper right")
    plt.title(f"Cluster: {cluster}", size=6)
    ax = plt.gca()
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)

    sns.despine()  # you know I despine
    plt.figure(dpi=1200)

    plt.show()
