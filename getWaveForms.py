#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 20:25:49 2022

@author: zacharymckenzie

This is modified from the Matlab original found in Nick Steinmetz's repository spike-master
Originally contributed by C. Schoonover and A.Fink

It requires dataType int16 for us
wfWin is waveform window desired. Phy puts out 82 for the templates so i put default like that
nWf set to default of 2000. If less than 2000 spikes occurred it only uses the max number
spikeTimes is nspikeTimesx1. Can be presorted if only one cluster is desired.
clu are nSpikeTimesx1 with the cluster identity of each spike time. This can also be curated if only
certain spikes are designed. 

clu = clu[clusterIDs]
spikeTimes = spikeTimes[clu==clusterIDs]

OUTPUTS:
    unitIDs: the clusters which were processed
    spikeTimeKeeps [nClu, nWf] to show what was processed
    waveForms [nClu, nWf, nCh, nSwf] all waveforms of a sample
    waveFormsMean = [nClu, nCh, nSwf] only the mean of the waveform/channel
    
    nClu = number of clusters
    nCh = number of channels (4, 16, 32, or 64 for us)
    nWf = number of waveforms will attempt the min(actual number of waveforms, nWf default)
    nSwf = the actual number of samples analyses I have set to 82 since Phy uses 82 for its
    default
"""


import numpy as np
import os
import glob
from collections import namedtuple
import zmgenhelpers as zmhelp


def getWaveForms(
    sp: dict, nCh: int, datatype=np.int16, wfWin=[-40, 41], nWf=2000
) -> dict:
    # First we set up all of our data and initialization values
    if nCh is None:
        nCh = int(input("number of channels"))
    sample_rate: float = sp["sampleRate"]
    wfWin: list = wfWin
    nWf = int(nWf)
    spikeTimes: np.array = sp["spikeTimes"]
    spikeTimes = np.ceil(np.multiply(spikeTimes, sample_rate))  # convert to samples
    spikeClusters: np.array = sp["clu"]
    file_name = sp["filename"]
    if file_name in os.getcwd():
        if "pyanalysis" in os.getcwd():
            os.chdir("..")
    else:
        oldDir, filePath, filename = zmhelp.getdirzm()
        os.chdir(filePath)

    fileName = glob.glob("*.bin")[0]

    wf = {}  # allocation of final dictionary
    wf["C"] = {}
    wf["F"] = {}

    fileSize = os.path.getsize(fileName)  # need file size for memory mapping below
    temp = np.array(
        [0, 0, 0], dtype=datatype
    )  # this is a nifty trick to get the bits/bytes etc
    temp2 = temp.view(np.uint8)
    dataTypeNBytes = len(temp2) / 3
    nSamp = int(
        fileSize / (nCh * dataTypeNBytes)
    )  # we use the bytes to see the samples
    wfNSamples = int(wfWin[1] - wfWin[0] + 1)  # put our number of samples into variable

    """Now we are ready to load our binary file into memory to allow quicker 
    access during the next few steps. For a laptop <30gb of RAM this may be 
    difficult, but without this step everything would be much slower. Also for 
    32-bit systems (hopefully no one actually has, but just in case) the file 
    size limit would be 2gb."""

    print("Creating C & F-ordered memory maps...")
    # mmf = np.memmap(fileName, dtype="int16", mode="r", shape=(nCh, nSamp))
    mmfF = np.memmap(fileName, dtype="int16", mode="r", shape=(nCh, nSamp), order="F")

    chMap = np.squeeze(np.load("channel_map.npy"))  # grab the channel_map
    nChInMap = int(len(chMap))  # get number of channels

    clusterIDs = list(sp["cids"])

    nCluster = len(clusterIDs)

    """to explain below we are limited to the 4gb, so I look at number of 
    clusters need * number of waveforms * number of samples/waveform * 28 (size
    of int32) I plan to store in int16 which only take 26 bytes, but the two 
    bytes help contribute to a safety range. Then 3.91e-10 is the conversion 
    from bytes to gigabytes. It is easier to compare to the gigabyte limit of 
    pickling"""
    gbs_needed: float = (
        nCluster * nWf * wfNSamples * nChInMap * (28 * 3.91e-10)
    )  # file size need

    """if the file is going to be bigger than 4gb it will fail.So if it will be
    bigger than 3.9 then change the nWf to to be able to save. My logic is that
    82 samples is the standard so although that can be changed I don't want to 
    change it mid-program. Clusters can't be changed since it comes from the 
    data being analyzed at the time and number of channels in the map is fixed 
    for a particular recording although can change between recordings. So the 
    only value we can change is number of waveforms analyzed. So below I 
    calculate the max number of waveforms possible within a margin of error."""

    if gbs_needed > 3.9:
        nWf = int(round(3.9 / (nCluster * wfNSamples * nChInMap * 28 * 3.91e-10)))

    """memory allocation with nan's to tell the difference between 0 as a value
    and an actual not present value"""

    spikeTimeKeeps = np.empty((nCluster, nWf))
    spikeTimeKeeps[:] = np.nan
    # waveForms = np.empty((nCluster, nWf, nChInMap, wfNSamples))
    # waveForms[:] = np.nan
    waveFormsF = np.empty((nCluster, nWf, nChInMap, wfNSamples))
    waveFormsF[:] = np.nan
    # waveFormsMean = np.empty((nCluster, nChInMap, wfNSamples))
    # waveFormsMean[:] = np.nan
    waveFormsMeanF = np.empty((nCluster, nChInMap, wfNSamples))
    waveFormsMeanF[:] = np.nan

    """iterate through clusters and get the spikes which belong to that cluster"""
    for curUnit in range(nCluster):
        curSpikeTimes = spikeTimes[spikeClusters == clusterIDs[curUnit]]
        curUnitNSpikes = np.shape(curSpikeTimes)[0]
        spikeTimesRP = curSpikeTimes[np.random.permutation(curUnitNSpikes)]
        spikeTimeKeeps[curUnit, : min(nWf, curUnitNSpikes)] = sorted(
            spikeTimesRP[: min(nWf, curUnitNSpikes)]
        )

        """Grab those spikes from buffer and place them into our variable"""
        for curSpikeTime in range(min(nWf, curUnitNSpikes)):
            spikeKeepsIndexS = int(spikeTimeKeeps[curUnit, curSpikeTime] + wfWin[0])
            spikeKeepsIndexE = int(spikeTimeKeeps[curUnit, curSpikeTime] + wfWin[-1])
            # tmpWf = mmf[:nCh, spikeKeepsIndexS : spikeKeepsIndexE + 1]
            tmpWfF = mmfF[:nCh, spikeKeepsIndexS : spikeKeepsIndexE + 1]
            if np.shape(tmpWfF)[1] < 82:
                continue
            else:
                # waveForms[curUnit, curSpikeTime, :, :] = tmpWf[chMap]
                waveFormsF[curUnit, curSpikeTime, :, :] = tmpWfF[chMap]
        # waveFormsMean[curUnit] = np.nanmean(waveForms[curUnit], axis=0)
        waveFormsMeanF[curUnit] = np.nanmean(waveFormsF[curUnit], axis=0)
        print(
            "Completed unit {unit} of {number} of units.".format(
                unit=curUnit + 1, number=nCluster
            )
        )

    "Final loading of stuff into output variable wf"

    # wf["C"]["ClusterIDs"] = clusterIDs
    # wf["C"]["spikeTimeKeeps"] = spikeTimeKeeps
    # wf["C"]["waveForms"] = np.array(
    #    waveForms, dtype=datatype
    # )  # the datatype is int16 so there isn't really valuable to save as float64
    # wf["C"]["waveFormsMean"] = np.array(waveFormsMean, dtype=datatype)

    wf["F"]["ClusterIDs"] = clusterIDs
    wf["F"]["spikeTimeKeeps"] = spikeTimeKeeps
    wf["F"]["waveForms"] = np.array(waveFormsF, dtype=datatype)
    wf["F"]["waveFormsMean"] = np.array(waveFormsMeanF, dtype=datatype)

    try:  # first we try to save both 'c' and 'f' ordered files
        zmhelp.savefile(fileName[:-3] + "wf.npy", wf)
    except OverflowError:  # if this fails we delete the 'c' since 'f' are important
        try:
            wfF = (
                wf.copy()
            )  # deep copy because we will return the full dataset to the work space
            del wfF["C"]
            zmhelp.savefile(fileName[:-3] + "wf.npy", wfF)  # try to save 1/2 the data
            return wfF  # if this works just return this and try to use it
        except OverflowError:
            print("wf file is too large to be saved")
    finally:
        return wf  # even w/o save return the value to analyze during the session


"""INPUTS: wf the dictionary of the raw waveforms. Likely the "F" order of data, ie
Fortran rather than C style. This is a numpy - matlab issue. SP to get the ycoords of
the spikes. dataOrder defaults to 'F' (fortran), but if values seem nonsensical
it can be switched to 'C' to get data read from the binary with C ordering

OUTPUTS: max_waveforms: the max set of samples for each neuron
         waveform_dur: the duration of the spikes
         final_depth: depth of each unit
         waveform_amps: the amps of each unit
         shank_dict: a dict for multi shank probes to figure out which shank has
         which spike -currently just for H7


"""


def getWaveFormVals(
    wf: dict, sp: dict, dataOrder="F", depth=None, laterality=None
) -> namedtuple:
    xcoords: np.array = sp["xcoords"]
    shank_set = set(xcoords)  # set gets rid of duplicates
    ycoords: np.array = sp["ycoords"]
    mean_waveforms: np.array = wf[dataOrder][
        "waveFormsMean"
    ]  # first we get our maxwaveform
    mean_amplitudes = mean_waveforms.max(axis=2) - mean_waveforms.min(axis=2)

    """this is a center of mass for the ycoords weighted by the mean amplitudes
    this keeps it vectorized in numpy"""
    depth_raw = np.sum((mean_amplitudes * ycoords), axis=1) / np.sum(
        mean_amplitudes, axis=1
    )

    if depth:  # check if true depths given and correct if given
        final_depth = depth - depth_raw
    else:
        final_depth = depth_raw

    max_site = np.argmax(
        mean_waveforms.max(axis=2), axis=1
    )  # grab max waveform and look for max channel
    templates_max = np.zeros((np.shape(mean_waveforms)[0], np.shape(mean_waveforms)[2]))
    for curr_temp in range(np.shape(mean_waveforms)[0]):
        templates_max[curr_temp] = mean_waveforms[curr_temp, max_site[curr_temp], :]

    max_waveforms = templates_max
    waveform_trough = np.argmin(templates_max, axis=1)

    """this is a bit of a complicated list comprehension. Basically 'x' is each
    cluster identity that's why we go along the [0] axis of templates_max. Then
    we access this waveform with templates_max[x], but we also know that the AP
    max should only be after the trough--thus any max before the trough is noise
    so we want to go from the bin which is the trough to the end. Thus we use
    waveform_trough[x]:. np.argmax returns the index of that max and since we are
    start at the trough as 0 the argmax value will return the number of samples 
    away the max is and thus the duration of the waveform"""
    waveform_dur_raw = np.array(
        [
            np.argmax(templates_max[x, waveform_trough[x] :]) + 1  # index + 1 for time
            for x in range(np.shape(templates_max)[0])
        ]
    )
    waveform_dur = (
        waveform_dur_raw / sp["sampleRate"]
    )  # converts from samples to time (s)

    """same idea for getting our amplitudes. We get our real max value using the
    argmax as our index and iterate x through each cluster. To try to shorten
    the line a bit I use a separate line to convert to a numpy array"""

    amp_list = [
        templates_max[x, np.argmax(templates_max[x, waveform_trough[x] :])]
        - templates_max[x, waveform_trough[x]]
        for x in range(np.shape(templates_max)[0])
    ]
    waveform_amps = np.array(amp_list)

    if (
        len(shank_set) > 3
    ):  # single shanks have between 1-3 x positions more than this likely indicates multiple shanks
        """need to know the laterality for the claw--for dual shank will add more code later"""
        assert (
            laterality
        ), "For multi shank indicate whether right (r) of midline or left of midline (l)"

        x_pos = np.sum((mean_amplitudes * xcoords), axis=1) / np.sum(
            mean_amplitudes, axis=1
        )

        """just organize the shanks and then decide medial lateral identity with 
        conditional below"""
        x_shank = [
            1
            if x < 150
            else 2
            if x > 150 and x < 450
            else 4
            if x > 450 and x < 750
            else 3
            for x in x_pos
        ]

        if laterality.lower() == "r" or laterality.lower() == "right":
            med_lat = ["lateral" if x == 2 or x == 4 else "medial" for x in x_shank]
        elif laterality.lower() == "l" or laterality.lower() == "left":
            med_lat = ["lateral" if x == 1 or x == 3 else "medial" for x in x_shank]
        else:
            raise NameError("Laterality must be 'r' or 'l'")

        shank_dict = {}
        shank_dict["x_pos"] = x_pos
        shank_dict["x_shank"] = x_shank
        shank_dict["med_lat"] = med_lat
    else:
        shank_dict = None

    wfStorage = namedtuple(
        "wfStorage", "max_waveforms waveform_dur final_depth waveform_amps shank_dict"
    )
    wf_vals = wfStorage(
        max_waveforms, waveform_dur, final_depth, waveform_amps, shank_dict
    )

    return wf_vals


"""DEPRECATED


waveFormMetrics gets the weighted spikeDepths from the raw data the duration of
of max waveform, and the max waveform. 


INPUTS: wf the dictionary of the raw waveforms. Likely the "F" order of data, ie
Fortran rather than C style. This is a numpy - matlab issue. SP to get the ycoords of
the spikes. dataOrder defaults to 'F' (fortran), but if values seem nonsensical
it can be switched to 'C' to get data read from the binary with C ordering

OUTPUTS: maxWaveForm nClu x nSamp(default 82). To make plotting maxes easy
         tempDur nClu x 3. First column nSamp between peak and trough
                            Second column is a boolean indicating 1 if the trough
                            came before max and 0 if max came before trough, ie I
                            don't use the true max but the local max after the trough
                            assuming there is some sort of noise
        waveFormDepth nClu  An array of the weighted depths of the spikes using the
                            amplitude at each channel
                            
        waveAmps nClu x 2 First column are the differences between max value and min value for the
                          max waveform for each cluster
                          Second column is a booleean 1 indicating if the trough
                          came before max and 0 if max came before trough, ie I
                          don't use the true max but the local max after the trough
                          assuming there is some sort of noise
                          
                          
"""


def waveFormMetrics(wf, sp, dataOrder="F", depth=None):
    # xcoords = sp['xcoords']
    ycoords = sp["ycoords"]

    mean_waveforms = wf[dataOrder]["waveFormsMean"]  # first we get our maxwaveform

    """memory allocation to speed things up later"""
    waveFormDepth = np.zeros((np.shape(mean_waveforms)[0]))
    relativePeaks = np.zeros((np.shape(mean_waveforms)[0], np.shape(mean_waveforms)[1]))
    weightScaleNorm = np.zeros(
        (np.shape(mean_waveforms)[0], np.shape(mean_waveforms)[1])
    )
    weightScale = np.zeros((np.shape(mean_waveforms)[0], np.shape(mean_waveforms)[1]))
    clusterAbsPeak = np.zeros(np.shape(mean_waveforms)[0])
    channelIndex = np.zeros(np.shape(mean_waveforms)[0])
    maxWaveform = np.zeros((np.shape(mean_waveforms)[0], np.shape(mean_waveforms)[2]))
    tempDur = np.zeros((np.shape(mean_waveforms)[0], 3))
    waveAmps = np.zeros((np.shape(mean_waveforms)[0], 2))

    """need to iterate through each cluster and create our weighting metric
    to decide final depth. So this is math for getting the highest value and 
    then we standardize to that max value"""

    for cluster in range(np.shape(mean_waveforms)[0]):
        print(
            f"Analyzing waveset {cluster + 1} of {np.shape(mean_waveforms)[0]} waveforms"
        )
        clusterAbsPeak[cluster] = np.min(mean_waveforms[cluster])
        for channel in range(np.shape(mean_waveforms)[1]):
            relativePeaks[cluster, channel] = np.min(mean_waveforms[cluster, channel])

            weightScaleNorm[cluster, channel] = abs(
                relativePeaks[cluster, channel] / clusterAbsPeak[cluster]
            )

            if weightScaleNorm[cluster, channel] == 1:
                channelIndex[cluster] = channel
            channelIndex = np.array(channelIndex, dtype="int")

        for channel in range(np.shape(mean_waveforms)[1]):
            weightScale[cluster, channel] = (
                weightScaleNorm[cluster, channel]
            ) / np.sum(weightScaleNorm[cluster])
            waveFormDepth[cluster] += weightScale[cluster, channel] * ycoords[channel]

        maxWaveform[cluster] = mean_waveforms[cluster, channelIndex[cluster], :]

        """Now we find our trough and crest in order to calculate duration of the
        action potential"""

        waveform_trough = np.where(
            maxWaveform[cluster] == np.min(maxWaveform[cluster])
        )[0][0]
        waveform_max = np.where(maxWaveform[cluster] == np.max(maxWaveform[cluster]))[
            0
        ][0]
        if waveform_trough < waveform_max:
            tempDur[cluster, 0] = waveform_max - waveform_trough
            tempDur[cluster, 1] = 1

            waveAmps[cluster, 0] = (
                maxWaveform[cluster, waveform_max]
                - maxWaveform[cluster, waveform_trough]
            )
            waveAmps[cluster, 1] = 1

        else:
            waveform_max = np.where(
                maxWaveform[cluster] == np.max(maxWaveform[cluster, waveform_trough:])
            )[0][0]
            tempDur[cluster, 0] = waveform_max - waveform_trough
            tempDur[cluster, 1] = 0
            waveAmps[cluster, 0] = (
                maxWaveform[cluster, waveform_max]
                - maxWaveform[cluster, waveform_trough]
            )
            waveAmps[cluster, 1] = 0

        tempDur[cluster, 2] = tempDur[cluster, 0] * 1000 / sp["sampleRate"]

    """if depth exists (ie we have given the depth of the probe) then we correct
    our depths for this"""
    if depth:
        waveFormDepth = depth - waveFormDepth

    return maxWaveform, tempDur, waveFormDepth, waveAmps
