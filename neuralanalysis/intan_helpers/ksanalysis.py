#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 17:36:48 2022
@author: zacharymckenzie


Quick load of data based on using Kilosort/Phy and Intan for recordings (.rhd files)

If using different recording device then you need a different way to generate eventTimes

If not using Phy output files then this code won't work at all

sp['spikeTimes'] = nSpikes np.array in seconds
sp['clu'] nSpikes np.array with curated classifications
sp['spikeTemplates'] np.array of orginal classifications
sp['tempScalingAmps'] = np.array for scaling the kilosort templates
sp['cids'] np.array of current ids
sp['cgs'] np.array of current quality
sp['filename'] str of file name
sp['sampleRate']: float: sample rate of device to switch from samples to seconds
sp['pcFeat'] np.array of pc features for each spike
sp['pcFeatInd'] np.array of pc features for each cluster
sp['temps'] = np.array of the kilosort templates used
sp['winv'] = np.array of whitening matrix
sp['xcoords'] np.array of xcoords of the recording probe
sp['ycoords'] np.array of ycoords of recording probe
sp['noise'] np.array of bool for noise clusterIDs

eventTimes is split into digital or analog channels (DIG1 or ADC1) and then subdivided
e.g. 
eventTimes['DIG1']['EventTimes'] np.array of onsets of stimuli
eventTimes['DIG1']['Lengths'] np.array of length of each stimulus (sec)
eventTimes['DIG1']['TrialGroup'] np.array of which grouping a stimulus belongs to 
eventTimes['DIG1']['Stim'] str the name of the stimulus
eventTimes['DIG1']['Rest']: float deprecated but was there for historic reasons

importantly 'TrialGroup' for digital channels can't be hard-coded a priori so fill the
trial group by hand or write a function which fills the trial group. For this to work in
future code it should just be a float value 1.0, 2.0, 3.0 for each stimulus condition.
"""


import os.path
import os

import numpy as np

from ..analysis.spsetup import loadsp
from ..intan_helpers.stim_alignment import stim_alignment
from ..intan_helpers.stimulushelpers import metadatafn, optoproc
from ..intan_helpers.zmbin import binConvert
from ..misc_helpers.genhelpers import getdir


def loadKS(baro=False):
    """loadKS automatically generates `sp` and `eventTimes` if using Intan.
    It runs `loadsp` to account for any new Phy curation. An *sp.npy file is save
    but is not loaded. Then we try to find the `*eventTimes.npy`. If this file
    doesn't not exist this function will generate it automatically for Intan users.
    Eventually other data types will be incorporated."""
    print("Select the directory with the *sp.npy file")
    sp: dict = loadsp()
    print("Select the directory with the *eventTimes.npy file.")
    try:
        eventTimes: dict = stim_alignment(baro=baro)
    except FileNotFoundError:
        print("eventTimes.npy was not found now select dir with .rhd file")
        binConvert()
        print("Select pyanalyis folder to generate the *eventTimes.npy file")
        eventTimes = stim_alignment(baro=baro)
    finally:
        for event in eventTimes.keys():
            if eventTimes[event].get("Stim", 0) == 0:
                eventTimes = metadatafn(eventTimes)
                eventTimes = optoproc(eventTimes)
                if sp["filename"] in os.getcwd():
                    np.save(sp["filename"] + "eventTimes.npy", eventTimes)
                else:
                    (
                        _,
                        _,
                        _,
                    ) = getdir()
                    np.save(sp["filename"] + "eventTimes.npy", eventTimes)
        return sp, eventTimes
