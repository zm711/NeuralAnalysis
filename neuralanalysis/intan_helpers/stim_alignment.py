#!/usr/bin/env python3
"""
Created on Wed Jun 15 11:02:38 2022

@author: ZacharyMcKenzie

This function Organizes digital and analog data into the intan. It is based on finding
binary differences (1 & 0s) in the intan inputs. It is extracted from npy if it already 
exists otherwise it reads the data from the rhd file. Then it organizes the data into
a large dictionary structure which is saved.

"""

from . import stimulus_setupzm as stim
import os
import glob
import numpy as np
from ..misc_helpers.genhelpers import getdir, savefile, loadvalues


def stim_alignment(baro=False) -> dict:

    _, currPath, _ = getdir()
    print("Setting Path")
    os.chdir(currPath)
    eventTimePresent: list = glob.glob("*eventTimes.npy")
    if len(eventTimePresent) != 0:
        eventTimes = np.load(eventTimePresent[0], allow_pickle=True)[()]
        return eventTimes
    eventTimeTot = {}

    intan = loadvalues()  # returns namedtuple that I can pull values otu of

    # board_adc_channels = intan.board_adc_channels
    board_adc_data: np.array = intan.board_adc_data
    board_dig_in_data: np.array = intan.board_dig_in_data
    board_dig_in_channels: dict = intan.board_dig_in_channels

    sample_rate: float = intan.frequency_parameters["amplifier_sample_rate"]

    try:

        board_dig_in_data.any()
        print("dig")
        board_dig_in_data = board_dig_in_data
        eventTimes = stim.dig_stim_setup(
            board_dig_in_data, board_dig_in_channels, sample_rate, eventTimeTot
        )
    except AttributeError:
        print("No dig data")

        eventTimes: dict = eventTimeTot

    try:
        board_adc_data.any()
        print("adc")
        board_adc_data = np.squeeze(board_adc_data)
        eventTimeBaro, eventTimewRamp = stim.barostat_stim_setup(
            board_adc_data, sample_rate, peak=baro
        )
        if baro:
            eventTimes["ADC1"] = {}
            eventTimes["ADC1"]["EventTime"] = eventTimeBaro[0]
            eventTimes["ADC1"]["Lengths"] = eventTimeBaro[1]
            eventTimes["ADC1"]["TrialGroup"] = eventTimeBaro[2]
        eventTimes["ADC1tot"] = {}
        eventTimes["ADC1tot"]["EventTime"] = eventTimewRamp[0]
        eventTimes["ADC1tot"]["Lengths"] = eventTimewRamp[1]
        eventTimes["ADC1tot"]["TrialGroup"] = eventTimewRamp[2]
    except AttributeError:
        print("No adc data")
    # if os.path.isfile(glob.glob('*board_adc_data*')):
    # filename2 = glob.glob('*board_dig_in_data')[0]
    # board_adc_data = np.load(filename2, allow_pickle=True)

    if os.path.basename(os.path.normpath(os.getcwd())) == "pyanalysis":
        filename: str = glob.glob("*npy")[0]
        savefile(filename[:-9] + "eventTimes" + ".npy", eventTimes)

    return eventTimes
