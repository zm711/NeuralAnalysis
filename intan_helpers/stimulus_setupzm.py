# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:39:32 2022

@author: ZacharyMcKenzie
"""

import numpy as np
from intan_helpers.stimulushelpers import spike_prep
import statistics

"""dig_stim_setup takes a nChxnSampl datat and a list of the channels (nCh) as 
well as the sampleing rate. I also load in the dict. This isn't technically crucial
it's basically a holdover from the matlab"""


def dig_stim_setup(
    board_dig_in_data: np.array,
    board_dig_in_channels: dict,
    sample_rate: float,
    eventTimeTot: dict,
) -> dict:

    for stim in range(np.shape(board_dig_in_data)[0]):
        dig_channel = int(board_dig_in_channels[stim]["native_channel_name"][-1])
        print("Analyzing data from digital channel {chan}".format(chan=dig_channel))
        board_dig_current = board_dig_in_data[stim, :]
        event_lengths, event_times, trial_group = spike_prep(
            board_dig_current, sample_rate
        )
        # if board_dig_in_channels[stim]['native_channel_name'] == 'DIGITAL-IN-0'+str(stim+1):

        if len(event_lengths) != 0:
            eventTimeTot["DIG" + str(dig_channel)] = {}
            eventTimeTot["DIG" + str(dig_channel)]["EventTime"] = event_times
            eventTimeTot["DIG" + str(dig_channel)]["Lengths"] = event_lengths
            if trial_group != 0:
                eventTimeTot["DIG" + str(dig_channel)]["TrialGroup"] = trial_group
            else:
                eventTimeTot["DIG" + str(dig_channel)]["TrialGroup"] = np.ones(
                    (len(event_times),)
                )
    return eventTimeTot


"""barostat_stim_setup takes in analog to digital data as well as the framerate/sample rate
and converts this into into events to be loaded into eventTimes along with lengths and the
the trail groups. Using conditions it completes the digitization then uses the calc binary
core to generate the eventTimes"""


def barostat_stim_setup(
    board_adc_data: np.array, sample_rate: float, peak=False
) -> dict:
    """Below is the old baro code that only looks at the peak pressure rather than the
    ramping up to pressure as well. I set it to false since I don't analyze it"""
    eventTimeBaro = None
    if peak:
        print("Digitalizing Data")
        # start by organizing each trial group into its own boardx
        # Sam (who made barostat) says he set 3V=60mmHg
        board12 = np.logical_and(board_adc_data > 3.60, board_adc_data < 3.92)
        board11 = np.logical_and(board_adc_data > 3.19, board_adc_data < 3.35)
        board10 = np.logical_and(board_adc_data > 2.90, board_adc_data < 3.14)
        board09 = np.logical_and(board_adc_data > 2.63, board_adc_data < 2.88)
        board08 = np.logical_and(board_adc_data > 2.39, board_adc_data < 2.62)
        board07 = np.logical_and(board_adc_data > 2.18, board_adc_data < 2.35)
        board06 = np.logical_and(board_adc_data > 1.90, board_adc_data < 2.15)
        board05 = np.logical_and(board_adc_data > 1.64, board_adc_data < 1.88)
        board04 = np.logical_and(board_adc_data > 1.36, board_adc_data < 1.62)
        board03 = np.logical_and(board_adc_data > 0.90, board_adc_data < 1.15)
        board02 = np.logical_and(board_adc_data > 0.37, board_adc_data < 0.61)
        board01 = np.logical_and(board_adc_data > 0.10, board_adc_data < 0.36)
        baro_dig = np.squeeze(
            np.array(
                (
                    board01,
                    board02,
                    board03,
                    board04,
                    board05,
                    board06,
                    board07,
                    board08,
                    board09,
                    board10,
                    board11,
                    board12,
                ),
                dtype=int,
            )
        )
        print("Converting data to binaries")

        """First I calculate just the onset of the stimulus itself rather than
        the balloon. This seems to be less accurate, but it's how we started. I
        also need this to keep track of the trial groups for my actual analysis
        Below is code I adapted from HH's matlab code for removing fluctations in 
        barostat. I don't use, but I'm leaving this for now"""
        print("Removing analog fluctuations in data")
        freezeCutoff = 6.5 * sample_rate
        for data in range(len(baro_dig)):
            onB = np.where((np.diff(baro_dig[data]) == 1))[0]
            offB = np.where(np.diff(baro_dig[data]) == -1)[0]
            if len(onB) != len(offB):
                print("Please try not to end recordings mid-stimulus")
                onB = onB[:-1]  # if event cutoff we delete that event for code to work
            difference = offB - onB
            cutOff = difference < freezeCutoff
            toClear = np.zeros((2, len(np.where(cutOff == True)[0])))
            toClear[0] = onB[cutOff]
            toClear[1] = offB[cutOff]
            toClear = np.array(toClear, dtype=int)
            for clear in range(np.shape(toClear)[1]):
                baro_dig[data, toClear[0, clear] : toClear[1, clear] + 1] = 0

        print("Organizing data for final storage")
        # Put in our data to main eventTime structure
        eventTimes_raw = []
        eventTimelength_raw = []
        trialGroup_raw = []
        for data in range(len(baro_dig)):
            eventTimeLength, eventTimes, trialGroup = spike_prep(
                baro_dig[data], sample_rate
            )
            for events in range(len(eventTimes)):
                eventTimes_raw.append(eventTimes[events])
                eventTimelength_raw.append(eventTimeLength[events])
                trialGroup_raw.append(data)

        eventTimeBaro = np.zeros((3, len(eventTimes_raw)))
        eventTimeBaro[0] = eventTimes_raw
        eventTimeBaro[1] = eventTimelength_raw
        eventTimeBaro[2] = trialGroup_raw

    """Now I do my current analysis which start recording from the beginning
    of the balloon belowing up, which seems to be a little more accurate in 
    general. In order to remove noise I cut off any voltage less than 0.09 mV. 
    This doesn't do too much, but does help a little bit"""

    baro_dig2 = np.array(
        np.logical_and(board_adc_data > 0.09, board_adc_data > 0), dtype=int
    )
    eventTimesDig_length, eventTimesDig, _ = spike_prep(baro_dig2, sample_rate)

    """my barostat stimuli are programmed as 20s. So anything below 15 is going
    to be an analog fluctation that must be removed. Do this for the start times
    and for the lengths"""

    eventStart = eventTimesDig[eventTimesDig_length > 15]
    eventLength = eventTimesDig_length[eventTimesDig_length > 15]
    trial_group = np.zeros((len(eventStart),))

    """we go through each event and find its trial group to the nears 0.25. This allows
    for easy conversion to a pressure later in the code"""

    for idx in range(len(eventStart)):
        start = int(eventStart[idx]) * int(sample_rate)
        end = start + int(eventLength[idx]) * int(sample_rate)
        trial_group[idx] = int(
            valueround(statistics.mode(board_adc_data[start:end])) / 0.25
        )

    eventTimeTotal = np.zeros((3, len(eventStart)))
    eventTimeTotal[0] = eventStart
    eventTimeTotal[1] = eventLength
    eventTimeTotal[2] = trial_group
    return eventTimeBaro, eventTimeTotal


def valueround(x, precision=2, base=0.25):
    return round(base * round(float(x) / base), precision)

    # Below are deprecated code stuff--I'm leaving to make it easier to revert
    # but all is uncessary
    # =============================================================================
    #     DEPRECATED-LEAVING FOR HISTORICAL FOR NOW
    #     counter = int(0)
    #     timeMax = len(eventTimesDig)
    #     i=0
    #     stimS = np.zeros((len(eventTimes_raw),), dtype=int)
    #     stimE = np.zeros((len(eventTimes_raw),), dtype=int)
    #     while counter <timeMax-1:
    #         for event in range(len(eventTimesDig)):
    #             if np.abs(eventTimesDig[counter]-eventTimesDig[event]) <25:
    #                 stimS[i] = counter
    #                 stimE[i] = event
    #         counter = stimE[i]+1
    #         i+=1
    #
    #     eventStart = eventTimesDig[stimS]
    #     eventEnd = eventTimesDig[stimE]
    #     eventLength = np.subtract(eventEnd,eventStart)
    # =============================================================================

    # =============================================================================
    #
    #     """This sets trial groups for my analysis based on the trial groups I set
    #     at the beginning of the code"""
    #     tgFinal = np.zeros((len(trialGroup_raw)))
    #     for eventInd in range(len(eventStart)):
    #         for events in range(len(eventStart)):
    #             if np.abs(eventStart[eventInd] - eventTimes_raw[events]) < 5:
    #                 tgFinal[eventInd] = trialGroup_raw[events]
    #
    # =============================================================================
