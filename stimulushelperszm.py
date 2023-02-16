#! /bin/env python
"""
Created on Tue Jun 14 13:40:04 2022

@author: ZacharyMcKenzie
"""

import numpy as np
import glob


def spike_prep(
    board_dig_data: np.array, frameRate: float, trialGroup=0
) -> tuple[np.array, np.array, np.array]:

    # frameRate = paramread()
    eventTimeLength, eventTime = calculate_binary(board_dig_data, frameRate)
    # if trialGroup == 0:
    # number = len(eventTime)
    # trialGroupList = list(map(int,input("\nEnter the numbers : ").strip().split()))[:number]
    # trialGroup = np.array(trialGroupList)
    return eventTimeLength, eventTime, trialGroup


"""this helper function calculates onset and offsets of 1s and 0s"""


def calculate_binary(binary: np.array, sample_rate: float) -> tuple[np.array, np.array]:
    binary_array = np.array(np.squeeze(binary), dtype=int)
    onset = np.where(np.diff(binary_array) == 1)[0] / sample_rate
    offset = (
        np.where(np.diff(binary_array) == -1)[0] / sample_rate
    )  # -1 end of stim for signed integers
    if len(offset) == 0:
        offset = (
            np.where(np.diff(binary_array) == 4294967295)[0] / sample_rate
        )  # unsigned int diff has to wrap so this is what we would wrap to currently

    # if the stimulus was on before starting or on after ending need to correct for this
    if binary_array[0] == 1:
        onset = np.pad(onset, (1, 0), "constant", constant_values=0)
    if binary_array[-1] == 1:
        offset = np.pad(
            offset, (0, 1), "constant", constant_values=binary_array[-1] / sample_rate
        )
    event_time_length = offset - onset
    event_times = onset

    return event_time_length, event_times


"""this function gets the sample frequency I use in the spsteupzm function"""


def paramread() -> float:
    if glob.glob("params.py"):
        with open("params.py", "r") as p:
            params = p.readlines()
        paramList = params[4].split()
        frequency = float(paramList[-1])
        return frequency
    else:
        print("Find params.py file in folder")


"""insert trial groups and eventTimes data"""


def metadatafn(eventTimes: dict) -> dict:
    stimuli = eventTimes.keys()
    for stim in stimuli:
        eventTimes[stim]["EventTime"] = eventTimes[stim]["EventTime"][
            eventTimes[stim]["Lengths"] > 0.01
        ]
        eventTimes[stim]["TrialGroup"] = eventTimes[stim]["TrialGroup"][
            eventTimes[stim]["Lengths"] > 0.01
        ]
        eventTimes[stim]["Lengths"] = eventTimes[stim]["Lengths"][
            eventTimes[stim]["Lengths"] > 0.01
        ]
        while len(eventTimes[stim]["EventTime"]) != len(eventTimes[stim]["TrialGroup"]):

            print(
                "Entering trial groups for {stim}. There are {event} events".format(
                    stim=stim, event=len(eventTimes[stim]["EventTime"])
                )
            )
            # eventTimes[stim]['TrialGroup'] = np.array(list(map(int,input("\nEnter the numbers: ").strip().split()))[:len(eventTimes[stim]['EventTime'])])
            eventTimes[stim]["TrialGroup"] = np.ones(
                (len(eventTimes[stim]["EventTime"]),)
            )
            answer = input("is your data organized by trial group? (y/n)")
            if answer == "y":
                answer2 = int(input("how many events per trial group?"))
                trials = list()
                for event in range(int(len(eventTimes[stim]["EventTime"]) / answer2)):
                    trials += list(event * np.ones((answer2,)))
                trials = np.array(trials)
                eventTimes[stim]["TrialGroup"] = np.array(trials)
                break
            else:
                break

        eventTimes[stim]["Stim"] = (
            input("Please enter stimulus name for {stim}\n".format(stim=stim))
            .strip()
            .title()
        )
        try:
            eventTimes[stim]["Rest"] = float(
                input("Please enter the rest period for {stim}\n".format(stim=stim))
            )
        except ValueError:
            eventTimes[stim]["Rest"] = np.nan

    return eventTimes


"""Processing opto data from intan"""


def optoproc(eventTimes: dict) -> dict:
    print("Processing Opto-Data")
    optochannel = ""
    for stim in eventTimes.keys():
        if (
            eventTimes[stim]["Stim"].title() == "Opto"
            or eventTimes[stim]["Stim"].title() == "Optogenetics"
        ):
            optochannel = stim
        else:
            optochannel = 0
    if optochannel == 0:
        optochannel = input(
            "Please input dig channel where opto data recorded. [Enter] for no-Opto.\n"
        )
    if len(optochannel) == 0:
        return eventTimes
    else:
        eventTimeOpto = eventTimes[optochannel]["EventTime"]

        stimLength = int(
            input(
                "length of train width (not pulse width) to assess? Enter 0 for default"
            )
        )
        if len(stimLength) == 0:
            stimLength = 2

        pulseNumber = 0

        for events in eventTimeOpto:
            if abs(eventTimeOpto[0] - eventTimeOpto[events]) < stimLength:
                pulseNumber += 1

        totalEvents = len(eventTimeOpto) / pulseNumber

        eventTimeOptoFinal = []

        eventTimeOptoLenFinal = []

        if totalEvents - int(totalEvents) == 0:
            for event in range(len(totalEvents)):
                eventTimeOptoFinal[event] = eventTimeOpto[event * pulseNumber]
                eventTimeOptoLenFinal[event] = (
                    eventTimeOpto[(event + 1) * pulseNumber]
                    - eventTimeOpto[event * pulseNumber]
                )
        else:
            print("Did not complete full stim cycle. Adjusting calculations.")
            totalEventCeiling = np.ceil(eventTimeOpto) / pulseNumber
            finalEvent = len(eventTimeOpto)
            for event in range(totalEventCeiling):
                eventTimeOptoFinal[event] = eventTimeOpto[event * pulseNumber]
                if event < totalEventCeiling:
                    eventTimeOptoLenFinal[event] = (
                        eventTimeOpto[(event + 1) * pulseNumber]
                        - eventTimeOpto[event * pulseNumber]
                    )
                else:
                    eventTimeOptoLenFinal[event] = (
                        eventTimeOpto[finalEvent] - eventTimeOpto[event * pulseNumber]
                    )

        eventTimeOptoTrialFinal = np.ones(len(eventTimeOptoFinal))

        eventTimes["OptoTrain"] = {}
        eventTimes["OptoTrain"]["EventTime"] = np.array(eventTimeOptoFinal)
        eventTimes["OptoTrain"]["Lengths"] = np.array(eventTimeOptoLenFinal)
        eventTimes["OptoTrain"]["TrialGroup"] = eventTimeOptoTrialFinal
        eventTimes["OptoTrain"]["Stim"] = "OptoTrain"
        eventTimes["OptoTrain"]["Rest"] = 2  # place filler for now

        return eventTimes
