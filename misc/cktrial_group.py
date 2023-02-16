# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:27:42 2022

@author: CharlineKambrun

This is a function to allow Charline to get correct trial groupings.
It needs sp--just to get filename. Readonly access.

Then it needs eventTimes--should have [stim]['TrialGroup'] with a list
of 1's. This is what we need to change. We need to know the number of 
distinct trial groups `n_tgs`. Then we have a hard coded labels dict
for CK. 

"""
import numpy as np


def ckTrialGroup(sp, eventTimes, n_tgs: int, dict_number=0):
    trialGroup_raw = eventTimes["DIG6"]["TrialGroup"]

    reps = int(len(trialGroup_raw) / n_tgs)  # replicates of trial group

    my_list = list()

    """we iterate over the number of trial groups
    and multiply the value by the number of replicates of
    that trial group
    """

    for value in range(n_tgs):
        my_list += reps * [value]

    trial_group = np.array(
        my_list
    )  # having this as a numpy array helps for future indexing

    eventTimes["DIG6"]["TrialGroup"] = trial_group

    """This section of the code is for creating a labels dict. Add elif
    if you want a separate dict to remember"""
    if dict_number == 0:
        my_dict = {
            "0": "0 PSI_B",
            "1": "0 PSI_M",
            "2": "0 PSI_N",
            "3": "0.15 PSI_B",
            "4": "0.15 PSI_M",
            "5": "0.15 PSI_N",
            "6": "0.3 PSI_B",
            "7": "0.3 PSI_M",
            "8": "0.3 PSI_N",
            "9": "0.6 PSI_B",
            "10": "0.6 PSI_M",
            "11": "0.6 PSI_N",
            "12": "0.75 PSI_B",
            "13": "0.75 PSI_M",
            "14": "0.75 PSI_N",
            "15": "0.9 PSI_B",
            "16": "0.9 PSI_M",
            "17": "0.9 PSI_N",
            "18": "1.45 PSI_B",
            "19": "1.45 PSI_M",
            "20": "1.45 PSI_N",
        }

    np.save(sp["filename"] + "eventTimes.npy", eventTimes)

    return eventTimes, my_dict
