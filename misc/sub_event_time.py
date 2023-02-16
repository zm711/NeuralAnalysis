# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:32:44 2023

@author: ZacharyMcKenzie
"""

import numpy as np


def sub_event_time(
    eventTimes: dict, stim_key: str, labels: dict, sub_list: list
) -> tuple[dict, dict]:

    trial_group = eventTimes[stim_key]["TrialGroup"]
    event_lengths = eventTimes[stim_key]["Lengths"]
    event_onsets = eventTimes[stim_key]["EventTime"]

    sub_keys = np.array([float(item) for item in sub_list])

    event_onsets = event_onsets[np.isin(trial_group, sub_keys)]
    event_lengths = event_lengths[np.isin(trial_group, sub_keys)]
    trial_group = trial_group[np.isin(trial_group, sub_keys)]

    sub_keys = np.array([str(item) for item in sub_keys])
    new_labels = dict()
    for element in sub_keys:
        new_labels[element] = labels[element]

    return eventTimes, new_labels
