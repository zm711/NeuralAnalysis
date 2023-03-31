# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:10:44 2023

@author: ZacharyMcKenzie
"""

import os
import glob
import numpy as np
from ..intan_helpers.stimulushelpers import calculate_binary, paramread
from ..misc_helpers.genhelpers import getdir, savefile
from typing import Optional


def gen_events_non_intan(
    filepath: Optional[None], stim_array: Optional[np.array], **kwargs
) -> dict:
    """this function will generate eventTimes (stimulus dict) based on a ndarray. Input
    is an optional filepath `filepath`. If not given it will provide a file selection
    gui. It can also input an nxm ndarray (type int, but it will autoconvert type bool)
    where `n` is number of distinct inputs and m are the number of samples of that input.
    Finally kwargs should be entered as a dict with up to two keys or separate as
    keyword arguments with `stimulus` refering to the channel number for each row in
    stim_array and trial being a list of ndarrays  indicating the trial groups for each
    stimulus event. np.ones((n_events,)) is fine if no distinct stimulus parameters were
    used."""

    if filepath is not None:
        os.chdir(filepath)
    else:
        print(
            "Please click on folder with the Phy parameter folder to load sample_rate"
        )
        (
            _,
            _,
            _,
        ) = getdir()

    sample_rate = paramread()

    if stim_array is None:
        print("Please select folder with file stim.npy file with stimulus array")
        _, _, _ = getdir()
        stim_array = np.load(glob.glob("*stim.npy")[0])

    stim_array = np.array(stim_array, dtype=np.int_)
    if len(stim_array.shape) == 1:
        stim_array = np.expand_dims(stim_array, axis=0)
    stim_labels = None
    trial_group = None
    for key in kwargs.keys():
        if key == "stimulus":
            stim_labels = np.array(kwargs[key])
        if key == "trial":
            trial_group = np.array(kwargs[key])

    eventTimes = dict()

    if stim_labels is None:
        stim_labels = list(range(np.shape(stim_array)[0]))

    eventTimes = stim_align_non_intan(
        stim_array, sample_rate, stim_labels, trial_group, eventTimes
    )
    return eventTimes


def stim_align_non_intan(
    stim_array: np.array,
    sample_rate: int,
    stim_labels: Optional[list],
    trial_group: Optional[np.array],
    eventTimes: dict,
) -> dict:
    for row in range(np.shape(stim_array)[0]):
        sub_array = stim_array[row]
        dict_key = stim_labels[row]
        eventTimes[dict_key] = {}
        event_len, events = calculate_binary(sub_array, sample_rate=sample_rate)
        eventTimes[dict_key]["EventTimes"] = events
        eventTimes[dict_key]["Lengths"] = event_len
        if trial_group is not None:
            eventTimes[dict_key]["TrialGroup"] = np.array(trial_group)

    return eventTimes


def set_trial_groups(eventTimes: dict, trial_group: dict) -> dict:
    for stim, group in trial_group.items():
        eventTimes[stim]["TrialGroup"] = np.array(group)
    return eventTimes


def set_stim_name(eventTimes: dict, stim_names: dict) -> dict:
    for stim, name in stim_names.items():
        eventTimes[stim]["Stim"] = name

    return eventTimes


def save_event_times(name: Optional[str], eventTimes: dict) -> None:
    if name is None:
        name = ""
    savefile(name + "eventTimes.npy", eventTimes)
