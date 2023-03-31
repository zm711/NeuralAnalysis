# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 08:35:58 2023

@author: ZacharyMcKenzie
"""

import numpy as np

from .non_intan_functions import (
    gen_events_non_intan,
    set_trial_groups,
    set_stim_name,
    save_event_times,
)
from typing import Optional


class EventTimesMaker:
    def __init__(self, filepath: Optional[str], filename: Optional[str]):
        """filepath is an optional filepath to the directory in which the stimulus data
        and params.py file live. filename would be the name that the user wants to give
        to the eventTimes.npy file. Once an instance is initialized the make_eventTimes
        can be used to generate the stimlus times, lengths, and optionally other info.
        Otherwise other methods can be used to set the additionally info later"""
        self.filepath = filepath
        self.filename = filename

    def make_eventTimes(
        self, stim_array: Optional[np.array], stimulus_metrics: Optional[dict]
    ) -> dict:
        """Requires input of the nxm stimlus array from recording equipment where n is
        the number of distinct stimulus inputs, eg. dig1, dig2, dig3, and m is the
        signal at each sample, ie True/False or 1/0. stimulus_metrics is an optional
        dict which should have a `stimulus` key to label stimuli (ie dig1, DIG2) and a
        `trial` key containing a list of np.arrays with trial group labels for each
        event. Thus if you did 10 stimuli the len(list)=n and each ndarray within the
        list should have ten numbers. If not trial groupings were used, ie all the same
        stimulus intensity, it is better to use np.ones((n_events,)) (where n_events=10
        for our example) and load that."""

        eventTimes = gen_events_non_intan(self.filepath, stim_array, stimulus_metrics)
        self.eventTimes = eventTimes

    def set_trial_groups(self, trial_groups: dict) -> dict:
        """to set trial groups later enter a dict with keys given by eventTimes.keys()
        and fill in one ndarray of len(n_events), for example {'dig1': np.ones((10,)),
        'dig2': np.array([1,2,1,2,1,2,1,2])}"""
        eventTimes = set_trial_groups(self.eventTimes, trial_group=trial_groups)
        self.eventTimes = eventTimes

    def set_stim_names(self, stim_name: dict) -> dict:
        """to set stimulus names enter a dict with keys given by eventTimes.keys() and
        fill in with a string name for each stimulus for example {'dig1': 'laser'}"""
        eventTimes = set_stim_name(self.eventTimes, stim_names=stim_name)
        self.eventTimes = eventTimes

    def save_eventTimes(self, name: Optional[str]):
        """If name is given this will override the initialized name otherwise it will
        preprend the self.filename onto eventTimes.npy"""
        if name is None:
            name = self.filename
        save_event_times(name=name, eventTimes=self.eventTimes)
