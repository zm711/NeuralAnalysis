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
        self.filepath = filepath
        self.filename = filename

    def make_eventTimes(
        self, stim_array: Optional[np.array], stimulus_metrics: Optional[dict]
    ) -> dict:
        eventTimes = gen_events_non_intan(self.filepath, stim_array, stimulus_metrics)
        self.eventTimes = eventTimes

    def set_trial_groups(self, trial_groups: dict) -> dict:
        eventTimes = set_trial_groups(self.eventTimes, trial_group=trial_groups)
        self.eventTimes = eventTimes

    def set_stim_names(self, stim_name: dict) -> dict:
        eventTimes = set_stim_name(self.eventTimes, stim_names=stim_name)
        self.eventTimes = eventTimes

    def save_eventTimes(self, name: Optional[str]):
        save_event_times(name=name, eventTimes=self.eventTimes)
