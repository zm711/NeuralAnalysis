# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:05:17 2023

@author: ZacharyMcKenzie
"""

import numpy as np
import os

from neuralanalysis.stimulus_helpers import non_intan_functions as nif

from neuralanalysis.stimulus_helpers import EventTimesMaker as etm


def test_gen_events_non_intan():
    stim_array = np.array([0, 1, 1, 1, 1, 0, 0, 0])

    events = nif.gen_events_non_intan(os.getcwd(), stim_array)
    assert list(events.keys())[0] == 0, "Key generation error"

    assert np.isclose(events[0]["Lengths"][0], 0.000133333), "length calculated wrong"


def test_gen_events_non_intan_kwarg():
    stim_array = np.array([0, 1, 1, 1, 1, 0, 0, 0])

    test_dict = {"stimulus": ["dig1"]}

    events = nif.gen_events_non_intan(os.getcwd(), stim_array, **test_dict)
    assert (
        list(events.keys())[0] == test_dict["stimulus"][0]
    ), "failure of loading stimulus"

    test_dict = {"stimulus": ["dig1"], "trial": np.array([1.0])}

    events = nif.gen_events_non_intan(os.getcwd(), stim_array, **test_dict)
    assert events["dig1"]["TrialGroup"], "failure of loading trial"
    assert events["dig1"]["TrialGroup"][0] == 1.0, "trial group loaded wrong value"


def test_set_stim_name():
    events = {"dig1": {"EventTimes": [1], "Lengths": [1], "Trial Group": [1]}}

    events = nif.set_stim_name(events, stim_names={"dig1": "Test"})

    assert events["dig1"]["Stim"] == "Test"


def test_set_trial_group():
    events = {"dig1": {"EventTimes": [1], "Lengths": [1]}}

    events = nif.set_trial_groups(events, trial_group={"dig1": [1]})

    assert events["dig1"]["TrialGroup"][0] == 1, "setting trial groups failed"


def test_EventTimesMaker():
    event_maker = etm.EventTimesMaker("test", "test2")

    assert event_maker.filename == "test2"
    assert event_maker.filepath == "test"
