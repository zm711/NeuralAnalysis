# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:47:43 2023

@author: ZacharyMcKenzie
"""

from neuralanalysis.intan_helpers import stimulushelpers
from neuralanalysis.intan_helpers import stimulus_setupzm as stim
import numpy as np
import numpy.testing


def test_calculate_binary():
    zeros = np.zeros((10,))
    ones = np.ones((10,))
    final_array = np.concatenate(
        (zeros, zeros, ones, ones, zeros, zeros, ones, ones, zeros, zeros)
    )

    sample_rate = 30_000.0

    length, time = stimulushelpers.calculate_binary(final_array, sample_rate)

    assert len(time) == len(length)
    assert len(time) == 2
    np.testing.assert_allclose(time, np.array([0.000633333, 0.00196667]), rtol=1e-05)


def test_spike_prep():
    zeros = np.zeros((10,))
    ones = np.ones((10,))
    final_array = np.concatenate(
        (zeros, zeros, ones, ones, zeros, zeros, ones, ones, zeros, zeros)
    )

    sample_rate = 30_000.0

    length, time, trial = stimulushelpers.spike_prep(
        final_array, sample_rate, trial_group=0
    )

    assert len(length) == len(time)
    assert trial == 0
    np.testing.assert_allclose(time, np.array([0.000633333, 0.00196667]), rtol=1e-05)


def test_dig_stim_setup():
    zeros = np.zeros((10,))
    ones = np.ones((10,))
    final_array = np.concatenate(
        (zeros, zeros, ones, ones, zeros, zeros, ones, ones, zeros, zeros)
    )
    dig_in = np.expand_dims(final_array, axis=0)

    sample_rate = 30_000.0
    channel_dict = {}
    channel_dict[0] = {}
    channel_dict[0]["native_channel_name"] = [7]

    eventTimes = {}

    eventTimes = stim.dig_stim_setup(dig_in, channel_dict, sample_rate, eventTimes)

    assert list(eventTimes.keys()) == ["DIG7"]
    assert list(eventTimes["DIG7"].keys())[0] == "EventTime"
    assert len(eventTimes["DIG7"]["TrialGroup"]) == 2


def test_baro():
    zeros = np.zeros((10,))
    peak = np.array([0.7, 1.5, 3.2, 4.6, 7.8, 8.2, 9.1, 10.3, 12.1])
    top = np.array([12.8, 13.1, 12.9, 13.2, 12.7, 13.1, 13.0, 13.1, 13.2, 13.1])
    trough = np.flip(peak)
    adc = np.concatenate(
        (
            zeros,
            zeros,
            peak,
            top,
            top,
            top,
            top,
            top,
            top,
            top,
            top,
            top,
            top,
            trough,
            zeros,
            zeros,
            peak,
            top,
            top,
            top,
            top,
            top,
            top,
            top,
            top,
            trough,
            zeros,
            zeros,
        )
    )
    sample_rate = 5
    baro, total = stim.barostat_stim_setup(adc, sample_rate, False)

    assert baro is None

    assert np.shape(total) == (3, 2)

    numpy.testing.assert_allclose(total[0, :], [3.8, 31.4])

    assert np.isclose(total[1, 1], 19.6)

    assert total[2, 1] == 52
    assert total[2, 0] == 52
