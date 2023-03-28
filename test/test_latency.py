# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:33:38 2023

@author: ZacharyMcKenzie
"""

from test.test_clusterAnalysis import gen_data

import numpy as np
from neuralanalysis.analysis import latency_calculator, psthfunctions


def test_latency():
    sp, eventTimes = gen_data(1234567890)

    spike_times = sp["spikeTimes"]
    clu = sp["clu"]
    events = eventTimes["DIG1"]["EventTime"]

    values = psthfunctions.psthAndBA(
        spike_times[clu == 0], event_times=events, window=[0, 10], psthBinSize=0.05
    )

    binned_array = values[-1]
    assert np.shape(binned_array) == (10, 200)

    mean, std = latency_calculator.latency_core(0.001, binned_array, 0.05)

    assert np.isclose(mean, 9.950000000000001)
    assert np.isclose(std, 0.0)

    med, std_med = latency_calculator.latency_median(binned_array, 0.05)

    assert np.isnan(med)
    assert np.isnan(std_med)


def test_latency_med():
    binned_array = np.array(
        [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]
    )

    med, _ = latency_calculator.latency_median(binned_array, 0.05)

    assert med == 0.1


def test_latency_mean():
    binned_array = np.array(
        [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]
    )

    mean, _ = latency_calculator.latency_core(1, binned_array, 0.05)

    assert mean == 0.2


def test_latency_calculator_ca():
    sp, eventTimes = gen_data(1234567890)
    bsl_win = [[-2, -1]]
    event_win = [[0, 2]]
    num_shuf = 10

    latency, shuffled = latency_calculator.latency_calculator(
        sp, eventTimes, 0.05, bsl_win=bsl_win, event_win=event_win, num_shuffle=num_shuf
    )

    assert type(latency) == dict
    assert len(latency["cluster_ids"]) == 10
    assert np.isclose(np.nanmean(latency["DIG1"]), 0.07777777777777778)

    assert type(shuffled) == dict
    assert np.shape(shuffled["DIG1"]) == (10, 4, 10)
