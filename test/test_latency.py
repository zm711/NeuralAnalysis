# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:33:38 2023

@author: ZacharyMcKenzie
"""

import numpy as np
from neuralanalysis.analysis import latency_calculator
from neuralanalysis.analysis import psthfunctions
from test.test_clusterAnalysis import gen_data


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
