# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 08:59:50 2023

@author: ZacharyMcKenzie
"""

from test.test_clusterAnalysis import gen_data
import numpy as np

from neuralanalysis import MergedCA
from neuralanalysis import ClusterAnalysis


def test_MCA():
    sp, eventTimes = gen_data(1234567890)
    sp1, eventTimes2 = gen_data(1234567894)
    neuron1 = ClusterAnalysis.ClusterAnalysis(sp, eventTimes)

    eventTimes2["DIG1"]["TrialGroup"][:3] = 3
    neuron2 = ClusterAnalysis.ClusterAnalysis(sp1, eventTimes2)

    labels = {"Test": {"1.0": "1.0", "2.0": "2.0", "3.0": "3.0", "4.0": "4.0"}}
    labels2 = {"Test": {"1.0": "1.0", "3.0": "3.0"}}
    neuron1.labels = labels
    neuron2.labels = labels2
    all_neurons = MergedCA.MCA(neuron1, neuron2)

    assert all_neurons

    all_neurons.m_zscore(
        window_list=[[-1, -0.1], [-1, 5]], time_bin_list=[0.05], tg=True
    )

    allP = all_neurons.allP["Test"]

    assert np.shape(allP) == (20, 4, 120)

    assert np.isnan(np.sum(allP))
    assert np.isnan(np.mean(allP))
    assert np.isclose(np.nansum(allP), 201.51012386743596)
    assert np.isclose(np.nanmean(allP), 0.02798751720381055)
