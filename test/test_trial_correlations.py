# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:05:26 2023

@author: ZacharyMcKenzie
"""

from neuralanalysis.analysis.trial_correlation import trial_corr
from test.test_clusterAnalysis import gen_data
import numpy as np

from numpy.random import RandomState


def test_trial_corr():
    seq = RandomState(1234567890)
    binned_array = seq.randint(0, 10, size=10000).reshape(10, -1)
    psthvalues = {"DIG1": {1: {"BinnedArray": binned_array, "Bins": [0, 0.05]}}}
    eventTimes = {"DIG1": {"TrialGroup": [1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1]}}

    corr_df, mean, std = trial_corr(psthvalues, eventTimes, 50)

    assert np.isclose(mean, 0.8744665056678191), "Correlation should be positive"
    assert np.isnan(std)
    assert np.shape(corr_df) == (1, 4)
    assert corr_df["Stim"][0] == "DIG1"


def test_trial_corr_neg_corr():
    array = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 0, 0],
            [
                1,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
        ]
    )

    psthvalues = {"DIG1": {1: {"BinnedArray": array, "Bins": [0, 0.05]}}}
    eventTimes = {"DIG1": {"TrialGroup": [1.0, 1.0, 1.0]}}

    corr_df, mean, std = trial_corr(psthvalues, eventTimes, 1)

    assert np.isclose(mean, -0.6066944001430321), "Correlation should be negative"
    assert np.isnan(std)
