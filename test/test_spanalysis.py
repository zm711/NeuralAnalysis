# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:11:09 2023

@author: ZacharyMcKenzie
"""


from test.test_clusterAnalysis import gen_data

from neuralanalysis import SPAnalysis


def test_spa():
    sp, _ = gen_data(123456789)
    spikes = SPAnalysis.SPAnalysis()

    assert spikes

    spikes.sp = sp
    assert spikes.sp

    assert spikes.get_waveforms
    assert spikes.plot_drift
    assert spikes.plot_cdf
