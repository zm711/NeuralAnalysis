# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:55:44 2023

@author: ZacharyMcKenzie
"""

from neuralanalysis.ClusterAnalysis import ClusterAnalysis
from test.test_clusterAnalysis import gen_data


def test_saving(mocker):
    mocker.patch.object(ClusterAnalysis, "save_analysis", return_value=4)

    sp, eventTimes = gen_data(1234567890)
    neuron = ClusterAnalysis(sp, eventTimes)
    assert neuron.save_analysis(title="") == 4


def test_get_files(mocker):
    mocker.patch.object(ClusterAnalysis, "get_files", return_value=4)

    sp, eventTimes = gen_data(1234567890)
    neuron = ClusterAnalysis(sp, eventTimes)
    assert neuron.get_files(title="") == 4


def test_gen_res(mocker):
    mocker.patch.object(ClusterAnalysis, "gen_resp", return_value=5)

    sp, eventTimes = gen_data(1234567890)
    neuron = ClusterAnalysis(sp, eventTimes)
    assert neuron.gen_resp() == 5


def test_gen_resdf(mocker):
    mocker.patch.object(ClusterAnalysis, "gen_respdf", return_value=5)

    sp, eventTimes = gen_data(1234567890)
    neuron = ClusterAnalysis(sp, eventTimes)
    assert neuron.gen_respdf() == 5
