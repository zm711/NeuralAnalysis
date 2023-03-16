# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:04:24 2023

@author: ZacharyMcKenzie
"""
from neuralanalysis.ClusterAnalysis import ClusterAnalysis
from neuralanalysis.SPAnalysis import SPAnalysis
from neuralanalysis.MergedCA import MCA
from test.test_clusterAnalysis import gen_data


def test_plot_waveforms(mocker):
    mocker.patch.object(ClusterAnalysis, "plot_wfs", return_value=1)

    sp, eventTimes = gen_data(1234567890)
    my_neuron = ClusterAnalysis(sp, eventTimes)
    assert my_neuron.plot_wfs(ind=True) == 1
    mocker.patch.object(SPAnalysis, "plot_wfs", return_value=2)
    spikes = SPAnalysis()
    spikes.sp = sp
    assert spikes.plot_wfs() == 2


def test_plot_psth_viewer(mocker):
    mocker.patch.object(ClusterAnalysis, "plot_spikes", return_value=1)

    sp, eventTimes = gen_data(1234567890)
    my_neuron = ClusterAnalysis(sp, eventTimes)
    assert my_neuron.plot_spikes() == 1


def test_acg(mocker):
    mocker.patch.object(ClusterAnalysis, "acg", return_value=1)

    sp, eventTimes = gen_data(1234567890)
    my_neuron = ClusterAnalysis(sp, eventTimes)
    assert my_neuron.acg() == 1


def test_plot_fr(mocker):
    mocker.patch.object(ClusterAnalysis, "plot_firingrate", return_value=1)

    sp, eventTimes = gen_data(1234567890)
    my_neuron = ClusterAnalysis(sp, eventTimes)
    assert my_neuron.plot_firingrate() == 1


def test_plot_CDFs(mocker):
    mocker.patch.object(ClusterAnalysis, "plot_cdf", return_value=1)

    sp, eventTimes = gen_data(1234567890)
    my_neuron = ClusterAnalysis(sp, eventTimes)
    assert my_neuron.plot_cdf() == 1


def test_plot_pc(mocker):
    mocker.patch.object(ClusterAnalysis, "plot_pc", return_value=1)

    sp, eventTimes = gen_data(1234567890)
    my_neuron = ClusterAnalysis(sp, eventTimes)
    assert my_neuron.plot_pc() == 1


def test_plot_drift(mocker):
    mocker.patch.object(ClusterAnalysis, "plot_drift", return_value=1)

    sp, eventTimes = gen_data(1234567890)
    my_neuron = ClusterAnalysis(sp, eventTimes)
    assert my_neuron.plot_drift() == 1


def test_MCA_api_call(mocker):
    mocker.patch.object(MCA, "plot_z", return_value=3)

    sp, eventTimes = gen_data(1234567890)
    neuron = ClusterAnalysis(sp, eventTimes)
    neurons = MCA(neuron)
    assert neurons.plot_z(plot=True) == 3
