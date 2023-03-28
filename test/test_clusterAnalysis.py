# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:18:27 2023

@author: ZacharyMcKenzie
"""

import numpy as np
import numpy.testing
from neuralanalysis import ClusterAnalysis
from numpy.random import RandomState


def gen_data(value):
    sp = dict()
    seq = RandomState(value)
    spike_times = np.array(sorted(list(seq.randint(0, 10000, size=10000) * 0.6543)))
    cluster_ids = np.array(seq.randint(0, 10, size=len(spike_times)))
    cids = np.array(list(set(cluster_ids)))
    sp["spikeTimes"] = spike_times
    sp["clu"] = cluster_ids
    sp["cids"] = cids
    sp["filename"] = "test"

    events = np.array(sorted(list(seq.randint(0, 10000, size=10) * 0.2984)))
    lengths = 10 * np.ones(
        10,
    )
    trial_groups = np.array(sorted(list(seq.randint(0, 5, size=10))))
    eventTimes = dict()
    eventTimes["DIG1"] = dict()

    eventTimes["DIG1"]["EventTime"] = events
    eventTimes["DIG1"]["Lengths"] = lengths
    eventTimes["DIG1"]["TrialGroup"] = trial_groups
    eventTimes["DIG1"]["Stim"] = "Test"

    return sp, eventTimes


def test_clu_z_score():
    sp, eventTimes = gen_data(1234567890)
    myNeuron = ClusterAnalysis.ClusterAnalysis(sp, eventTimes)

    allP, normVal, window = myNeuron.clu_zscore(
        time_bin_size=0.05, window=[[-1, -0.1], [-1, 5]]
    )

    assert window == [[-1, 5]]
    allP = allP["Test"]

    assert np.shape(allP) == (10, 4, 120)
    assert np.isclose(np.sum(allP), 278.54138484493467)
    allP_no_trial = np.mean(allP, axis=1)

    assert np.shape(allP_no_trial) == (10, 120)
    assert np.isclose(np.sum(allP_no_trial), 69.6353462112337)


def test_responsive_neurons_generation_tg_false():
    sp, eventTimes = gen_data(1234567890)
    myNeuron = ClusterAnalysis.ClusterAnalysis(sp, eventTimes)

    allP, _, _ = myNeuron.clu_zscore(
        time_bin_size=0.05, tg=False, window=[[-1, -0.1], [-1, 5]]
    )

    myNeuron.plot_z(labels=None, tg=False, plot=False)

    responsive_neurons = myNeuron.responsive_neurons
    assert type(responsive_neurons) == dict

    raw_responsive_neurons = myNeuron.raw_responsive_neurons
    assert type(raw_responsive_neurons) == dict


def test_spike_raster():
    sp, eventTimes = gen_data(1234567890)
    myNeuron = ClusterAnalysis.ClusterAnalysis(sp, eventTimes)
    myNeuron.spike_raster(time_bin_size=0.001, window_list=[[-1.0, 5.0]])
    psthvalues = myNeuron.psthvalues
    window = myNeuron.raster_window
    assert window == [[-1.0, 5.0]]

    psthvalues_ba = psthvalues["Test"]["0"]["BinnedArray"]
    assert np.shape(psthvalues_ba) == (10, 6000)
    assert np.sum(psthvalues_ba) == 6.0
    numpy.testing.assert_allclose(
        np.mean(psthvalues_ba, axis=1),
        np.array(
            [
                0.000166667,
                0,
                0.000166667,
                0,
                0.000166667,
                0.000166667,
                0,
                0.000333333,
                0,
                0,
            ]
        ),
        rtol=1e-04,
    )

    bincenters = psthvalues["Test"]["0"]["Bins"]
    assert np.shape(bincenters) == (6000,)
    assert np.isclose(bincenters[2], -0.9975)


def test_firing_window():
    sp, eventTimes = gen_data(1234567890)
    myNeuron = ClusterAnalysis.ClusterAnalysis(sp, eventTimes)
    myNeuron.firingratedf(window_dict={"Onset": [0, 2]})

    assert myNeuron.firing_rate_df.any().any()
    assert np.shape(myNeuron.firing_rate_df) == (40, 6)
    assert myNeuron.firing_rate_df["Spikes/sec"].max() == 0.75
