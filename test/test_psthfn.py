# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:33:21 2023

@author: ZacharyMcKenzie
"""

from test.test_clusterAnalysis import gen_data

import numpy as np
import numpy.testing
from neuralanalysis.analysis import histdiff
from neuralanalysis.analysis import psthfunctions as psfn
from numpy.random import RandomState


def test_histdiff():
    seq = RandomState(1234567890)
    timestamps = np.array(sorted(list(seq.randint(0, 1000, size=500) * 0.15)))
    bins = np.linspace(0.001, 0.2, num=10)

    cnts, centers = histdiff.histdiff(timestamps, timestamps, bins)

    assert cnts[6] == 238
    assert cnts[0] == 0
    assert len(centers) == 9

    numpy.testing.assert_allclose(
        centers[:4], np.array([0.0120556, 0.0341667, 0.0562778, 0.0783889]), rtol=1e-05
    )


def test_histdiff_acg_simple():
    test_array = np.array([1, 2, 3, 4, 5])
    bins = np.array([0, 1, 2])
    cnts, centers = histdiff.histdiff(test_array, test_array, bins)
    assert len(cnts) == 2
    numpy.testing.assert_allclose(cnts, np.array([5.0, 4.0]))
    numpy.testing.assert_allclose(centers, np.array([0.5, 1.5]))


def test_histdiff_refpt_simple():
    test_array = np.array([1, 2, 3, 4, 5])
    ref_pt = np.array([1, 5])
    bins = np.array([0, 1, 2])

    cnts, centers = histdiff.histdiff(test_array, ref_pt, bins)
    assert len(cnts) == 2
    numpy.testing.assert_allclose(cnts, np.array([2.0, 1.0]))
    numpy.testing.assert_allclose(centers, np.array([0.5, 1.5]))


def test_time_stamps_simple():
    test_array = np.array([1, 2, 3, 4, 5])
    bin_size = 1.0
    start = 0.0
    end = 2.0
    cnts, centers = psfn.time_stamps_to_bins(
        test_array, test_array, bin_size, start, end
    )
    numpy.testing.assert_allclose(centers, np.array([0.5, 1.5]))

    assert np.shape(cnts) == (5, 2)
    numpy.testing.assert_allclose(
        cnts, np.array([[1.0, 2], [1, 2], [1, 2], [1, 1], [1, 0]])
    )


def test_time_stamps_to_bins():
    seq = RandomState(1234567890)
    timestamps = np.array(sorted(list(seq.randint(0, 1000, size=500) * 0.15)))
    bins = np.linspace(0.001, 0.2, num=10)
    ref_pts = np.array(sorted(list(seq.randint(0, 1000, size=3) * 0.15)))
    bin_size = bins[1] - bins[0]

    array, center = psfn.time_stamps_to_bins(timestamps, ref_pts, bin_size, 0.0, 1.0)

    assert np.shape(array) == (3, 45)
    assert type(array) == np.ndarray

    array_summed = np.sum(array, axis=1)
    array_summed_0 = np.sum(array, axis=0)

    assert array_summed_0[13] == 2
    numpy.testing.assert_array_equal(array_summed, np.array([3, 3, 4.0]))

    numpy.testing.assert_allclose(
        center[:4], np.array([0.0111111, 0.0333333, 0.0555556, 0.0777778]), rtol=1e-05
    )


def test_rasterize():
    seq = RandomState(1234567890)
    timestamps = np.array(sorted(list(seq.randint(0, 1000, size=500) * 0.15)))

    xx, yy = psfn.rasterize(timestamps)

    assert type(xx) is np.ndarray
    assert yy[0, 0] == 0
    assert yy[0, 1] == 1
    assert xx[0, 0] == xx[0, 1]
    assert np.isnan(xx[0, 2])


def test_psthAndBA():
    sp, eventTimes = gen_data(1234567890)
    spike_times = sp["spikeTimes"]
    clu = sp["clu"]
    events = eventTimes["DIG1"]["EventTime"]

    psth, bins, raster_x, raster_y, spike_counts, binned_array = psfn.psthAndBA(
        spike_times[clu == 0], event_times=events, window=[0, 10], psthBinSize=0.05
    )

    assert np.shape(psth) == (200,)
    assert psth[26] == 2
    assert psth[0] == 0

    assert bins[0] == 0.025
    assert np.shape(bins) == np.shape(psth)

    assert np.shape(raster_x) == np.shape(raster_y)

    numpy.testing.assert_array_equal(
        spike_counts, np.array([1.0, 0, 1, 0, 0, 1, 1, 2, 0, 0])
    )
    assert np.shape(binned_array) == (10, 200)
