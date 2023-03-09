# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:33:21 2023

@author: ZacharyMcKenzie
"""

from neuralanalysis.analysis import histdiff
import numpy as np
from numpy.random import RandomState
import numpy.testing
from neuralanalysis.analysis import psthfunctions as psfn


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
