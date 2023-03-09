# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:47:43 2023

@author: ZacharyMcKenzie
"""

import numpy as np


from neuralanalysis.intan_helpers import stimulushelpers


import numpy.testing


def test_calculate_binary():
    zeros = np.zeros((10,))
    ones = np.ones((10,))
    final_array = np.concatenate(
        (zeros, zeros, ones, ones, zeros, zeros, ones, ones, zeros, zeros)
    )

    sample_rate = 30_000.0

    length, time = stimulushelpers.calculate_binary(final_array, sample_rate)

    assert len(time) == len(length)
    assert len(time) == 2
    np.testing.assert_allclose(time, np.array([0.000633333, 0.00196667]), rtol=1e-05)
