# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:12:02 2023

@author: ZacharyMcKenzie

Still working on 
"""

import numpy as np
import numpy.testing
from neuralanalysis.visualization_ca import detectdrift, plotCDFs
from numpy.random import RandomState


def test_detect_drift():
    seq = RandomState(1234567890)
    timestamps = np.array(
        sorted(list(set(seq.randint(0, 10000000, size=1000000) * 0.153443)))
    )

    seq1 = RandomState(1234567891)
    depths = np.array(seq1.randint(140, 1500, size=len(timestamps)))

    events = detectdrift.detect_drift_events(timestamps, depths)

    assert np.shape(events) == (0, 3)
    assert np.sum(events) == 0


def test_ADbins_wfamps():
    seq = RandomState(1234567890)

    probe_len = 775.0
    pitch = 50.0
    spike_times = np.array(
        sorted(list(set(seq.randint(0, 1000000, size=1000000) * 0.98)))
    )
    depth = None
    amps = np.array(sorted(list(seq.randint(0, 250, size=len(spike_times)) * 0.99)))
    depths = np.array(sorted(list(seq.randint(0, 1000, size=len(spike_times)) * 0.98)))

    depth_bins, amp_bins, recording_dur = plotCDFs.genADBins(
        amps, probe_len, pitch, spike_times, depth
    )

    assert len(depth_bins) == 15
    assert len(amp_bins) == 8
    assert np.isclose(recording_dur, 979998.04)

    numpy.testing.assert_allclose(
        depth_bins[:5], np.array([0.00, 55.3571, 110.714, 166.071, 221.429]), rtol=1e-05
    )

    numpy.testing.assert_allclose(
        amp_bins[:3], np.array([0.000, 35.2157, 70.4314]), rtol=1e-05
    )

    pdfs, cdfs = plotCDFs.computeWFamps(
        amps,
        depths,
        amp_bins,
        depth_bins,
        recording_dur,
    )

    assert np.shape(pdfs) == np.shape(cdfs)
    assert np.isclose(cdfs[13, 0], 0.0358582)
    assert np.isclose(pdfs[8, 3], 0.0370602)
