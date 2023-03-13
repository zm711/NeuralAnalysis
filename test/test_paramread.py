# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 18:25:22 2023

@author: ZacharyMcKenzie
"""

import numpy as np
import numpy.testing
from neuralanalysis.intan_helpers import stimulushelpers as sh

from neuralanalysis.analysis import spsetup as spset


def test_paramread():
    """test for obtaining sample rate from the params.py file"""
    frequency = sh.paramread()

    assert frequency == 30_000


def test_read_cgs():
    """test for reading cgs files for setting up sp"""
    cgsfile = ["cluster_id	group", "0    mua", "1    good", "2    noise"]

    cids, cgs = spset.readCGSfile(cgsfile)

    assert len(cids) == len(cgs)
    numpy.testing.assert_allclose(np.array(cids), np.array([0, 1, 2]))
    numpy.testing.assert_allclose(np.array(cgs), np.array([1, 2, 0]))
