# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 13:11:09 2023

@author: ZacharyMcKenzie
"""

import numpy as np
from neuralanalysis.analysis import latency_calculator
from neuralanalysis.analysis import psthfunctions
from test.test_clusterAnalysis import gen_data

from neuralanalysis import SPAnalysis


def test_spa():
    spikes = SPAnalysis.SPAnalysis()

    assert spikes
