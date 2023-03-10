# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 18:25:22 2023

@author: ZacharyMcKenzie
"""

import numpy as np
from neuralanalysis.intan_helpers import stimulushelpers as sh


def test_paramread():
    frequency = sh.paramread()

    assert frequency == 30_000
