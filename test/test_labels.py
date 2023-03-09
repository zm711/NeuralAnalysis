# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:20:05 2023

@author: ZacharyMcKenzie
"""

from neuralanalysis.misc_helpers import label_generator
import numpy as np


def test_labels():
    eventTimes = {"ADC1tot": {"TrialGroup": np.array([13.0])}}
    labels = label_generator.labelGenerator(eventTimes)
    assert labels["13.0"] == "65.0 mmHg"
