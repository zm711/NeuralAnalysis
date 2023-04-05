# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:20:05 2023

@author: ZacharyMcKenzie
"""

import numpy as np
import numpy.testing
import pandas as pd
from neuralanalysis.misc_helpers import label_generator


def test_labels():
    eventTimes = {"ADC1tot": {"TrialGroup": np.array([13.0])}}
    labels = label_generator.labelGenerator(eventTimes)
    assert "Barostat" in labels.keys()
    assert labels["Barostat"]["13.0"] == "65.0 mmHg"


def test_qc_only():
    sp = dict()
    sp["filename"] = "test"
    sp["cids"] = np.array([0, 1, 2])
    sp["noise"] = np.array([False, False, True])
    qc = dict()
    qc["uQ"] = np.array([75.0, 10.0, 0.2])
    qc["sil"] = np.array([0.9, 0.8, 0.8])
    qcthres = 15.0
    sil = 0.75
    isiv = {"0": {"nViol": 0.002}, "1": {"nViol": 0.001}}
    isi = 0.02

    sp, quality_df = label_generator.qc_only(qc, isiv, sp, qcthres, sil, isi)

    assert np.shape(sp["cids"]) == (1,)
    assert sp["cids"] == 0

    assert np.shape(quality_df) == (1, 6)
    assert quality_df["QC"][0] == 75.0
    assert quality_df["Silhouette Score"][0] == 0.9
    assert quality_df["ISI Violation Fraction"][0] == 0.002

    assert (
        quality_df["HashID"][0]
        == "37a45319b07f1d11b65476f871f4d47ecdc719223decdb1c20a7b7b4ea03c51f"
    )


def test_waveform_vals():
    wf = dict()
    wf["F"] = dict()
    wf["F"]["ClusterIDs"] = np.array([0, 1, 2])
    sp = {"filename": "test"}
    waveform_dur = np.array([1.2, 1.3, 1.4])
    waveform_depth = np.array([500, 600, 700])
    waveform_amps = np.array([254, 244, 100])
    shank_dict = None

    neuron_char = label_generator.waveform_vals_DF(
        wf, sp, waveform_dur, waveform_depth, waveform_amps, shank_dict
    )
    assert np.shape(neuron_char) == (3, 5)

    assert (
        neuron_char["HashID"][2]
        == "ead312b5d9795fee67deb9b6251732cffab8f6daa93edb10805fe0bbfb620371"
    )
    assert neuron_char["Waveform Amplitude (uV)"][2] == 100
    assert neuron_char["Waveform Duration (s)"][1] == 1.3
    assert neuron_char["Waveform Depth (um)"][0] == 500


def test_gen_resp():
    resp_df = pd.DataFrame({"IDs": [0, 1, 2, 3]})
    sp = {"cids": np.array([0, 1, 2, 3, 4, 5, 6])}

    sp = label_generator.gen_resp(resp_df, sp)

    assert len(sp["cids"]) == 4
    numpy.testing.assert_allclose(sp["cids"], np.array([0, 1, 2, 3]))
