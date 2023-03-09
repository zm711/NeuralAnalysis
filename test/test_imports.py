# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:56:58 2023

@author: ZacharyMcKenzie
"""


def test_import_SPA():
    from neuralanalysis import SPAnalysis

    spikes = SPAnalysis.SPAnalysis()
    assert spikes


def test_import_CA():
    from neuralanalysis import ClusterAnalysis

    assert ClusterAnalysis.ClusterAnalysis


def test_import_MCA():
    from neuralanalysis import MergedCA

    assert MergedCA.MCA


def test_import_all():
    import neuralanalysis.full as na

    spikes = na.SPAnalysis()
    assert spikes
    assert na.loadKS
