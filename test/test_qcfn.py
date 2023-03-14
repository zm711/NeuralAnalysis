# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:28:52 2023

@author: ZacharyMcKenzie
"""

import numpy as np


from neuralanalysis.qcfn import isiVqc
from neuralanalysis.qcfn import qcfns

from numpy.random import RandomState
import numpy.testing


def test_isisv():
    seq = RandomState(1234567890)
    timestamps = np.array(sorted(list(seq.randint(0, 1000, size=500) * 0.15)))

    isi = 0.0015
    ref_dur = 0.002

    fp, num_viol = isiVqc.isiVCore(timestamps, isi, ref_dur)
    num_viol = int(num_viol)
    assert np.isnan(fp)
    assert num_viol == 91

    ref_dur_long = 0.1
    fp, num_viol = isiVqc.isiVCore(timestamps, isi, ref_dur_long)

    assert np.isclose(fp, 0.27660304568527916)


def test_count_unique():
    seq = RandomState(1234567890)
    timestamps = np.array(seq.randint(0, 1000, size=500))
    values, instances = qcfns.count_unique(timestamps)

    assert len(values) == 409
    assert len(values) == len(instances)

    assert values[45] == 129
    assert instances[45] == 1


def test_count_unique_simple():
    test_array = np.array([1, 1, 2, 3, 4, 5, 5, 5])
    values, instance = qcfns.count_unique(test_array)

    assert len(values) == 5
    numpy.testing.assert_allclose(np.array(values), np.array([1, 2, 3, 4, 5]))
    numpy.testing.assert_allclose(np.array(instance), np.array([2, 1, 1, 1, 3]))


def test_tipping_point():
    x = np.array([0.0, 1, 3, 4, 5, 9, 10, 16, 17, 18])
    y = np.array(
        [
            1.2,
            1.4,
            1.6,
            7,
            8,
            12,
            13,
            14,
            15,
        ]
    )

    final_pos = qcfns.tipping_point(x, y)
    assert final_pos == 6


def test_tipping_point_simple():
    x = np.array([1, 4, 5])
    y = np.array([2, 3, 6])
    final_pos = qcfns.tipping_point(x, y)
    assert final_pos == 2


def test_tipping_point_simple_low_tip():
    x = np.array([1, 7, 8, 9, 10])
    y = np.array([2, 3, 4, 5, 6, 11])
    pos = qcfns.tipping_point(x, y)
    assert pos == 1


def test_masked_cluster_quality_core():
    seq = RandomState(1234567890)
    this_cluster = np.array(seq.rand(20, 15))
    seq2 = RandomState(1234567891)
    that_cluster = np.array(seq2.rand(30, 15))

    unit_quality, contam_rate = qcfns.masked_cluster_quality_core(
        this_cluster, that_cluster
    )
    # good unit 0 contam, high unit quality
    contam_rate = int(contam_rate)
    assert np.isclose(unit_quality, 74.78633819077363)
    assert contam_rate == 0

    # fail because n< nfeat -> contam is nan
    this_cluster_fail = np.array(seq.rand(10, 15))
    unit_quality, contam_rate = qcfns.masked_cluster_quality_core(
        this_cluster_fail, that_cluster
    )
    assert np.isnan(contam_rate)

    # fail because n > nOther -> contam is nan
    this_cluster_fail2 = np.array(seq.rand(40, 15))
    unit_quality, contam_rate = qcfns.masked_cluster_quality_core(
        this_cluster_fail2, that_cluster
    )
    assert np.isnan(contam_rate)


def test_qc_masked_cluster_quality_sparse():
    clu = np.array([0, 1])
    fet_ind = np.array(([0, 12, 14, 2, 10, 3, 0], [0, 12, 14, 2, 10, 3, 0]))
    fet = np.array(
        (
            [
                [6.03075, 7.51328, 2.29162, 0.637997, -1.4542, 0.443041, 4.31619],
                [
                    1.66104,
                    -0.343693,
                    -1.22469,
                    -0.0662854,
                    0.0165626,
                    -0.905885,
                    3.07569,
                ],
                [3.15145, 2.5234, -2.54112, 0.113551, -1.33154, -2.31246, -0.372592],
            ],
            [
                [11.5723, -1.30386, -0.88831, -4.61045, -2.05544, -0.384842, -3.10064],
                [-2.61024, 2.82361, 2.13182, -2.68165, -0.309417, -0.294074, 0.2321],
                [3.62874, 3.0533, -2.56696, 3.14891, -0.356746, 0.551253, 0.701876],
            ],
        )
    )

    uq, cr = qcfns.masked_cluster_quality_sparse(
        clu,
        fet,
        fet_ind,
    )
    assert len(uq) == 2
    assert len(cr) == 2
    assert uq[0] == 0.0
    assert np.isnan(cr[0])
