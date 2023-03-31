#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 13:00:30 2022

@author: zacharymckenzie

Adapted from Nick Steinmetz's matlab code which is based on 
Dan Hill's work. Original paper was Dan Hill (J.Neuro 2012), but
Nick's code is based on the equation from UltraMegaSort. See Nick's code
in the sortingQuality repo on Github
inputs: sp--to get spikes_times, cids, and clu. For a cluster to be analyzed there
        must be at least 2 spikes to assess if the refractory period is being
        violated. 
        
OUTPUTS: the dict isiV which has three keys for each cluster: 
        nSpikes (spikes which occurred for a cluster
        nViol: number of violations for that cluster
        nViol%: nViol/nSpikes * 100
        fp: false positive rate, which is a statistical assessment of spike
        violations based on Dan Hill's work
         
"""

import numpy as np
from ..misc_helpers.genhelpers import savefile


def isi_viol(sp: dict, isi=0.0005, ref_dur=0.0015) -> dict:
    spike_times = np.squeeze(sp["spikeTimes"])
    filename = sp["filename"]

    clu = np.squeeze(sp["clu"])  # need clusters to reference spikes from spikeTimes

    clusterIDs = list(sp["cids"])  # analyze our current cids

    isiV = {}

    for cluster in clusterIDs:
        print("Cluster Number         FP Rate           Contamination Percent")

        if (
            len(spike_times[clu == cluster]) < 2
        ):  # need to check if there are at least 2 spikes for that cluster
            continue
        else:
            fp_rate, nViolations = isiVCore(spike_times[clu == cluster], isi, ref_dur)

            isiV[str(cluster)] = {}
            isiV[str(cluster)]["fp"] = fp_rate

            nSpikes = len(spike_times[clu == cluster])
            isiV[str(cluster)]["nSpikes"] = nSpikes
            isiV[str(cluster)]["nViol"] = nViolations
            isiV[str(cluster)]["nViol"] = nViolations / nSpikes

            print(
                "      {}                 {:.2f}               {:.3f}%       ".format(
                    cluster, fp_rate, nViolations / nSpikes * 100
                )
            )
    savefile(filename + "isiv.npy", isiV)
    return isiV


def isiVCore(spikes: np.array, isi: float, ref_dur: float) -> tuple[float, int]:
    total_rate = float(len(spikes) / (spikes[-1] - spikes[0]))
    num_violations = float(len(np.where(np.diff(spikes) <= ref_dur)[0]))

    violation_time = 2 * len(spikes) * (ref_dur - isi)
    violation_rate = num_violations / violation_time
    fp_rate = violation_rate / total_rate

    if fp_rate > 1:
        fp_rate = np.nan

    return fp_rate, num_violations
