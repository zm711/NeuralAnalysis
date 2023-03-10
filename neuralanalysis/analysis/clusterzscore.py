# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 09:59:19 2022

@author: ZacharyMcKenzie
"""
import numpy as np
from . import psthfunctions as psthfn


def clusterzscore(
    sp: dict,
    eventTimes: dict,
    time_bins: list,
    tg=False,
    window_list=None,
) -> tuple[dict, dict, list]:
    allP = {}
    normVal = {}
    spikeTimes = np.squeeze(sp["spikeTimes"])
    clu = np.squeeze(sp["clu"])
    clusterIDs = list(sp["cids"])
    windowlst = list()

    if len(eventTimes.keys()) > 1 and len(time_bins) == 1:
        time_bins *= len(eventTimes.keys())

    for index, stim in enumerate(eventTimes.keys()):
        if len(eventTimes[stim]["EventTime"]) == 0:
            continue
        else:
            if window_list:
                bslWin: list = window_list[0]
                window: list = window_list[1]
            else:
                bslWinIn = input(
                    "Enter baseline window with reference to event onset in format x.y for stim {stim}".format(
                        stim=eventTimes[stim]["Stim"]
                    )
                )
                bslWinStr = bslWinIn.split(",")
                windowIn = input(
                    "Enter stimulus window to be analyzed for each event in format x.y for stimulus {stim}".format(
                        stim=eventTimes[stim]["Stim"]
                    )
                )
                windowStr = windowIn.split(",")
                bslWin = [float(bslWinStr[0]), float(bslWinStr[-1])]
                window = [float(windowStr[0]), float(windowStr[-1])]
            windowlst.append(window)
            allP[eventTimes[stim]["Stim"]] = {}
            normVal[eventTimes[stim]["Stim"]] = {}
            eventTimesOnset: np.array = eventTimes[stim]["EventTime"]
            bslEventTime: np.array = eventTimesOnset

            timeBinSize = time_bins[index]
            if tg == False:
                suballP = np.empty(
                    (len(clusterIDs), int((window[1] - window[0]) / timeBinSize))
                )
                suballP[:] = np.nan
                subnormVal = np.empty((len(clusterIDs), 2))
                subnormVal[:] = np.nan

                for cluster in range(len(clusterIDs)):
                    print("Processing cluster {clu}".format(clu=clusterIDs[cluster]))
                    psthbsl, _, _, _, _, _ = psthfn.psthAndBA(
                        spikeTimes[clu == clusterIDs[cluster]],
                        bslEventTime,
                        bslWin,
                        timeBinSize,
                    )
                    normMn: float = np.mean(psthbsl)  # for z scoring (x-mu)/sigma
                    normStd: float = np.std(psthbsl)  # for z scoring

                    psth, _, _, _, _, _ = psthfn.psthAndBA(
                        spikeTimes[clu == clusterIDs[cluster]],
                        eventTimesOnset,
                        window,
                        timeBinSize,
                    )

                    if normStd != 0:
                        suballP[cluster] = (psth - normMn) / normStd
                        subnormVal[cluster, 0] = normMn
                        subnormVal[cluster, 1] = normStd
                    else:
                        suballP[cluster] = psth
                        subnormVal[cluster, 0] = np.nan
                        subnormVal[cluster, 1] = np.nan

                allP[eventTimes[stim]["Stim"]] = suballP
                normVal[eventTimes[stim]["Stim"]] = subnormVal

            else:
                trialGroups = np.array(eventTimes[stim]["TrialGroup"])
                trialGroupLabel = list(sorted(set(trialGroups)))
                psthbsl = np.empty(
                    (len(trialGroupLabel), int((bslWin[1] - bslWin[0]) / timeBinSize))
                )
                psth = np.empty(
                    (len(trialGroupLabel), int((window[1] - window[0]) / timeBinSize))
                )
                normMn = np.zeros((len(trialGroupLabel), 1))
                normStd = np.zeros((len(trialGroupLabel), 1))

                suballP = np.empty(
                    (
                        len(clusterIDs),
                        len(set(trialGroups)),
                        int((window[1] - window[0]) / timeBinSize),
                    )
                )
                suballP[:] = np.nan
                subnormVal = np.empty((len(clusterIDs), len(set(trialGroups)), 2))
                subnormVal[:] = np.nan
                for cluster in range(len(clusterIDs)):
                    print("Processing cluster {clu}".format(clu=clusterIDs[cluster]))
                    _, _, _, _, _, ba = psthfn.psthAndBA(
                        spikeTimes[clu == clusterIDs[cluster]],
                        bslEventTime,
                        bslWin,
                        timeBinSize,
                    )

                    for trial in range(len(trialGroupLabel)):
                        psthbsl[trial] = np.mean(
                            np.divide(
                                ba[trialGroups == trialGroupLabel[trial]], timeBinSize
                            ),
                            axis=0,
                        )
                        normMn[trial] = np.mean(psthbsl[trial])
                        normStd[trial] = np.std(psthbsl[trial])

                    _, _, _, _, _, ba = psthfn.psthAndBA(
                        spikeTimes[clu == clusterIDs[cluster]],
                        eventTimesOnset,
                        window,
                        timeBinSize,
                    )

                    for trial in range(len(trialGroupLabel)):
                        psth[trial] = np.mean(
                            np.divide(
                                ba[trialGroups == trialGroupLabel[trial]], timeBinSize
                            ),
                            axis=0,
                        )

                        if normStd[trial] != 0:
                            suballP[cluster, trial] = (
                                psth[trial] - normMn[trial]
                            ) / normStd[trial]
                            subnormVal[cluster, trial, 0] = normMn[trial]
                            subnormVal[cluster, trial, 1] = normStd[trial]
                        else:
                            suballP[cluster, trial] = psth[trial]
                            subnormVal[
                                cluster, trial, 0
                            ] = np.nan  # nan lets us now it is a non-Gaussian neuron
                            subnormVal[
                                cluster, trial, 1
                            ] = np.nan  # same nan for non-Gaussian

                    allP[eventTimes[stim]["Stim"]] = suballP
                    normVal[eventTimes[stim]["Stim"]] = subnormVal

    return allP, normVal, windowlst


def clu_z_score_merged(
    sp_list: list[dict],
    event_list: list[dict],
    time_bin_list: list,
    window_list: list[list[float, float], list[float, float]],
    tg: bool,
    label_list: list,
) -> tuple[dict, dict, np.array]:
    if window_list is None:
        window_list = [[-30, -10], [-10, 30]]

    final_allP = dict()
    final_normVal = dict()

    for idx, curr_sp in enumerate(sp_list):
        curr_eventTimes = event_list[idx]
        if len(label_list) != 0:
            curr_label = label_list[idx]
        curr_cids = curr_sp["cids"]
        curr_filename = curr_sp["filename"]

        allP, normVal, _ = clusterzscore(
            sp=curr_sp,
            eventTimes=curr_eventTimes,
            time_bins=time_bin_list,
            tg=tg,
            window_list=window_list,
        )

        curr_file_hash = np.array([hash(str(cid) + curr_filename) for cid in curr_cids])

        for stim in allP.keys():
            sub_allP = allP[stim]
            sub_normVal = normVal[stim]
            curr_label = curr_label[stim]

            if idx == 0:
                final_allP[stim] = sub_allP
                final_normVal[stim] = sub_normVal
                file_ids = curr_file_hash
                final_trialgroups = list(curr_label.keys())
            else:
                n_trial = np.shape(final_allP[stim])[1]

                if np.shape(sub_allP)[1] != n_trial:
                    missing_groups_bool = [
                        True if key not in list(curr_label.keys()) else False
                        for key in final_trialgroups
                    ]
                    missing_groups = np.array(final_trialgroups)[missing_groups_bool]
                    missing_index = [int(float(value)) for value in missing_groups]
                    for index in missing_index:
                        if index < np.shape(sub_allP)[1]:
                            np.insert(sub_allP, index, np.nan, axis=1)
                            np.insert(sub_normVal, index, np.nan, axis=1)
                        else:
                            sub_allP = np.append(
                                sub_allP,
                                np.zeros(
                                    (
                                        np.shape(sub_allP)[0],
                                        1,
                                        np.shape(sub_allP)[2],
                                    )
                                ),
                                axis=1,
                            )
                            sub_normVal = np.append(
                                sub_normVal,
                                np.zeros(
                                    (
                                        np.shape(sub_normVal)[0],
                                        1,
                                        np.shape(sub_normVal)[2],
                                    )
                                ),
                                axis=1,
                            )
                            sub_allP[:, index - 1, :] = np.nan
                            sub_normVal[:, index - 1, :] = np.nan
                final_allP[stim] = np.vstack((final_allP[stim], sub_allP))
                final_normVal[stim] = np.vstack((final_normVal[stim], sub_normVal))
                file_ids = np.append(file_ids, curr_file_hash)

    return final_allP, final_normVal, file_ids
