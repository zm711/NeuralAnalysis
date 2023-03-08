# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 10:06:41 2022

@author: ZacharyMcKenzie
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from  ..analysis import psthfunctions as psfn
import scipy.cluster.hierarchy as sch


def neuronCorr(
    sp: dict,
    eventTimes: dict,
    allP: dict,
    normVal,
    window: list,
    datatype="frraw",
    timeBinSize=0.001,
    tg=False,
    labels=None,
) -> None:

    clusterIDs = sp["cids"]
    psthvalues, windowlst = psfn.rasterPSTH(sp, eventTimes, timeBinSize=timeBinSize)
    if datatype == "frbsl":
        print("please enter baseline window for this run of code")
        psthvaluesbsl, _ = psfn.rasterPSTH(sp, eventTimes, timeBinSize=timeBinSize)
    elif datatype == "frsm":
        param = int(input("input the smoothing factor"))
        gw = signal.windows.gaussian(round(param * 6), (round(param * 6) - 1) / 6)
        smWin = gw / np.sum(gw)
    elif datatype == "frraw":
        pass
    elif datatype == "zscore":
        neuroCorrZ(
            allP,
            eventTimes,
            normVal,
            window,
            timeBinSize=timeBinSize,
            tg=tg,
            labels=labels,
        )
        return "Z score completed"
    else:

        return "Incorrect datatype entered (options are frraw, frbsl, frsm, zscore)"

    for (idx, stim) in enumerate(psthvalues.keys()):

        neuronData = pd.DataFrame()

        if tg == False:

            for cluster in clusterIDs:

                binned_array = psthvalues[stim][str(cluster)]["BinnedArray"]
                psth = np.mean(binned_array, axis=0)

                if datatype == "frraw":

                    psth_final = psth

                elif datatype == "frbsl":
                    ba_bsl = psthvaluesbsl[stim][str(cluster)]["BinnedArray"]

                    psth_bsl_mean = np.mean(ba_bsl)

                    psth_final = psth - psth_bsl_mean

                elif datatype == "frsm":
                    bins = psthvalues[stim][str(cluster)]["Bins"]

                    baSm = np.zeros(
                        (np.shape(binned_array)[0], np.shape(binned_array)[1])
                    )

                    for row in range(np.shape(binned_array)[0]):
                        baSm[row] = signal.convolve(
                            binned_array[row], smWin, mode="same"
                        ) / (bins[1] - bins[0])

                    psth_final = np.mean(baSm, axis=0)

                neuronData[str(cluster)] = psth_final

            corrPlot(neuronData, sample_list=None, datatype=datatype, stim=stim)

        else:
            trial_group = list()
            for channel in eventTimes.keys():
                trial_group += [list(eventTimes[channel]["TrialGroup"])]

            trialgroup = trial_group[idx]

            tg_set = set(trialgroup)

            cluster_list = list()
            trial_list = list()
            psth_list = list()
            bins_list = list()
            for cluster in clusterIDs:
                binned_array = psthvalues[stim][str(cluster)]["BinnedArray"]
                # psth = np.zeros((len(tg_set), np.shape(binned_array)[1]))
                # psth_bsl_mean = np.zeros((len(tg_set), 1))

                for trial in tg_set:
                    psth = np.mean(binned_array[trialgroup == trial], axis=0)

                    if datatype == "frraw":

                        psth_final = psth

                    elif datatype == "frbsl":
                        ba_bsl = psthvaluesbsl[stim][str(cluster)]["BinnedArray"]

                        psth_bsl_mean = np.mean(ba_bsl[trialgroup == trial])

                        psth_final = psth - psth_bsl_mean

                    elif datatype == "frsm":
                        bins = psthvalues[stim][str(cluster)]["Bins"]

                        baSm = np.zeros(
                            (np.shape(binned_array)[0], np.shape(binned_array)[1])
                        )

                        for row in range(np.shape(binned_array)[0]):
                            baSm[row] = signal.convolve(
                                binned_array[row], smWin, mode="same"
                            ) / (bins[1] - bins[0])

                        psth_final = np.mean(baSm[trialgroup == trial], axis=0)

                    psth_list += list(psth_final)
                    trial_list += len(psth_final) * [trial]
                    cluster_list += len(psth_final) * [cluster]
                    bins_list += list(
                        np.linspace(0, len(psth_final) + 1, len(psth_final))
                    )

            neuronData = pd.DataFrame(
                {
                    "Trial Group": trial_list,
                    "Cluster Number": cluster_list,
                    "PSTH Values": psth_list,
                    "Bins": bins_list,
                }
            )

            for trial in tg_set:
                neuronData_sub = neuronData[neuronData["Trial Group"] == trial]
                neuronData_sub.drop(["Trial Group"], axis=1)
                neuronData_final = neuronData_sub.pivot(
                    index="Bins", columns="Cluster Number", values="PSTH Values"
                )

                if labels:
                    trial_label = labels[str(trial)]
                else:
                    trial_label = str(trial)

                corrPlot(
                    neuronData_final,
                    sample_list=None,
                    datatype=datatype,
                    stim=stim,
                    trial=trial_label,
                )


def neuroCorrZ(
    allP: dict,
    eventTimes: dict,
    normVal: dict,
    window: list,
    timeBinSize: float,
    tg: bool,
    labels: dict,
):

    if not allP or not normVal:
        raise Exception("To run 'zscore' you have to have already run `cluZscore` ")

    eventLst = (
        list()
    )  # need to create a list of the stim since allP only stores stim name not Intan channel
    for key in eventTimes.keys():
        eventLst.append(key)

    """First we iterate over each stimulus and generate a sub zscore in allP_sub"""
    for (i, stim) in enumerate(allP.keys()):
        curr_window = window[i]
        allP_sub = allP[stim]
        normVal_sub = normVal[stim]

        """We create the len of our events to mark out events on the graph"""
        event_len = np.mean(eventTimes[eventLst[i]]["Lengths"]) / timeBinSize
        time_start = curr_window[0]
        if time_start < 0:  # in case we have - times relative to event start
            event_len += abs(time_start / timeBinSize)
        event_len = int(event_len)  # need to be an integer for indexing

        time_end = curr_window[1]
        if tg == False:
            time_bins = list(np.linspace(time_start, time_end, np.shape(allP_sub)[1]))
        else:
            time_bins = list(np.linspace(time_start, time_end, np.shape(allP_sub)[2]))
        time_bins = [float("%.3f" % x) for x in time_bins]  # 3decimal are min allowed

        """I like putting stuff into dataframes since seaborn places super well with df's. 
        so the next set of code is just setting up the dataframes"""
        final_time_bins = list()
        final_to_keep = list()
        zscore_final = list()

        if tg == False:  # if we didn't separate by trial groups
            to_keep = list(
                set(np.where(~np.isnan(normVal_sub))[0])
            )  # nan in bsl-no zscore
            allP_subKeep = allP_sub[to_keep]  # make our sublist

            """Since the dataframe needs a value for each row I basically need to take my values
            and repeat them the appropriate number of times.... ie each cluster needs a timebin
            and zscore value, so I create duplicate values so that the nClu becomes nClu x ntimebin
            structure"""

            for (idx, i) in enumerate(to_keep):
                final_time_bins += time_bins
                final_to_keep += len(time_bins) * [i]
                zscore_final += list(allP_subKeep[idx, :])

            zscore = pd.DataFrame(
                {  # load the data into the dataframe
                    "Time (s)": final_time_bins,
                    "Clusters": final_to_keep,
                    "Zscore": zscore_final,
                }
            )

            zscore_pivot = zscore.pivot(
                index="Time (s)", columns="Clusters", values="Zscore"
            )
            corrPlot(
                zscore_pivot, sample_list=None, datatype="zscore", stim=stim, trial=None
            )

        else:  # if we do have trial group separated out

            trialGroups = list(eventTimes[eventLst[i]]["TrialGroup"])
            tgs = sorted(list(set(trialGroups)))  # need the set of trial groups

            """As above we need to generate a dataframe with rows = to nClu x ntimeBins for
            each TrialGroups seaparately, so the loops below create all the data needed for 
            each trial group"""
            for trial in tgs:
                final_time_bins = list()
                final_to_keep = list()
                zscore_final = list()

                allP_sub_tg = allP_sub[:, tgs == trial, :]

                to_keep = list(set(np.where(~np.isnan(allP_sub_tg))[0]))

                allP_sub_toKeep = np.squeeze(allP_sub_tg[to_keep])

                for (idx, i) in enumerate(to_keep):
                    final_time_bins += time_bins
                    final_to_keep += len(time_bins) * [i]
                    zscore_final += list(allP_sub_toKeep[idx, :])

                zscore = pd.DataFrame(
                    {
                        "Time (s)": final_time_bins,
                        "Clusters": final_to_keep,
                        "Zscore": zscore_final,
                    }
                )

                if labels:
                    trial_name = labels[str(trial)]
                else:
                    trial_name = trial

                zscore_pivot = zscore.pivot(
                    index="Time (s)", columns="Clusters", values="Zscore"
                )

                corrPlot(
                    zscore_pivot,
                    sample_list=None,
                    datatype="zscore",
                    stim=stim,
                    trial=trial_name,
                )


def corrPlot(
    neuronData: pd.DataFrame, sample_list=None, datatype="frraw", stim=None, trial=None
):

    if sample_list:
        neuro_df = pd.DataFrame(neuronData, columns=sample_list)
    else:
        neuro_df = neuronData

    neuro_corr = neuro_df.corr()

    if any(np.isnan(neuro_corr)):
        neuro_corr.dropna(how="all", inplace=True)
        neuro_corr.dropna(axis=1, how="all", inplace=True)

    if datatype == "frraw":
        correction = "Raw Data"
    elif datatype == "frbsl":
        correction = "Baseline Subtraction"
    elif datatype == "frsm":
        correction = "Gaussian Smoothing"
    elif datatype == "zscore":
        correction = "Z scored"

    mask = neuro_corr == 1

    f, ax = plt.subplots(figsize=(10, 8))  # can be whatever size

    if datatype == "frbsl" or datatype == "zscore":

        ax = sns.heatmap(
            cluster_corr(neuro_corr),
            mask=mask,
            vmin=-1,
            cmap="vlag",
            cbar_kws={"label": "Pearson R Score"},
        )
    else:
        ax = sns.heatmap(
            cluster_corr(neuro_corr),
            mask=mask,
            vmin=0,
            cmap="Reds",
            cbar_kws={"label": "Pearson R Score"},
        )

    if trial:
        plt.title(
            f"Corr of {stim} with {correction} for {trial} trial group", weight="bold"
        )
    else:
        plt.title(f"Corr of {stim} with {correction}", weight="bold")
    plt.figure(dpi=1200)
    plt.show()


"""Strategy from Wil Yegelwel on wil.yegelwel.com"""


def cluster_corr(corr_array: pd.DataFrame, inplace=False):

    """Input:
    corr_array: NxN correlation matrix from pandas or numpy

    Returns:
        NxN correlation matrix sorted by size of correlation either a df or numpy array

    """

    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method="complete")
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(
        linkage, cluster_distance_threshold, criterion="distance"
    )
    idx = np.argsort(idx_to_cluster_array)
    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]

    return corr_array[idx, :][:, idx]
