# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:14:18 2023

@author: ZacharyMcKenzie
"""

import numpy as np
from scipy import signal
import pandas as pd
from typing import Union


def multi_corr(psth_list: list, event_list: list, sm_param: Union[int, list]):
    total_r = np.zeros((len(psth_list), 2))
    for idx in len(psth_list):
        final_dataframe, mean_r, std_r = trial_corr(
            psth_list[idx], event_list[idx], sm_param
        )

        total_r[idx, 0] = mean_r
        total_r[idx, 1] = std_r

    return total_r


def trial_corr(
    psthvalues: dict, eventTimes: dict, sm_param: Union[int, list]
) -> tuple[pd.DataFrame, float, float]:
    """
    function for calculation Pearson correlation coefficients between trial groups for
    gaussian smoothed data.

    Parameters
    ----------
    psthvalues : dict
        raster data for each cluster
    eventTimes : dict
        stimulus data
    sm_param : Union[int, list]
        guassian smoothing filter in milliseconds

    Returns
    -------
    final_dataframe : pd.DataFrame
        dataframe of trial-trial correlations
    mean_r : float
        mean correlation value
    std_r : float
        std of correlation values

    """

    eventLst = list()
    for stimE in eventTimes.keys():
        eventLst.append(stimE)
    stim_list = list()
    corr_vals = list()
    cluster_id = list()
    trial_list = list()
    for index, stim in enumerate(psthvalues.keys()):
        if type(sm_param) != list:
            gw = signal.windows.gaussian(
                round(sm_param * 6), (round(sm_param * 6) - 1) / 6
            )  # this takes std vs alpha for matlab version
            # std = (L-1)/2alpha (matlab alpha = 3)
        else:
            gw = signal.windows.gaussian(
                round(sm_param[index] * 6), (round(sm_param[index] * 6) - 1) / 6
            )
        trialGroup = np.array(eventTimes[eventLst[index]]["TrialGroup"])

        for cluster in psthvalues[stim].keys():
            ba = psthvalues[stim][cluster]["BinnedArray"]  # BinnedArray are the counts
            bins = psthvalues[stim][cluster]["Bins"]  # these are the centers of bins
            tg = list(np.unique(trialGroup))

            smWin = gw / np.sum(gw)
            # baT = np.reshape(ba, (np.shape(ba)[1], np.shape(ba)[0]))
            baSm = np.zeros((np.shape(ba)[0], np.shape(ba)[1]))  # memory allocation

            for row in range(np.shape(ba)[0]):
                baSm[row] = signal.convolve(ba[row], smWin, mode="same") / (
                    bins[1] - bins[0]
                )  # convolution to apply gaussians

            for trial in tg:
                baSm_sub = baSm[trialGroup == trial]
                smoothed_trial_df = pd.DataFrame(baSm_sub.T)
                trial_corr = smoothed_trial_df.corr()

                trial_corr_no_one = trial_corr[trial_corr != 1]
                final_corr_val = np.nanmean(trial_corr_no_one.iloc[0, :])

                corr_vals.append(final_corr_val)
                cluster_id.append(cluster)
                trial_list.append(trial)
                stim_list.append(stim)

        final_dataframe = pd.DataFrame(
            {
                "Cluster": cluster_id,
                "Trial Group": trial_list,
                "R score": corr_vals,
                "Stim": stim_list,
            }
        )

        # plot_by_animal(final_dataframe, stim)
        mean_r = final_dataframe["R score"].mean(axis=0)
        std_r = final_dataframe["R score"].std(axis=0)

    return final_dataframe, mean_r, std_r
