# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:11:21 2023

@author: ZacharyMcKenzie
"""

import numpy as np
import pandas as pd
from analysis.psthfunctions import psthAndBA

"""wrapper for mergedClusterAnalysis which needs to iterate through sp_list and 
event_list. needs same window_dict and timeBinSize"""


def firingRateWinMCA(
    sp_list: list, event_list: list, window_dict: dict, timeBinSize: float
) -> pd.DataFrame:

    for idx in range(len(sp_list)):
        curr_sp = sp_list[idx]
        curr_eventTimes = event_list[idx]

        firing_rate_df = firingRateWin(
            curr_sp, curr_eventTimes, window_dict, timeBinSize
        )

        if idx == 0:
            merged_fr_df = firing_rate_df

        merged_fr_df = pd.concat([merged_fr_df, firing_rate_df], ignore_index=True)

    return merged_fr_df


def firingRateWin(
    sp: dict, eventTimes: dict, window_dict: dict, timeBinSize: float
) -> pd.DataFrame:

    if eventTimes.get("ADC1tot", 0):
        window_dict = {
            "Rest": [-30, -10],
            "Onset": [0, 2],
            "Sustained": [4, 16],
            "Offset": [18, 20],
            "Relief": [20, 30],
        }

    spike_times = np.squeeze(sp["spikeTimes"])
    filename = sp["filename"]
    file_hash = hash(filename)
    clu = np.squeeze(sp["clu"])

    cluster_ids = np.squeeze(sp["cids"])
    window_list = list()
    cluster_list = list()
    stim_list = list()
    trial_group_list = list()
    psth_list = list()
    file_list = list()

    for window in window_dict.keys():
        current_window = [float(i) for i in window_dict[window]]

        for cluster in cluster_ids:
            these_spikes = spike_times[clu == cluster]

            for stim in eventTimes.keys():
                event_onset = eventTimes[stim]["EventTime"]
                trial_group = np.array(eventTimes[stim]["TrialGroup"])
                stim_name = eventTimes[stim]["Stim"]

                _, _, _, _, _, ba = psthAndBA(
                    these_spikes, event_onset, current_window, timeBinSize
                )

                trials = set(trial_group)

                for trial in trials:
                    psth = np.mean(np.sum(ba[trial_group == trial], axis=1)) / (
                        current_window[-1] - current_window[0]
                    )
                    psth_list.append(psth)

                    window_list.append(window)
                    cluster_list.append(cluster)
                    stim_list.append(stim_name)
                    trial_group_list.append(trial)
                    file_list.append(file_hash)

    firing_rate_df = pd.DataFrame(
        {
            "File Hash": file_list,
            "Spikes/sec": psth_list,
            "Window": window_list,
            "IDs": cluster_list,
            "Stimulus": stim_list,
            "Trial Group": trial_group_list,
        }
    )

    firing_rate_df.sort_values(by="Trial Group", inplace=True)
    firing_rate_df["Trial Group"] = firing_rate_df["Trial Group"].apply(
        lambda x: str(x)
    )

    return firing_rate_df
