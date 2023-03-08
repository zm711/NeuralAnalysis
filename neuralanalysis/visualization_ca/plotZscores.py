#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 18:44:01 2022

@author: zacharymckenzie

plotZscores has two functions first plotZscores takes in all the values we need to make heatmaps and organized the data either overall
or by trial groups based on the tg flag. Then plotZscoreCore takes in the preprocessed data and plots it.

INPUTS: allP: a dict of the zscores organized by stimuli
        normVal: a set of the mean and std of the baseline firing rate
        eventTimes: a dict of stimuli
        window: a list of lists with a start time and end time used to make psth/zscore
        timeBinSize: a float that gives the binSize used during psth/zscore calculation
        tg is a flag on whether data was separated by trial group or averaged overall
        labels: an optional dictionary to convert trial groups from numbers to desired labels
        
OUTPUT: graphs of the z-scores by each stimulus OR each trial group for each stimulus
        responsive_neurons: a dict of neurons sorted based on `events`, `relief`, and
        `onset`. For example responsive_neurons['event'][:(len(responsive_neurons['event'])/4)] 
        would only take the top 25% of neurons responding to stimulus

"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from itertools import compress
from .plot_z_settings import z_score_cutoff, raw_count, sorter_dict_adder


@sorter_dict_adder
def plotZscores(
    allP: dict,
    normVal: dict,
    eventTimes: dict,
    window: list[list[float, float]],
    time_bin_list: list,
    tg: bool,
    sorter_dict=None,
    labels=None,
    time_point=0,
    plot=True,
) -> tuple[dict, dict]:

    eventLst = (
        list()
    )  # need to create a list of the stim since allP only stores stim name not Intan channel
    for key in eventTimes.keys():
        eventLst.append(key)

    responsive_neurons = {}  # finally return dict
    responsive_neurons_raw = {}

    """First we iterate over each stimulus and generate a sub zscore in allP_sub"""
    for (i, stim) in enumerate(allP.keys()):
        curr_window: list = window[i]
        allP_sub: np.array = allP[stim]
        normVal_sub: np.array = normVal[stim]
        timeBinSize = time_bin_list[i]
        responsive_neurons[stim] = {}
        responsive_neurons_raw[stim] = {}
        sub_label = labels[stim]

        """We create the len of our events to mark out events on the graph"""
        event_len: float = np.mean(eventTimes[eventLst[i]]["Lengths"]) / timeBinSize
        time_start: float = curr_window[0]
        if time_start < 0:  # correction if start time is before the event
            event_len += abs(time_start / timeBinSize)
        event_len = int(event_len)  # integer for indexing

        time_end: float = curr_window[1]

        """we make time bins based on organization of allP which is dependent on the
        presence of trial groups or not"""
        if tg == False:
            time_bins = list(
                np.linspace(time_start, time_end, np.shape(allP_sub)[1])
            )  # need to make bins from arb units
        else:
            time_bins = list(np.linspace(time_start, time_end, np.shape(allP_sub)[2]))

        time_bins = [
            float("%.3f" % x) for x in time_bins
        ]  # long floats convert to 3 decimals. Req. Minimum

        zero_point = int(
            abs(time_start) / timeBinSize
        )  # if we start in neg time this converts it to stim_start "0" in stim time
        time_point = int(time_point / timeBinSize)  # convert time_point to bins

        """I like putting stuff into dataframes since seaborn places super well with df. 
        so the next set of code is just setting up the dataframes. Of note we have
        two sets of neurons the first set are the 'Gaussian' which have z-scores
        the second set are the ones with mean & std of 0 and so have only raw spike
        counts so I basically repeat the process for both sets of neurons"""

        final_time_bins = list()  # z-scored
        final_to_keep = list()
        zscore_final = list()

        final_raw_time_bins = list()  # raw count neurons
        final_raw = list()
        raw_spikes_list = list()

        if sorter_dict is None:
            sorter_dict = {
                "sustained": [zero_point, event_len],
                "relief": [event_len, len(time_bins)],
                "onset": [zero_point, zero_point + time_point],
                "onset-offset": [
                    zero_point,
                    zero_point + time_point,
                    event_len - time_point,
                    event_len + time_point,
                ],
                "inhib": [zero_point, zero_point + time_point],
            }

        if tg == False:  # if we didn't separate by trial groups
            to_keep = list(
                set(np.where(~np.isnan(normVal_sub))[0])
            )  # nans d/t bsl firing--need to take only z scored neurons
            allP_subKeep = allP_sub[to_keep]  # make our sublist
            raw_spikes = list(set(np.where(np.isnan(normVal_sub))[0]))
            allP_raw_spikes = allP_sub[raw_spikes]

            """Since the dataframe needs a value for each row I basically need to take my values
            and repeat them the appropriate number of times.... ie each cluster needs a timebin
            and zscore value, so I create duplicate values so that the nClu becomes nClu x ntimebin
            structure"""

            for (idx, i) in enumerate(to_keep):
                final_time_bins += time_bins
                final_to_keep += len(time_bins) * [i]
                zscore_final += list(allP_subKeep[idx, :])

            zscore = pd.DataFrame(
                {
                    "Time (s)": final_time_bins,
                    "Units": final_to_keep,
                    "Zscore": zscore_final,
                }
            )

            zscore_pivot = zscore.pivot(
                index="Units", columns="Time (s)", values="Zscore"
            )

            resp_dict = responsive_neurons_calculator(
                zscore_pivot, zero_point, sorter_dict, timeBinSize
            )

            """we also need to account for neurons which are low baseline activity 
            which only respond to the stimulus. These neurons are not z-scoreable since
            their mean and std = 0 . So we take raw spike counts/sec only. """
            for (idy, spike) in enumerate(raw_spikes):
                final_raw_time_bins += time_bins
                final_raw += len(time_bins) * [spike]
                raw_spikes_list += list(allP_raw_spikes[idy, :])

            raw_spike_dict = pd.DataFrame(
                {
                    "Time (s)": final_raw_time_bins,
                    "Units": final_raw,
                    "Spikes/sec": raw_spikes_list,
                }
            )

            raw_spike_pivot = raw_spike_dict.pivot(
                index="Units", columns="Time (s)", values="Spikes/sec"
            )

            resp_raw_dict = responsive_neuron_calculator_nonz(
                raw_spike_pivot, zero_point, sorter_dict, timeBinSize
            )

            for sorter in resp_dict.keys():
                responsive_neurons[stim][sorter] = list(
                    compress(to_keep, resp_dict[sorter])
                )
                responsive_neurons_raw[stim][sorter] = list(
                    compress(raw_spikes, resp_raw_dict[sorter])
                )

            if plot == True:
                """I plot organized by during stimulus and after stimulus which are common
                for my baro neuros. The `event` flag should look pretty nice for most stims
                """
                plotZscoreCore(
                    zscore_pivot, zero_point, event_len, stim, sorter="event"
                )

                plotZscoreCore(
                    zscore_pivot, zero_point, event_len, stim, sorter="relief"
                )

                plotZscoreCore(
                    raw_spike_pivot,
                    zero_point,
                    event_len,
                    stim + " raw",
                    sorter="event",
                )

        else:  # if we do have trial group separated out

            trialGroups = np.array(list(eventTimes[eventLst[i]]["TrialGroup"]),dtype=np.float64)
            tgs = sorted(list(set(trialGroups)))  # need the set of trial groups

            """As above we need to generate a dataframe with rows = to nClu x ntimeBins for
            each TrialGroups seaparately, so the loops below create all the data needed for 
            each trial group"""
            for trial in tgs:
                final_time_bins = list()
                final_to_keep = list()
                zscore_final = list()
                final_raw_time_bins = list()
                final_raw = list()
                raw_spikes_list = list()

                allP_sub_tg = allP_sub[:, tgs == trial, :]
                normVal_sub_tg = normVal_sub[:, tgs == trial, :]

                to_keep = list(set(np.where(~np.isnan(normVal_sub_tg))[0]))

                raw_spikes = list(set(np.where(np.isnan(normVal_sub_tg))[0]))

                allP_sub_toKeep = np.squeeze(allP_sub_tg[to_keep])
                allP_sub_raw = np.squeeze(allP_sub_tg[raw_spikes])

                responsive_neurons[stim][trial] = {}
                responsive_neurons_raw[stim][trial] = {}

                for (idx, i) in enumerate(to_keep):
                    final_time_bins += time_bins
                    final_to_keep += len(time_bins) * [i]
                    zscore_final += list(allP_sub_toKeep[idx, :])

                zscore = pd.DataFrame(
                    {
                        "Time (s)": final_time_bins,
                        "Units": final_to_keep,
                        "Zscore": zscore_final,
                    }
                )

                for (idy, spike) in enumerate(raw_spikes):
                    final_raw_time_bins += time_bins
                    final_raw += len(time_bins) * [spike]
                    if len(np.shape(allP_sub_raw)) > 1:
                        raw_spikes_list += list(allP_sub_raw[idy, :])
                    else:
                        raw_spikes_list += list(allP_sub_raw)

                raw_spike_dict = pd.DataFrame(
                    {
                        "Time (s)": final_raw_time_bins,
                        "Units": final_raw,
                        "Spikes/sec": raw_spikes_list,
                    }
                )

                if labels:
                    trial_name = sub_label[str(trial)]
                else:
                    trial_name = trial

                zscore_pivot = zscore.pivot(
                    index="Units", columns="Time (s)", values="Zscore"
                )

                raw_spike_pivot = raw_spike_dict.pivot(
                    index="Units", columns="Time (s)", values="Spikes/sec"
                )

                resp_dict = responsive_neurons_calculator(
                    zscore_pivot, zero_point, sorter_dict, timeBinSize
                )
                resp_raw_dict = responsive_neuron_calculator_nonz(
                    raw_spike_pivot, zero_point, sorter_dict, timeBinSize
                )

                for sorter in resp_dict.keys():
                    responsive_neurons[stim][trial][sorter] = list(
                        compress(to_keep, resp_dict[sorter])
                    )
                    responsive_neurons_raw[stim][trial][sorter] = list(
                        compress(raw_spikes, resp_raw_dict[sorter])
                    )

                if plot == True:
                    plotZscoreCore(
                        zscore_pivot,
                        zero_point,
                        event_len,
                        stim,
                        trial=trial_name,
                        sorter="event",
                    )

                    plotZscoreCore(
                        zscore_pivot,
                        zero_point,
                        event_len,
                        stim,
                        trial=trial_name,
                        sorter="relief",
                    )

                    # for non-z scoreable values
                    plotZscoreCore(
                        raw_spike_pivot,
                        zero_point,
                        event_len,
                        stim + " raw",
                        trial=trial_name,
                        sorter="event",
                    )

    return responsive_neurons, responsive_neurons_raw


"""plotZscoreCore is the function for plotting z scores--. It takes in a dataframe
pivot, ie, a dataframe which has only clusters for the rows and times for the columnns. 
The values are the z-scores. 
INPUTS:
         zscore_pivot: dataframe
         zero_point: where stimulus starts converted to bin number
         event_len: numbers of bins that a stim lasts
         stim: label for the stimulus type
         trial: which trial group is occurring
         sorter: the type of response sorting desired, `event` looks over event
                 `relief` looks after event
                 `onset` looks from zero_point to time_point
        time_point: the time to be used for `onset` sorter
OUTPUTS:
        z_idx: a list of the neurons sorted by the `sorter` type in order of most
        responsive to least responsive"""


def plotZscoreCore(
    zscore_pivot: pd.DataFrame,
    zero_point: int,
    event_len: int,
    stim: str,
    sorter: str,
    trial=False,
) -> None:

    """So I set by highest Z score sum ocurring within the stimulus time
    `ie` between zero_point and event_len. Then we argsort the negative
    values to get the descending id's of clusters. finally we create our
    z_sorted so that it goes descending

    for relief I take the end of the event and sum of z-scores there. I see
    a lot of barostat neurons which are inhibited and then burst. This let me
    sees that"""

    if sorter == "event":
        z_sorter = zscore_pivot.iloc[:, zero_point:event_len]

    elif sorter == "relief":
        z_sorter = zscore_pivot.iloc[:, event_len:]

    z_idx = np.argsort(-z_sorter.sum(axis=1))  # idxmax(axis=1)
    z_sorted = zscore_pivot.iloc[z_idx]

    # future code for if people need to keep neuron ids in same order for comparisons
    if sorter != "event" and sorter != "relief":
        z_sorted = zscore_pivot

    """This is just for picking plotting values that maximize our visual
    range"""
    if sorter != "inhib":
        z_max = z_sorted.max(axis=1).max()

        if z_max > 40:
            vmax_val = 35
        elif z_max > 20:
            vmax_val = z_max - 10
        else:
            vmax_val = z_max - 5

        """Actual plotting stuff below"""

        if "raw" in stim:
            cbar_label = "Spikes/sec"
            vmin = 0
            cmap = "viridis"
            line_color = "white"
        else:
            cbar_label = "Z score"
            vmin = -vmax_val
            cmap = "vlag"
            line_color = "black"

        fig = plt.subplots(figsize=(10, 8))
        ax = sns.heatmap(
            data=z_sorted,
            vmin=vmin,
            vmax=vmax_val,
            cmap=cmap,
            cbar_kws={"label": cbar_label},
        )
        x_list = list()
        for (
            label
        ) in (
            ax.get_xticklabels()
        ):  # grabs the values that seaborn autotakes from dataframe
            x_list.append(
                label.get_text()
            )  # converts matplot Text object to normal string
        if len(x_list) == 0:
            pass
        else:
            if abs(float(x_list[0])) > 1.2 and abs(float(x_list[-1])) > 1.2:
                x_list = [
                    "%.1f" % float(x) for x in x_list
                ]  # converts to 1 decimal point
            else:
                x_list = [
                    "%.2f" % float(x) for x in x_list
                ]  # if shorter than 1 sec time frame give two decimals
            ax.set_xticklabels(x_list)  # set the xlabels

        plt.axvline(zero_point, color=line_color, linestyle=":", linewidth=0.5)
        plt.axvline(event_len, color=line_color, linestyle=":", linewidth=0.5)
        plt.rc("axes", labelsize=14)
        plt.rc("xtick", labelsize=12)

        if trial == False:
            plt.title(f"{stim.title()}, {sorter.title()}", weight="bold")
        else:
            plt.title(f"{stim.title()}, {trial}, {sorter.title()}", weight="bold")

        plt.tight_layout()
        plt.figure(dpi=1200)
        plt.show()


"""new function for calculating responsive neurons. I removed this from the plotting
function to try to follow one function one transformation. Basically this takes in 
the zscore data, and the sorter_dict to check for responsivity values"""


@z_score_cutoff  # turn off decorator to allow for debugging of z -score cutoff
def responsive_neurons_calculator(
    zscore_pivot: pd.DataFrame,
    zero_point: int,
    sorter_dict: dict,
    time_bin_size: float,
    **kwargs,
) -> dict:

    for key, value in kwargs.items():
        if key == "inhib":
            inhib = value[0]
            inhib_len = value[1]
        elif key == "sustained":
            sust = value[0]
            sus_len = value[1]
        elif key == "onset":
            onset = value[0]
            onset_len = value[1]
        elif key == "offset":
            offset = value[0]
            offset_len = value[1]

    if len(kwargs) == 0:
        inhib = -2
        inhib_len = 3
        sust = 3
        sus_len = 3
        onset = 3
        onset_len = 3
        offset = 2.5
        offset_len = 3

    # = zscore_pivot.iloc[:, :zero_point].mean(
    #    axis=1
    # ) + 0.75 * zscore_pivot.iloc[:, :zero_point].std(axis=1)
    # mean_baseline_below = zscore_pivot.iloc[:, :zero_point].mean(
    #     axis=1
    # ) - zscore_pivot.iloc[:, :zero_point].std(axis=1)

    # print(sust) uncomment to see what value function is currently using
    responsive_neurons = {}

    for sorter in sorter_dict.keys():

        responsive_neurons[sorter] = list()
        event_window = sorter_dict[sorter]
        sorter = sorter.lower()
        if len(event_window) == 4:  # this indicates onset-offset only
            z_sorter1 = zscore_pivot.iloc[:, event_window[0] : event_window[1]]
            z_sorter2 = zscore_pivot.iloc[:, event_window[2] : event_window[3]]
        else:
            z_sorter = zscore_pivot.iloc[
                :, event_window[0] : event_window[1]
            ]  # all others

        """if inhib we look for -2 std Chirella et al used -1.
        for sustained I used >3SDs (Alan Emanuel et al 2021 Nature used 2.58 to get the
        99%CI of at least 3 bins and mean above baseline)
        for relief I do the same, but with a different time period
        for onset-offset I just look for short extreme--current 10 SDs for 2 bins"""
        if sorter == "inhib":
            resp_neurons = list(
                np.logical_and(
                    z_sorter[z_sorter < inhib].count(axis=1) > inhib_len,
                    z_sorter.min(axis=1) < inhib,
                )
            )
        elif sorter == "sustained" or sorter == "relief":
            resp_neurons = list(
                np.logical_and(
                    z_sorter[z_sorter > sust].count(axis=1) > sus_len,
                    z_sorter.max(axis=1) > 3,
                )
            )
        elif sorter == "onoff":
            resp_neurons = list(
                np.logical_and(
                    z_sorter1[z_sorter1 > onset].count(axis=1) > onset_len,
                    z_sorter2[z_sorter2 > offset].count(axis=1) > offset_len,
                )
            )
        else:
            resp_neurons = list(z_sorter[z_sorter > onset].count(axis=1) > onset_len)

        responsive_neurons[sorter] = resp_neurons

    return responsive_neurons


@raw_count
def responsive_neuron_calculator_nonz(
    raw_pivot: pd.DataFrame,
    zero_point: int,
    sorter_dict: dict,
    time_bin_size: float,
    **kwargs,
) -> dict:

    for key, value in kwargs.items():

        if key == "sustained":
            sust = value

    if len(kwargs) == 0:
        sust = 75

    responsive_neurons = {}
    for sorter in sorter_dict.keys():

        responsive_neurons[sorter] = list()
        event_window = sorter_dict[sorter]
        sorter = sorter.lower()
        if len(event_window) == 4:  # this indicates onset-offset only
            raw_pivot_sorter = (
                raw_pivot.iloc[:, event_window[0] : event_window[1]]
                + raw_pivot.iloc[:, event_window[2] : event_window[3]]
            )
        else:
            raw_pivot_sorter = raw_pivot.iloc[
                :, event_window[0] : event_window[1]
            ]  # all o

        if sorter == "inhib":
            resp_neurons = list()
        elif sorter == "sustained" or sorter == "relief":
            resp_neurons = list(raw_pivot_sorter.sum(axis=1) > sust)
        else:
            resp_neurons = list(raw_pivot_sorter.sum(axis=1) > 25)
        responsive_neurons[sorter] = resp_neurons

    return responsive_neurons
