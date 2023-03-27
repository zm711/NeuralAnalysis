#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:56:12 2023

@author: zacharymckenzie
"""

from scipy import stats
import numpy as np
from numba import jit
from .psthfunctions import time_stamps_to_bins


def latency_calculator(
    sp: dict,
    eventTimes: dict,
    timeBinSize: float,
    bsl_win: list[float, float],
    event_win: list[float, float],
    num_shuffle: int,
) -> dict:
    np.random.seed(42)
    spike_times: np.array = np.squeeze(sp["spikeTimes"])
    clu: np.array = np.squeeze(sp["clu"])
    cluster_ids: np.array = np.squeeze(sp["cids"])
    latency_dict = {}
    shuffled_dict = {}
    for idx, event in enumerate(eventTimes.keys()):
        curr_bsl = bsl_win[idx]
        curr_event = event_win[idx]
        events = eventTimes[event]["EventTime"]

        bsl_shuffled = (
            np.random.rand(num_shuffle, len(events)) * (curr_bsl[1] - curr_bsl[0])
            + curr_bsl[0]
        )
        trial_groups = eventTimes[event]["TrialGroup"]
        uniq_tgs = set(trial_groups)
        latency_dict["cluster_ids"] = cluster_ids
        shuffled_dict[event] = np.zeros((len(cluster_ids), len(uniq_tgs), num_shuffle))
        latency_dict[event] = np.zeros((len(cluster_ids), len(uniq_tgs)))
        latency_dict[event][:] = np.nan
        shuffled_dict[event][:] = np.nan
        for clu_idx, cluster in enumerate(cluster_ids):
            # print(f"processing {cluster}")
            these_spikes = spike_times[clu == cluster]
            ba_bsl, _ = time_stamps_to_bins(
                these_spikes, events, timeBinSize, curr_bsl[0], curr_bsl[1]
            )
            ba, _ = time_stamps_to_bins(
                these_spikes, events, timeBinSize, curr_event[0], curr_event[1]
            )
            # print(np.sum(ba)

            for tr_idx, trial in enumerate(uniq_tgs):
                ba_bsl_tg = ba_bsl[trial_groups == trial]
                mean_by_trial = np.mean(ba_bsl_tg, axis=1)

                bsl_mean = np.mean(mean_by_trial)

                ba_tg = ba[trial_groups == trial]

                if bsl_mean < 2:  # Mormann et al. 2008 J Neuro use 2 Hz cut off
                    median_lat, lat_std = latency_median(
                        ba_tg, time_bin_size=timeBinSize
                    )
                    latency_dict[event][clu_idx, tr_idx] = median_lat
                    latency_dict[event][clu_idx, tr_idx] = lat_std
                else:
                    # final_ba = np.sum(ba, axis=0)
                    # n_tg= np.shape(final_ba)[0]

                    # mean2 = latency_core2(bsl_fr = bsl_mean, firing_counts=final_ba, time_bin_size=timeBinSize, n_tg=n_tg)

                    mean_lat, std_lat = latency_core(
                        bsl_fr=bsl_mean, firing_counts=ba_tg, time_bin_size=timeBinSize
                    )

                    latency_dict[event][clu_idx, tr_idx] = mean_lat
                    # latency_dict[event][cluster][trial]['Lat2'] = mean2
                    latency_dict[event][clu_idx, tr_idx] = std_lat

                if np.isnan(latency_dict[event][clu_idx, tr_idx]):
                    continue
                else:
                    for shuffle in range(num_shuffle):
                        ba_shuf, _ = time_stamps_to_bins(
                            these_spikes,
                            events[trial_groups == trial]
                            + bsl_shuffled[shuffle, trial_groups == trial],
                            timeBinSize,
                            curr_event[0],
                            curr_event[1],
                        )
                        # shuffle_tg = ba_shuf[trial_groups == trial]
                        shuffle_tg = ba_shuf

                        if bsl_mean < 1:
                            median_lat, lat_std = latency_median(
                                shuffle_tg, time_bin_size=timeBinSize
                            )
                            shuffled_dict[event][clu_idx, tr_idx, shuffle] = median_lat
                        else:
                            mean_lat, std_lat = latency_core(
                                bsl_fr=bsl_mean,
                                firing_counts=shuffle_tg,
                                time_bin_size=timeBinSize,
                            )
                            shuffled_dict[event][clu_idx, tr_idx, shuffle] = mean_lat
    return latency_dict, shuffled_dict


"""idea modified from Chase and Young, 2007: PNAS
p_tn(>=n) = 1 - sum_m_n-1 ((rt)^m e^(-rt))/m!"""


def latency_core(
    bsl_fr: float, firing_counts: np.array, time_bin_size: float
) -> tuple[float, float]:
    latency = np.zeros((np.shape(firing_counts)[0],))
    for trial in range(np.shape(firing_counts)[0]):
        for n_bin in range(np.shape(firing_counts)[1] - 1):
            final_prob = 1 - stats.poisson.cdf(
                np.sum(firing_counts[trial][: n_bin + 1]) - 1,
                bsl_fr * ((n_bin + 1) * time_bin_size),
            )
            if final_prob <= 10e-6:
                break
        latency[trial] = (n_bin + 1) * time_bin_size

    mean_latency = np.mean(latency)
    std_latency = np.std(latency)

    return mean_latency, std_latency


"""This would be the original implementation of Chase and Young, 2007 bringing all
trial groups together"""


def latency_core2(
    bsl_fr: float, firing_counts: np.array, time_bin_size, n_tg: int
) -> float:
    for n_bin in range(len(firing_counts) - 1):
        final_prob = 1 - stats.poisson.cdf(
            np.sum(firing_counts[: n_bin + 1]),
            n_tg * bsl_fr * ((n_bin + 1) * time_bin_size),
        )
        if final_prob <= 10e-6:  # cutoff given in Chase and Young
            break

    latency_mean = (n_bin + 1) * time_bin_size

    return latency_mean


"""According to Mormann et al. 2008 if neurons fire less than 2Hz they won't really 
follow a poisson distribution and so instead just take latency to first spike as the 
latency and then get the median of the trials. If there are more than 3 closest data
sets with differences greater thean +/- 200 ms they exclude"""


@jit(nopython=True, cache=True)
def latency_median(
    firing_counts: np.array, time_bin_size: float
) -> tuple[float, float]:
    latency = np.zeros((np.shape(firing_counts)[0]))
    for trial in range(np.shape(firing_counts)[0]):
        min_spike_time = np.nonzero(firing_counts[trial])[0]
        if len(min_spike_time) == 0:
            latency[trial] = np.nan
        else:
            latency[trial] = (np.min(min_spike_time) + 1) * time_bin_size

    final_latency = np.nanmedian(latency)
    lat_std = np.nanstd(latency)

    exclude_count = 0
    for lat in latency:  # Mormann et al. JNeuro2008 use 0.2s as cutoff
        if lat > 0.2 + final_latency or lat < (final_latency - 0.2):
            exclude_count += 1
    if exclude_count >= 3:  # Morman does 3 closest
        final_latency = np.nan
        lat_std = np.nan

    return final_latency, lat_std
