# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:05:32 2023

@author: ZacharyMcKenzie
"""


import pandas as pd
import numpy as np
import hashlib
from typing import Optional


"""labelGenerator is a function which takes in the eventTimes dict and automatically
generates a labels dict based on the V labeled in the labels. In order to make sure
everything is an integer I divide by 0.25 in the stimulus_setupzm.py so I need to 
multiple by 0.25. Then we multiple by 20 since the barostat is 20mmHg/V."""


def labelGenerator(eventTimes: dict) -> dict:
    barostat = eventTimes["ADC1tot"]
    trial_groups = barostat["TrialGroup"]

    trials = set(trial_groups)  # will have repeats so take the set

    labels = {"Barostat": {}}

    for trial in trials:
        labels["Barostat"][str(trial)] = str(trial * 0.25 * 20) + " mmHg"

    return labels


"""responseDF generates a dataframe of responsive neurons which also have high enough
isolation distance if qcthres is give--I default to 10, but setting this to False or 0
would prevent qc from being applied. I include the trial group if give as well as the 
sorter defined in plotZ and the unit/cluster id. Finally I apply a hash of the file
to blind to genotype etc that is include in the master list"""


def responseDF(
    responsive_neurons: dict,
    isiv: dict,
    qcvalues: dict,
    sp: dict,
    labels: dict,
    qcthres: float,
    sil: float,
    isi=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    function for creating responsive neuron DataFrame

    Parameters
    ----------
    responsive_neurons : dict
        dictionary of responsive neurons as determined by z-score cutoffs
    isiv : dict
        dictionary of interspike interval violations.
    qcvalues : dict
        the isolation distances, contamination rates, and simplified silhouette scores
    sp : dict
        spike train and associated data
    labels : dict
        labels to translate integers to strings for labels.
    qcthres : float
        isolation distance value cutoff for qc (0, inf)
    sil : float
        silhouette score value cutoff (-1, 1)
    isi : TYPE, optional
        Optional interspike violation fraction cutoff. (0,1) 0 with no violation The default is None.

    Returns
    -------
    resp_neuron_df : pd.DataFrame
        DataFrame of responsive neurons which have passed qc
    non_resp_df : TYPE
        DataFrame of non-responsive neurons which have passed qc

    """
    # cids = sp["cids"]
    try:
        unit_quality = qcvalues["uQ"]
        run_qc = True
    except TypeError:
        print("no qcvalues")
        run_qc = False
    try:
        sil_quality = qcvalues["sil"]
        run_sil = True
    except TypeError:
        print("no silhouette scores")
        run_sil = False
    filename = sp["filename"]
    neuron_idx = hashlib.sha256((filename).encode()).hexdigest()
    cids = sp["cids"]
    noise = sp["noise"]

    if qcthres > 0 and len(cids) != len(unit_quality):
        unit_quality = unit_quality[~noise]
    if sil > 0 and len(cids) != len(sil_quality):
        sil_quality = sil_quality[~noise]

    if isiv is not None:
        viol_percent = list()
        for key in isiv.keys():
            viol_percent.append(isiv[key]["nViol"])

    stim_list = list()
    sorter_list = list()
    trialgroup_list = list()
    idx_list = list()
    qc_list = list()
    viol_list = list()
    resp_list = list()
    sil_list = list()

    """need to account for if trial groups exist"""
    for stim in responsive_neurons.keys():
        resp_stim = responsive_neurons[stim]
        if "sustained" in resp_stim:
            for sorter in resp_stim.keys():
                for idx in resp_stim[sorter]:
                    stim_list += [stim]
                    sorter_list += [sorter]
                    idx_list.append(cids[idx])
                    resp_list.append(["Resp"])
                    if run_qc:
                        qc_list.append(unit_quality[idx])
                    if isiv is not None:
                        viol_list.append(viol_percent[idx])
                    if run_sil:
                        sil_list.append(sil_quality[idx])
        else:
            for tg in resp_stim.keys():
                resp_stim_tg = resp_stim[tg]
                for sorter in resp_stim_tg.keys():
                    for idx in list(resp_stim_tg[sorter]):
                        stim_list += [stim]
                        sorter_list += [sorter.title()]
                        trialgroup_list.append(tg)
                        idx_list.append(cids[idx])
                        resp_list.append(["Resp"])
                        if run_qc:
                            qc_list.append(unit_quality[idx])
                        if isiv is not None:
                            viol_list.append(viol_percent[idx])
                        if run_sil:
                            sil_list.append(sil_quality[idx])

    neuron_idx_list = [neuron_idx] * len(stim_list)  # same id for number of neurons
    hash_idx_list = [
        hashlib.sha256((str(ids) + filename).encode()).hexdigest() for ids in idx_list
    ]

    if labels:  # if we have labels change the int trial group to the actual stim value
        for idx, value in enumerate(trialgroup_list):
            trialgroup_list[idx] = labels[stim_list[idx]][str(value)]

    resp_neuron_df = pd.DataFrame({})
    if len(trialgroup_list) != 0:
        resp_neuron_df["Trial Group"] = trialgroup_list

    resp_neuron_df["Stim"] = stim_list
    resp_neuron_df["Sorter"] = sorter_list
    resp_neuron_df["IDs"] = idx_list
    if run_qc:
        resp_neuron_df["QC"] = qc_list
    resp_neuron_df["File Hash"] = neuron_idx_list
    resp_neuron_df["HashID"] = hash_idx_list
    if isiv is not None:
        resp_neuron_df["ISI Violation Fraction"] = viol_list
    if run_sil:
        resp_neuron_df["Silhouette Score"] = sil_list
    """Following section makes the non-responsive neurons into a dataframe so we can
    analyze them separately"""
    non_resp = list()
    non_resp_qc = list()
    non_resp_sil = list()
    non_viol_list = list()
    for idx, cluster in enumerate(cids):
        if cluster not in idx_list:
            non_resp.append(cluster)
            if run_qc:
                non_resp_qc.append(unit_quality[idx])
            if run_sil:
                non_resp_sil.append(sil_quality[idx])
            if isiv is not None:
                non_viol_list.append(viol_percent[idx])

    non_resp_hash_idx = [
        hashlib.sha256((str(ids) + filename).encode()).hexdigest() for ids in non_resp
    ]
    non_neuron_idx_list = [neuron_idx] * len(non_resp_hash_idx)

    non_resp_df = pd.DataFrame(
        {
            "IDs": non_resp,
            "File Hash": non_neuron_idx_list,
            "HashID": non_resp_hash_idx,
        }
    )

    if run_qc:
        non_resp_df["QC"] = non_resp_qc
    if run_sil:
        non_resp_df["Silhouette Score"] = non_resp_sil
    if isiv is not None:
        non_resp_df["ISI Violation Fraction"] = non_viol_list

    # final quality run through if offered
    if qcthres and run_qc:
        resp_neuron_df = resp_neuron_df.loc[resp_neuron_df["QC"] >= qcthres]
        non_resp_df = non_resp_df.loc[non_resp_df["QC"] >= qcthres]
    if isi is not None:
        resp_neuron_df = resp_neuron_df.loc[
            resp_neuron_df["ISI Violation Fraction"] < isi
        ]
        non_resp_df = non_resp_df.loc[non_resp_df["ISI Violation Fraction"] > isi]
    if sil and run_sil:
        resp_neuron_df = resp_neuron_df.loc[resp_neuron_df["Silhouette Score"] > sil]
        non_resp_df = non_resp_df.loc[non_resp_df["Silhouette Score"] > sil]
    return resp_neuron_df, non_resp_df


def qc_only(
    qcvalues: dict, isiv: dict, sp: dict, qcthres: float, sil: float, isi: float
) -> tuple[dict, pd.DataFrame]:
    """
    function for creating a dataframe just based on quality metrics.

    Parameters
    ----------
    qcvalues : dict
        the isolation distances, contamination rates, and simplified silhouette scores
    isiv : dict
        dictionary of interspike interval violations.
    sp : dict
        spike train and associated data
    labels : dict
        labels to translate integers to strings for labels.
    qcthres : float
        isolation distance value cutoff for qc (0, inf)
    sil : float
        silhouette score value cutoff (-1, 1)
    isi : TYPE, optional
        Optional interspike violation fraction cutoff. (0,1) 0 with no violation The default is None.

    Returns
    -------
    sp : dict
        this is sp with only qc neurons loaded into sp['cids'] for ease of running methods
    quality_df : pd.DataFrame
        DataFrame of all neurons which passed qc. Non response data included.

    """
    filename = sp["filename"]
    neuron_idx = hashlib.sha256((filename).encode()).hexdigest()
    cids = sp["cids"]
    noise = sp["noise"]
    qc_list = qcvalues["uQ"]
    qc_list = qc_list[~noise]
    sil_list = qcvalues["sil"]
    sil_list = sil_list[~noise]

    if isiv is not None:
        isi_list = list()
        for key in isiv.keys():
            isi_list.append(isiv[key]["nViol"])

        isi_list = np.array(isi_list)
        # isi_list = isi_list[~noise]

    qc_threshold = np.where(qc_list > qcthres, True, False)
    sil_threshold = np.where(sil_list > sil, True, False)
    isi_threshold = np.where(np.array(isi_list) < isi, True, False)

    threshold = np.logical_and(qc_threshold, sil_threshold)
    threshold = np.logical_and(threshold, isi_threshold)

    if len(cids) != len(threshold):
        cids = np.array(cids)[~noise]
    final_cids = np.array(cids)[threshold]
    final_qc = np.array(qc_list[threshold])
    final_sil = np.array(sil_list[threshold])
    final_isi = np.array(np.array(isi_list)[threshold])
    if np.size(final_cids) != 1:
        filename_list = [neuron_idx] * len(final_cids)
        hash_idx = [
            hashlib.sha256((str(ids) + filename).encode()).hexdigest()
            for ids in final_cids
        ]
    else:
        filename_list = [neuron_idx]
        hash_idx = [hashlib.sha256((str(final_cids) + filename).encode()).hexdigest()]

    quality_df = pd.DataFrame(
        {
            "IDs": final_cids,
            "QC": final_qc,
            "Silhouette Score": final_sil,
            "ISI Violation Fraction": final_isi,
            "File Hash": filename_list,
            "HashID": hash_idx,
        }
    )

    sp["cids"] = final_cids

    return sp, quality_df


def waveform_vals_DF(
    wf: dict,
    sp: dict,
    waveform_dur: np.array,
    waveform_depth: np.array,
    waveform_amps: np.array,
    shank_dict: Optional[dict],
) -> pd.DataFrame:
    """
    Creates a pd.DataFrame of waveform values (depth,amps, duration, shank location)

    Parameters
    ----------
    wf : dict
        raw waveform data
    sp : dict
        spike train data
    waveform_dur : np.array
        array of waveform durations
    waveform_depth : np.array
        array of waveform depths
    waveform_amps : np.array
        array of waveform amps
    shank_dict : Optional[dict]
        dictionary of shank values; limited functionality should avoid for now

    Returns
    -------
    neuron_characteristics : pd.DataFrame
        DataFrame of waveform characterics for each cluster

    """
    cluster_ids = wf["F"]["ClusterIDs"]
    filename = sp["filename"]
    ids_hash = [
        hashlib.sha256((str(cluster_id) + filename).encode()).hexdigest()
        for cluster_id in cluster_ids
    ]

    if shank_dict is None:
        laterality_list = [np.nan] * len(ids_hash)
    else:
        laterality_list = shank_dict["med_lat"]

    neuron_characteristics = pd.DataFrame(
        {
            "HashID": ids_hash,
            "Waveform Duration (s)": waveform_dur,
            "Waveform Depth (um)": waveform_depth,
            "Waveform Amplitude (uV)": waveform_amps,
            "Laterality": laterality_list,
        }
    )

    return neuron_characteristics


""" It loads sp['cids'] to only look at the desired neurons"""


def gen_resp(resp_neuron_df: pd.DataFrame, sp: dict) -> dict:
    current_cids = resp_neuron_df["IDs"].unique()
    sp["cids"] = current_cids

    return sp
