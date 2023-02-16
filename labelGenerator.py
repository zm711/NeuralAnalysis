# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:05:32 2023

@author: ZacharyMcKenzie
"""


import pandas as pd
import numpy as np


"""labelGenerator is a function which takes in the eventTimes dict and automatically
generates a labels dict based on the V labeled in the labels. In order to make sure
everything is an integer I divide by 0.25 in the stimulus_setupzm.py so I need to 
multiple by 0.25. Then we multiple by 20 since the barostat is 20mmHg/V."""


def labelGenerator(eventTimes: dict) -> dict:
    barostat = eventTimes["ADC1tot"]
    trial_groups = barostat["TrialGroup"]

    trials = set(trial_groups)  # will have repeats so take the set

    labels = dict()

    for trial in trials:
        labels[str(trial)] = str(trial * 0.25 * 20) + " mmHg"

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
    isi=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # cids = sp["cids"]
    try:
        unit_quality = qcvalues["uQ"]
        run_qc = True
    except TypeError:
        print("no qcvalues")
        run_qc = False
    filename = sp["filename"]
    neuron_idx = hash(filename)
    cids = sp["cids"]
    noise = sp["noise"]
    if len(cids) != len(unit_quality) and qcthres > 0:
        unit_quality = unit_quality[~noise]

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

    """need to account for if trial groups exist"""
    for stim in responsive_neurons.keys():
        resp_stim = responsive_neurons[stim]
        if "event" in resp_stim:
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

    neuron_idx_list = [neuron_idx] * len(stim_list)  # same id for number of neurons
    hash_idx_list = [hash(str(ids) + filename) for ids in idx_list]

    if labels:  # if we have labels change the int trial group to the actual stim value
        for idx, value in enumerate(trialgroup_list):
            trialgroup_list[idx] = labels[str(value)]

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

    """Following section makes the non-responsive neurons into a dataframe so we can
    analyze them separately"""
    non_resp = list()
    non_resp_qc = list()
    for idx, cluster in enumerate(cids):
        if cluster not in idx_list:
            non_resp.append(cluster)
            non_resp_qc.append(unit_quality[idx])

    non_resp_hash_idx = [hash(str(ids) + filename) for ids in non_resp]
    non_neuron_idx_list = [neuron_idx] * len(non_resp_hash_idx)

    non_resp_df = pd.DataFrame(
        {
            "IDs": non_resp,
            "QC": non_resp_qc,
            "File Hash": non_neuron_idx_list,
            "HashID": non_resp_hash_idx,
        }
    )

    # final quality run through if offered
    if qcthres and run_qc:
        resp_neuron_df = resp_neuron_df.loc[resp_neuron_df["QC"] >= qcthres]
        non_resp_df = non_resp_df.loc[non_resp_df["QC"] >= qcthres]

    if isi is not None:
        resp_neuron_df = resp_neuron_df.loc[resp_neuron_df["ISI Violations"] < isi]

    return resp_neuron_df, non_resp_df


def qc_only(
    qcvalues: dict,
    sp: dict,
    qcthres: float,
) -> tuple[dict, pd.DataFrame]:

    filename = sp["filename"]
    neuron_idx = hash(filename)
    cids = sp["cids"]
    noise = sp["noise"]
    qc_list = qcvalues["uQ"]
    qc_list = qc_list[~noise]

    threshold = np.squeeze(np.argwhere(qc_list > qcthres))

    final_cids = cids[threshold]
    final_qc = qc_list[threshold]
    filename_list = [neuron_idx] * len(final_cids)
    hash_idx = [hash(str(ids) + filename) for ids in final_cids]

    quality_df = pd.DataFrame(
        {
            "IDs": final_cids,
            "QC": final_qc,
            "File Hash": filename_list,
            "Hash ID": hash_idx,
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
    shank_dict: dict,
) -> pd.DataFrame:

    cluster_ids = wf["F"]["ClusterIDs"]
    filename = sp["filename"]
    ids_hash = [hash(str(cluster_id) + filename) for cluster_id in cluster_ids]

    if shank_dict is None:
        laterality_list = [np.nan] * len(ids_hash)
    else:
        laterality_list = shank_dict["med_lat"]

    neuron_characteristics = pd.DataFrame(
        {
            "HashID": ids_hash,
            "Waveform Duration (s)": waveform_dur,
            "Waveform Depth (um)": waveform_depth,
            "Waveform Amplitude (p or n Amps)": waveform_amps,
            "Laterality": laterality_list,
        }
    )

    return neuron_characteristics


def merge_df(*args: pd.DataFrame) -> pd.DataFrame:

    final_df = pd.DataFrame({})
    for idx in range(len(args)):
        if idx == 0:
            final_df = args[idx]
        else:
            final_df = final_df.merge(args[idx], left_on="HashID", right_on="HashID")

    return final_df


""" It loads sp['cids'] to only look at the desired neurons"""


def genResp(resp_neuron_df: pd.DataFrame, sp: dict) -> dict:

    current_cids = resp_neuron_df["IDs"].unique()
    sp["cids"] = current_cids

    return sp
