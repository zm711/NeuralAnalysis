# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:07:25 2023

@author: ZacharyMcKenzie
"""

from psthviewer import plotPSTH
import copy
import numpy as np
from ksanalysis import ClusterAnalysis
import pandas as pd


def ck_sub_region(myNeuron: ClusterAnalysis):

    psthvalues, _ = myNeuron.spikeRaster(timeBinSize=0.01)
    eventTimes = copy.deepcopy(myNeuron.eventTimes)

    for region in range(3):
        region_lst = ["Back", "Neck", "Midline"]
        trial_group = np.array(
            [
                trial
                for trial in eventTimes["DIG6"]["TrialGroup"]
                if trial % 3 == region and trial != region
            ]
        )
        curr_trial = np.isin(eventTimes["DIG6"]["TrialGroup"], trial_group)
        curr_event = eventTimes["DIG6"]["EventTime"][curr_trial]
        curr_lens = eventTimes["DIG6"]["Lengths"][curr_trial]
        curr_trial = trial_group
        current_eventTimes = copy.deepcopy(eventTimes)
        current_eventTimes["DIG6"]["EventTime"] = curr_event
        current_eventTimes["DIG6"]["Lengths"] = curr_lens
        current_eventTimes["DIG6"]["TrialGroup"] = trial_group
        print(f"{region_lst[region]}")
        plotPSTH(
            psthvalues,
            eventTimes=current_eventTimes,
            labels=None,
            raster_window=[[-0.5, 1]],
            eb=True,
        )


def ck_prevalence(myNeuron: ClusterAnalysis, sorter: dict):

    try:
        _ = myNeuron.allP
    except AttributeError:
        (_, _, _,) = myNeuron.cluZscore(
            timeBinSize=0.01, tg=True, window=[[-1.8, -0.1], [-0.5, 1]]
        )
    try:
        _ = myNeuron.responsive_neurons
    except AttributeError:
        myNeuron.plotZ(None, tg=True, time_point=0.1, sorter_dict=sorter, plot=False)

    try:
        myNeuron.genRespDF(qcthres=10)
    except IndexError:
        if myNeuron.resp_neuro_df:
            print("already run qc. Can't run again. You're doing great CK!")
        else:
            myNeuron.qc = None
            myNeuron.genRespDF(qcthres=10)
            print("Generating Neurons without quality metrics")

    resp_neuro_df = myNeuron.resp_neuro_df
    myNeuron.genResp()
    sus_list = resp_neuro_df[
        (resp_neuro_df["Sorter"] == "Sustained")
        | (resp_neuro_df["Sorter"] == "sustained")
    ]["IDs"].unique()
    on_list = resp_neuro_df[
        (resp_neuro_df["Sorter"] == "Onset") | (resp_neuro_df["Sorter"] == "onset")
    ]["IDs"].unique()
    relief_list = resp_neuro_df[
        (resp_neuro_df["Sorter"] == "relief") | (resp_neuro_df["Sorter"] == "Relief")
    ]["IDs"].unique()
    inhib_list = resp_neuro_df[
        (resp_neuro_df["Sorter"] == "Inhib") | (resp_neuro_df["Sorter"] == "inhib")
    ]["IDs"].unique()
    
    onoff_list = resp_neuro_df[
        (resp_neuro_df["Sorter"] == "Onoff") | (resp_neuro_df["Sorter"] == "onoff")
    ]["IDs"].unique()
    
    final_on_list = on_list[~np.isin(on_list, sus_list)]
    final_on_list = final_on_list[~np.isin(final_on_list, onoff_list)]

    print(f"Onset Neuron number is {len(final_on_list)}")
    print(f"Sustained Neuron number is {len(sus_list)}")
    print(f"Relief Neuron Number is {len(relief_list)}")
    print(f"Inhib Neuron Number is {len(inhib_list)}")
    print(f"Onset-Offset Neuron Number is {len(onoff_list)}")
    print(
        resp_neuro_df.drop_duplicates(subset=["IDs", "Sorter"], keep="first")[
            "Sorter"
        ].value_counts()
    )
    return resp_neuro_df


def ck_tuning_curve(myNeuron: ClusterAnalysis):

    
    
    
    allP, _, _ = myNeuron.cluZscore(
        timeBinSize=0.01, tg=True, window=[[-1.8, -0.1], [-0.5, 1]]
    )

    try:
        allP_sub = allP["Airpuff"]
    except KeyError:
        allP_sub = allP["airpuff"]

    maxes = np.mean(allP_sub, axis=0)
    auc = np.sum(allP_sub, axis=2)

    total_means = np.max(maxes, axis=1)
    mean_auc = np.mean(auc, axis=0)

    neck = list()
    neck_auc = list()
    midline = list()
    midline_auc = list()
    back = list()
    back_auc =list()

    for trial in range(len(total_means)):
        if trial % 3 == 0:
            back.append(total_means[trial])
            back_auc.append(mean_auc[trial])
        elif trial % 3 == 1:
            midline.append(total_means[trial])
            midline_auc.append(mean_auc[trial])
        else:
            neck.append(total_means[trial])
            neck_auc.append(mean_auc[trial])

    tuning_curve = pd.DataFrame(
        {
            "Region": ["Neck"] * 7 + ["Midline"] * 7 + ["Back"] * 7,
            "Pressure (PSI)": [
                0,
                0.15,
                0.3,
                0.6,
                0.75,
                0.9,
                1.45,
            ]
            * 3,
            "Z score": neck + midline + back,
            "AUC": neck_auc + midline_auc+ back_auc,
        }
    )

    return tuning_curve
