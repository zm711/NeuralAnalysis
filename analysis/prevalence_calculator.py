# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:06:01 2023

@author: ZacharyMcKenzie
"""


import numpy as np
import pandas as pd


def prevalence_calculator(resp_neuro_df: pd.DataFrame, *args) -> None:
    
    if args:
        non_resp_df = args[0]
        print(f"Number of neurons which passed qc checks, but which did not respond to stimuli is {len(non_resp_df['IDs'].unique())}\n")
    
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
        (resp_neuro_df["Sorter"] == "OnOff")
        | (resp_neuro_df["Sorter"] == "onoff")
        | (resp_neuro_df["Sorter"] == "Onset-Offset")
    ]["IDs"].unique()

    final_on_list = on_list[~np.isin(on_list, sus_list)]  # if sus not onset
    final_on_list = final_on_list[
        ~np.isin(final_on_list, onoff_list)
    ]  # if on/off not onset

    final_onoff_list = onoff_list[~np.isin(onoff_list, sus_list)]

    print("\nTotal Unique Neurons\n")
    print(len(resp_neuro_df["IDs"].unique()))
    print("\nRaw DATA\n")
    print(
        resp_neuro_df.drop_duplicates(subset=["IDs", "Sorter"], keep="first")[
            "Sorter"
        ].value_counts()
    )

    print("\nFinal Prevalence Data\n")
    print(f"Onset Neuron number is {len(final_on_list)}")
    print(f"Sustained Neuron number is {len(sus_list)}")
    print(
        f"Relief Neuron Number is {len(relief_list)}\n\t sus {len(relief_list[np.isin(relief_list, sus_list)])}\t on-off {len(relief_list[np.isin(relief_list, final_onoff_list)])}"
    )
    print(f"Inhib Neuron Number is {len(inhib_list)}")
    print(f"Onset-Offset Neuron Number is {len(final_onoff_list)}")
