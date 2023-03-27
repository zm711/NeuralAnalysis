# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:34:49 2023

@author: ZacharyMcKenzie
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_trial_core(trial_corr: pd.DataFrame, cluster: str):
    mask = trial_corr == 1  # white out autocorrelations between fibers which are all 1
    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(
        data=trial_corr,
        mask=mask,
        cmap="viridis",
        vmin=0,
        cbar_kws={"label": "R Score"},
    )
    plt.title(f"{cluster}", weight="bold")

    plt.figure(dpi=1200)
    plt.show()


def plot_by_animal(final_dataframe: pd.DataFrame, stim: str):
    final_dataframe = final_dataframe.loc[final_dataframe["Stim"] == stim]
    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.stripplot(
        data=final_dataframe,
        x="Trial Group",
        y="R score",
        jitter=True,
        hue="Cluster",
        marker="*",
        palette="viridis",
    )
    sns.boxplot(data=final_dataframe, x="Trial Group", y="R score")
    sns.despine()
    plt.legend([], [], frameon=False)
    plt.figure(dpi=1200)
    plt.show()
