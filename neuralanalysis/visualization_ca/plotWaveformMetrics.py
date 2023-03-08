#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:09:25 2022

@author: zacharymckenzie

Function for plotting the waveform metrics
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plotWaveformMetrics(metric, name, wf, depthval=None):

    clusterIDs = wf["F"]["ClusterIDs"]

    try:
        if np.shape(metric)[1] == 2:
            metric = np.squeeze(metric[:, 0])
            y_label = "Amplitude (mV/uV)"
        elif np.shape(metric)[1] == 3:
            metric = np.squeeze(metric[:, 2])
            y_label = "Time (mSec)"
    except IndexError:
        y_label = "Depth (um)"
        if depthval:
            metric = metric - depthval

    finally:

        plt.subplots(figsize=(10, 8))
        # ax.xaxis.label.set_size(14)
        # ax.yaxis.label.set_size(14)
        sns.barplot(x=clusterIDs, y=metric, color="k")
        plt.title(f"{name.title()}", weight="bold")
        plt.ylabel(y_label)
        plt.xlabel("Cluster Number")
        plt.figure(dpi=800)
        plt.show()
