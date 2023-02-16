# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:27:52 2022

@author: ZacharyMcKenzie

Quick function for comparing lateral vs medial neurons. Need to wait to have
more datasets. Then will be cool. Not stitched into my main pipeline yet.

"""

import numpy as np

import matplotlib.pyplot as plt


def plotmedLat(sp, wf, shank_dict):

    cids = sp["cids"]
    clusterIDs = wf["F"]["ClusterIDs"]

    desired_spikes = np.array([True if x in cids else False for x in clusterIDs])

    met_lat = np.array(shank_dict["med_lat"])
    final_met_lat = met_lat[desired_spikes].tolist()

    medial_neurons = final_met_lat.count("medial")
    lateral_neurons = final_met_lat.count("lateral")
    print(f"Medial neurons are {medial_neurons}")
    print(f"Lateral neurons are {lateral_neurons}")
    # new_cids = np.array(clusterIDs)[desired_spikes]

    plotmedlatCore(medial_neurons, lateral_neurons)
    return medial_neurons, lateral_neurons


def plotmedlatCore(medial_neurons: int, lateral_neurons: int):

    wedges = ["Medial", "Lateral"]
    counts = [medial_neurons, lateral_neurons]

    f, ax = plt.subplots(figsize=(10, 8))
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
    ax.pie(
        counts,
        labels=wedges,
        autopct=lambda pct: "{:.1f}%\n(n={:d})".format(
            pct, int(np.round(pct / 100 * np.sum(counts)))
        ),
        shadow=False,
        startangle=90,
        colors=colors,
    )
    ax.axis("equal")
    plt.tight_layout()
    plt.figure(dpi=1200)
    plt.show()
