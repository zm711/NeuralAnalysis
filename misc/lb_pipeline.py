# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 09:09:21 2023

@author: ZacharyMcKenzie
"""


from ksanalyis import loadKS, ClusterAnalysis
from misc.cap_conversion import cap_conversion


def lb_datapipeline(init: bool, cap: bool):

    if init:
        sp, eventTimes = loadKS()
        depth = float(input("Enter depth of recording.\n"))
        lat = input("Enter laterality ('l' or 'r') if multi-shanks Else type None.\n")
        if lat.title() == "None":
            lat = None
        if cap == True:
            time_pt = float(input("enter the time when capsaicin applied"))
            eventTimes, new_labels = cap_conversion(eventTimes, time_pt=time_pt)
            myNeuron = ClusterAnalysis(sp, eventTimes)

            myNeuron.setLabels(labels=new_labels, depth=depth, laterality=lat)
        else:
            myNeuron.setLabels(labels=None, depth=depth, laterality=lat)

        _, _ = myNeuron.qcfn()
        (_,) = myNeuron.getWaveForms()

    else:
        myNeuron = ClusterAnalysis(*loadKS())

        myNeuron.getFiles()

    myNeuron.cluZscore(timeBinSize=0.05, tg=True, window=[[-30, -10], [-10, 30]])
    myNeuron.spikeRaster()
    myNeuron.plotZ(tg=True, plot=False)

    return myNeuron
