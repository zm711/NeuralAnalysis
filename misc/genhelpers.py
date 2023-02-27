#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 08:40:44 2022

@author: zacharymckenzie

these are ease of life for getting a file directory specifically look for rhd files. It still works without
rhd files it just returns NONE, but returns the directory. Other functions include some easy to use/MATLAB
translations


"""

from tkinter import Tk, filedialog
import os
import glob
import numpy as np
from collections import namedtuple

"""This is how I like grabbing my directory for python stuff wrapped up nice and neat also puts
out the old directory for easy of moving back if one gets lost"""


def getdir() -> tuple[str, str, str]:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    old_dir = os.getcwd()
    open_file = filedialog.askdirectory()
    os.chdir(open_file)
    filenameRaw = glob.glob("*.rhd")
    if len(filenameRaw) > 0:
        filenamestr = filenameRaw[0]
        file_name = filenamestr[:-4]
    else:
        try:
            filenameRaw = glob.glob("*sp.npy")
            filenamestr = filenameRaw[0]

            file_name = filenamestr[:-6]
        except IndexError:
            file_name = None

    return old_dir, open_file, file_name


"""my implementation of the [row,col] = find(X) from matlab. makes it easier than running 
the extra lines of code"""


def findindex(matrix: np.array) -> tuple[np.array, np.array]:
    my_array = np.array(matrix)
    my_index = np.transpose(np.nonzero(my_array))
    row = my_index[:, 0]
    col = my_index[:, 1]
    return row, col


"""This gives a warning before overwritting a numpy file"""


def savefile(filename: str, file: str) -> None:
    if len(filename) > 90:
        filename = filename[:90]
        
    if os.path.isfile(filename):
        print("File already exists.")
        overWrite = input("Would you like to overwrite y/n?\n")

        while overWrite != "y" and overWrite != "n":
            overWrite = input("Would you like to overwrite y/n?\n")

    else:
        overWrite = "y"

    if overWrite == "y":
        np.save(filename, file, allow_pickle=True)
        print("File overwritten. Save complete")

    elif overWrite == "n":
        print("Saving process aborted")
    else:
        print("Error. Rerun")


"""This grabs any analyis values and loads to the local variable space for use
I've had to iterate this multipe times due to local vs global variable space issue
so currently this gets the 'results' dict from intan and returns specific data we use
it is easy to grab another value with the format below. Then it returns those values 
to the workspace."""


def loadvalues() -> namedtuple:

    filename = glob.glob("*intan.npy")
    IntanValues = namedtuple(
        "IntanValues",
        "board_adc_channels board_adc_data frequency_parameters board_dig_in_data board_dig_in_channels",
    )
    if len(filename) > 0:
        fileName = filename[0]
        results = np.load(fileName, allow_pickle=True)[()]
        board_adc_channels = results.get("board_adc_channels", None)
        board_adc_data = results.get("board_adc_data", None)
        frequency_parameters = results["frequency_parameters"]
        board_dig_in_data = results.get("board_dig_in_data", None)
        board_dig_in_channels = results.get("board_dig_in_channels", None)

        intanvals = IntanValues(
            board_adc_channels,
            board_adc_data,
            frequency_parameters,
            board_dig_in_data,
            board_dig_in_channels,
        )  # load potential values into a namedtuple to reduce risk in other functions
        return intanvals
    else:
        print("No intan data. Please run zmbin.py")
        fileName = filename[0]


"""This just gets out the cgs classifications from phy curation."""


def readCGSfile(cgsfile) -> tuple[list, list]:
    cids = list()
    cgs = list()
    for row in range(1, len(cgsfile)):
        cids.append(int(cgsfile[row].split()[0]))

        cgs.append(cgsfile[row].split()[1])
    for row in range(len(cgs)):
        if cgs[row] == "mua":
            cgs[row] = 1
        elif cgs[row] == "good":
            cgs[row] = 2
        elif cgs[row] == "unsorted":
            cgs[row] = 3
        else:
            cgs[row] = 0

    return cids, cgs


"""getFiles allows us to load the 'wf' data structure of the raw data of the 
spikes. It also looks for past firingrate data and qc data. If these exist it 
loads these files with their standard names. If they don't exist it will return
None to let you know that it did not open that subfile--ie it doesn't exist
in your current folder"""


def getFiles(filename: str) -> namedtuple:

    Filevals = namedtuple("Filevals", "wf firingrate qcvalues labels")
    _, filepath, _ = getdirzm()

    os.chdir(filepath)
    try:
        wf_file = glob.glob("*wf*.npy")[0]
    except IndexError:
        wf_file = False
    try:
        firingrate_file = glob.glob("*firingrate*.npy")[0]
    except IndexError:
        firingrate_file = False
    try:
        qc_file = glob.glob("*qc*.npy")[0]
    except IndexError:
        qc_file = False
    try:
        labels_file = glob.glob("*labels*.npy")[0]
    except IndexError:
        labels_file = False

    wf = None
    firingrate = None
    qcvalues = None
    labels = None

    if wf_file:
        wf = np.load(wf_file, allow_pickle=True)[()]

    if firingrate_file:
        firingrate = np.load(firingrate_file, allow_pickle=True)[()]

    if qc_file:
        qcvalues = np.load(qc_file, allow_pickle=True)[()]

    if labels_file:
        labels = np.load(labels_file, allow_pickle=True)[()]

    metrics = Filevals(wf, firingrate, qcvalues, labels)

    return metrics


"""loadPreviousAnalysis is a trick to allow me to take data from a previous
initialization of ClusterAnalysis and load it for the current analysis. Similar
to above returns the value if it exists otherwise returns none to let you know
it doesn't exist in the current folder"""


def loadPreviousAnalysis(title="") -> namedtuple:
    print(
        "Please select folder with previous analysis.\n If ClusterAnalysis save used it will be stored in pyanalysis"
    )
    _, filepath, _ = getdirzm()
    os.chdir(filepath)

    Value = namedtuple(
        "Value",
        "responsive_neurons note, qcthres depth laterality labels isiv resp_neuro_df non_resp_df",
    )
    if title:
        try:
            clusterAnalysis = glob.glob("*" + title + "analysis.npy")[0]
        except IndexError:
            return "no analysis with that subtitle"
    else:
        try:
            clusterAnalysis = glob.glob("*analysis.npy")[0]
        except IndexError:
            return "no previous"

    cluster = np.load(clusterAnalysis, allow_pickle=True)[()]

    try:
        responsive_neurons = cluster.responsive_neurons
    except AttributeError:
        responsive_neurons = None
    try:
        note = cluster.note
    except AttributeError:
        note = None
    try:
        qcthres = cluster.qcthres
    except AttributeError:
        qcthres = None
    try:
        depth = cluster.depth
    except AttributeError:
        depth = None
    try:
        laterality = cluster.laterality
    except AttributeError:
        laterality = None
    try:
        labels = cluster.labels
    except AttributeError:
        labels = None
    try:
        isiv = cluster.isiv
    except AttributeError:
        isiv = None
    try:
        resp_df = cluster.resp_neuro_df
    except AttributeError:
        resp_df = None
    try:
        non_resp_df = cluster.non_resp_df
    except AttributeError:
        non_resp_df = None

    metrics = Value(
        responsive_neurons,
        note,
        qcthres,
        depth,
        laterality,
        labels,
        isiv,
        resp_df,
        non_resp_df,
    )
    return metrics
