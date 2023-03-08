#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:20:34 2023

@author: zacharymckenzie

SPAnalysis class is a quick way to examine spike data from a phy curated recording.
It will load the spike property dictionary automatically. using the `loadsp` function.
Once this is done both qc values and waveforms can be generated. These will be saved
and this can be used for the full ClusterAnalysis class.

ATTRIBUTES:
    sp: dict All spike properties
    qc: dict of isolation values
    wf: dict of raw waveform data
    isiv: dict of isi violations
    
METHODS:
    loadsp: loads an sp generated from kilosort/phy output files
    get_waveforms: collects raw electrical data from bin file
    qcfn: generates the isolation distance and isi violations
    plot_wfs: plots the raw waveforms
    acg: plots the autocorrelograms
    plot_pc: plots the top two pc spaces
    plot_drift: plots the drift map to look for instances of drift in recording
    plot_cdf: plots a pdf and cdf of spike info
"""

from .analysis.spsetup import loadsp

from .qcfn.qcfns import maskedClusterQuality
from .qcfn.isiVqc import isiV
from .misc_helpers.getWaveForms import getWaveForms

from .visualization_ca.plotWaveforms import plotWaveforms
from .visualization_ca.acg import plotACGs
from .visualization_ca.plottingPCs import plotPCs
from .visualization_ca.plotCDFs import makeCDF, getTempPos
from .visualization_ca.detectdrift import plotDriftmap


class SPAnalysis:
    def __init__(self, *args):
        if args:
            self.filepath = args[0]
        self.sp = dict()
        self.qc = dict()
        self.wf = dict()
        self.isiv = dict()

    def __repr__(self):
        print(f"File being analyzed is {self.sp['filename']}")
        print(f"Attributes are {vars(self)}")

    def loadsp(self) -> dict:
        sp = loadsp()
        self.sp = sp
        return sp

    def qcfn(self, isi=0.0005, ref_dur=0.002) -> tuple[dict, dict]:
        qcvalues, _, _ = maskedClusterQuality(self.sp)
        self.qc = qcvalues
        isiv = isiV(self.sp, isi=isi, ref_dur=ref_dur)
        self.isiv = isiv
        return qcvalues, isiv

    def get_waveforms(self, num_chans: int) -> dict:
        wf = getWaveForms(self.sp, nCh=num_chans)
        self.wf = wf
        return wf

    def plot_wfs(self, ind: bool) -> None:
        plotWaveforms(self.wf, Ind=ind)

    def acg(self, ref_dur: float) -> None:
        plotACGs(self.sp, refract_time=ref_dur)

    def plot_pc(self) -> None:
        plotPCs(self.sp)

    def plot_drift(self) -> None:
        spike_depths, spike_amps, _ = getTempPos(self.sp)
        plotDriftmap(self.sp["spikeTimes"], spike_amps, spike_depths)

    def plot_cdf(self) -> None:
        makeCDF(self.sp)
