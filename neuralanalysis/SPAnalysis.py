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

from .qcfn.qcfns import masked_cluster_quality
from .qcfn.isi_viol import isi_viol
from .misc_helpers.get_waveforms import get_waveforms

from .visualization_ca.plot_waveforms import plot_waveforms
from .visualization_ca.acg import plot_acgs
from .visualization_ca.plotting_pcs import plot_pc
from .visualization_ca.plot_cdf import make_cdf, get_temp_pos
from .visualization_ca.detectdrift import plot_driftmap


class SPAnalysis:
    """SPAnalysis is a class for assessing in vivo recording quality with some basic
    metrics. It can look at waveforms to assess proper extracellular potential shape. It
    can calculate the PC spaces. It can look at autocorrelograms to asses refractory
    period violations.And it can look for instance of drift. When initializing the class
    it can be given an optional filepath to the Phy root directory, but if this is not
    given it will request

    ### ATTRIBUTES ###
        sp: dict of spike properties
        qc: dict of quality metrics
        wf: dict of raw waeveform data
        isiv: dict of interspike interval violations

    ### METHODS ###
        loadsp: method for loading Phy data
        get_waveforms: method for getting raw waveform data
        plot_wfs: method for plotting raw waveforms
        qcfn: method for generating isolatin distance and isi violations
        acg: method for plotting autocorrelogram
        plot_pc: method for plotting top two PC spaces
        plot_drift: method for creating a drift plot
        plot_cdf: method for plotting pdf and cdf of spiking info
    """

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
        """this function will request a directory from the user which contains the
        standard `Phy` output numpy files (e.g. `spike_times.npy`). It will then load
        all of these values and return it to the workspace."""
        sp = loadsp()
        self.sp = sp
        return sp

    def qcfn(self, isi=0.0005, ref_dur=0.002) -> tuple[dict, dict]:
        """qcfn runs an isolation distance and refractory period violation calcs and
        returns those values. `isi` is the minimual interspike interval as limited by
        the sampling rate of the recording device. Change based on device's sample rate.
        ref_dur is the length of the refractory period of the neurons. Although 2ms is
        standard in neuroscience this is based on the neuron population being studied.
        """
        qcvalues, _, _ = masked_cluster_quality(self.sp)
        self.qc = qcvalues
        isiv = isi_viol(self.sp, isi=isi, ref_dur=ref_dur)
        self.isiv = isiv
        return qcvalues, isiv

    def get_waveforms(self, num_chans: int) -> dict:
        """function to generate raw waveforms from the data rather than just templates.
        `nCh` is the number of channels for making the memory map. So a 64 channel probe
        would put 64 in. It will request a directory if needed. Returns the wf data."""
        wf = get_waveforms(self.sp, nCh=num_chans)
        self.wf = wf
        return wf

    def plot_wfs(self, ind: bool) -> None:
        """plot the raw waveforms. `ind` is True if ~500 waveforms are desired with mean
        in the middle. If `ind` False it will only display the mean"""
        plot_waveforms(self.wf, Ind=ind)

    def acg(self, ref_dur: float) -> None:
        """Autocorrelogram plots. It will display `ref_dur` as red lines in the figure.
        This value indicates refractory period which should be somewhere in the range of
        1-3 ms (0.001-0.003)"""
        plot_acgs(self.sp, refract_time=ref_dur)

    def plot_pc(self) -> None:
        """Plots top 2 pc spaces to give a rough idea of cluster separation. Not perfect
        since these are many dimensional spaces, but is a good first pass"""
        plot_pc(self.sp)

    def plot_drift(self) -> None:
        """Plots and marks potential instance of drift within the recording"""
        spike_depths, spike_amps, _ = get_temp_pos(self.sp)
        plot_driftmap(self.sp["spikeTimes"], spike_amps, spike_depths)

    def plot_cdf(self) -> None:
        """plot_cdf creates a pdf and cdf-like figure with depth on the y axis and amps
        on the x axis. The colormap is based on the number of spikes occurring."""
        make_cdf(self.sp)
