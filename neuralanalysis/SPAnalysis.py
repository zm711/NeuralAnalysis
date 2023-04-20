#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:20:34 2023

@author: zacharymckenzie
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
    can calculate the PC spaces. It can look at autocorrelograms to assess refractory
    period violations. And it can look for instance of drift. When initializing the class
    it can be given an optional filepath to the Phy root directory, but if this is not
    given it will request

    ATTRIBUTES
    ------------
        sp: dict of spike properties
        qc: dict of quality metrics
        wf: dict of raw waeveform data
        isiv: dict of interspike interval violations

    METHODS
    ------
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
        """
        Function for loading neural data from Phy structure into dictionary

        Returns
        -------
        sp: dict
            dictionary of neural data

        """
        sp = loadsp()
        self.sp = sp
        return sp

    def qcfn(self, isi=0.0005, ref_dur=0.002) -> tuple[dict, dict]:
        """
        function for running qc metrics including isi violations, isolation distance
        and simplified silhouette score

        Parameters
        ----------
        isi : float, optional
            Recording device enforce minimimal interspike interval. The default is 0.0005.
        ref_dur : float, optional
            refractory period given in seconds. The default is 0.002 (2ms)

        Returns
        -------
        qcvalues: dict
            dictionary of isolation distance and silhouette score for each cluster
        isiv: dict
            dictionary of interspike period violations

        """
        qcvalues, _, _ = masked_cluster_quality(self.sp)
        self.qc = qcvalues
        isiv = isi_viol(self.sp, isi=isi, ref_dur=ref_dur)
        self.isiv = isiv
        return qcvalues, isiv

    def get_waveforms(self, num_chans: int) -> dict:
        """
        function to read raw waveforms from the binary file. If in correct folder will
        read, otherwise will create a gui file prompt.

        Parameters
        ----------
        num_chans : int
            Number of channels recorded from. This should be equal to the number of rows
            in the raw data matrix.

        Returns
        -------
        wf: dict
            Dictionary of raw wave form data

        """
        wf = get_waveforms(self.sp, nCh=num_chans)
        self.wf = wf
        return wf

    def plot_wfs(self, ind: bool) -> None:
        """
        plots raw waveforms to access shape of waveforms

        Parameters
        ----------
        ind : bool
            If true will plot 300 waveforms in gray with mean waveform in black.
            If false will only plot the mean waveform

        Returns
        -------
        plot of each cluster's waveforms

        """
        plot_waveforms(self.wf, Ind=ind)

    def acg(self, ref_dur: float) -> None:
        """
        function for computing and displaying autocorrelograms for each cluster

        Parameters
        ----------
        ref_dur : float
            refractory period which will be marked with red lines in graphs

        Returns
        -------
        plots of each cluster's acg with refractory period marked in red

        """
        plot_acgs(self.sp, refract_time=ref_dur)

    def plot_pc(self) -> None:
        """
        plots 2 pc spaces to give an approximation of clustering quality. Not rigorous,
        so for real quality metrics use the `qcfn` method.


        """
        plot_pc(self.sp)

    def plot_drift(self) -> None:
        """Plots and marks potential instance of drift within the recording"""
        spike_depths, spike_amps, _ = get_temp_pos(self.sp)
        plot_driftmap(self.sp["spikeTimes"], spike_amps, spike_depths)

    def plot_cdf(self) -> None:
        """plot_cdf creates a pdf and cdf-like figure with depth on the y axis and amps
        on the x axis. The colormap is based on the number of spikes occurring."""
        make_cdf(self.sp)
