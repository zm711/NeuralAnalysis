# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 08:34:43 2023

@author: ZacharyMcKenzie

ClusterAnalysis is an oop class for analyzing neural data. It is initialized by loading a spike
property dictionary (sp) and a stimuli data dictionary (eventTimes). Use of loadsp will
automatically generate the necessary sp dictionary, but to do it by hand look for keys
in the ksanalysis file. eventTimes keys are also listed there. After intialization a 
variety of methods can be called. The core is clu_zscore which will generate z-scored 
firing rates and spike_raster which will generate raw firing rates. Additionally plot_z
gives the option for generating responsive neuron profiles based on z-score values.

the second half of the class is based on plotting functions to help with visualizing 
aspects of the data including heatmaps of z scores, smoothing firing rates and rasters
drift mapping, and waveform plots.
"""

import copy
import os.path
import os
from typing import Union, Optional

import numpy as np
import pandas as pd

# general functions--glue for the class
from .misc_helpers.genhelpers import getFiles, loadPreviousAnalysis, getdir
from .misc_helpers.dataframe_fns import merge_datasets, gen_zscore_df
from .misc_helpers.label_generator import (
    labelGenerator,
    responseDF,
    gen_resp,
    qc_only,
    waveform_vals_DF,
)

# functions which calculate various metrics for the neural data
from .analysis import psthfunctions as psthfn
from .misc_helpers.get_waveforms import get_waveforms, get_wf_values
from .qcfn.qcfns import masked_cluster_quality
from .qcfn.isi_viol import isi_viol
from .analysis.clusterzscore import clusterzscore
from .analysis.firingratedf import firingRateWin
from .analysis.prevalence_calculator import prevalence_calculator
from .analysis.latency_calculator import latency_calculator
from .analysis.trial_correlation import trial_corr

# plotting functions
from .visualization_ca.psthviewer import plot_psth
from .visualization_ca.acg import plot_acgs
from .visualization_ca.plotting_pcs import plot_pc
from .visualization_ca.plotFiringRate import plotFiringRate
from .visualization_ca.plot_waveforms import plot_waveforms
from .visualization_ca.plot_zscores import plot_zscores
from .visualization_ca.plot_cdf import make_cdf, get_temp_pos, plotDepthSpike
from .visualization_ca.neuronprevalence import plotmedLat
from .visualization_ca.detectdrift import plot_driftmap
from .visualization_ca.neurocorrs import neuronCorr
from .visualization_ca.plot_smfr import plot_smfr
from .visualization_ca.plot_raster import plot_raster


class ClusterAnalysis:
    """ClusterAnalysis is a class which takes in `sp` spike properties data as well as
    `eventTimes` the stimulus data. It deep copies these values so that there won't be
    any accidental mutation. This is for one Phy set of files. Parameters are given with
    reasonable defaults as well as some optional parameters to make faster reuse, but if
    these optional parameters are not used, the user will be prompted with request for
    the necessary values.

    Attributes"""

    def __init__(self, sp: dict, eventTimes: dict):
        self.sp: dict = copy.deepcopy(sp)  # dict is mutable, so deep copy
        self.clu: np.array = sp["clu"].copy()
        self._clusterIDs: list = list(sp["cids"]).copy()
        self.spikeTimes = sp["spikeTimes"].copy()
        self.eventTimes: dict = copy.deepcopy(eventTimes)  # need to prevent overwriting
        self.filename: str = sp["filename"]
        self.allP = None
        self.zwindow = None
        self.normVal = None
        self.depth = None
        self.laterality = None

    """The repr will just print out the filename being analyzed--repr instead of str"""

    def __repr__(self):
        var_methods = dir(self)
        var = list(vars(self).keys())  # get our currents variables
        methods = list(set(var_methods) - set(var))
        final_methods = [method for method in methods if "__" not in method]
        return f"This is the analysis of {self.filename}.\n\nThe initialized variables are {var}\n\n The methods are {final_methods}"

    def get_files(self, title="") -> None:
        """this grabs all cached files (e.g. wf and qcvalues). it will also load a
        previous analysis if given `title` and loads some metrics from that previous
        analysis."""

        metrics = getFiles(self.filename)
        if metrics.firingrate:
            self.firingrate = metrics.firingrate
        if metrics.wf:
            self.wf: dict = metrics.wf
        if metrics.qcvalues:
            self.qc: dict = metrics.qcvalues
        if metrics.labels:
            self.labels: dict = metrics.labels
        if metrics.isiv:
            self.isiv: dict = metrics.isiv
        try:
            clustermetrics = loadPreviousAnalysis(title=title)
            if type(clustermetrics) != str:
                if clustermetrics.responsive_neurons:
                    self.responsive_neurons: dict = clustermetrics.responsive_neurons
                if clustermetrics.note:
                    self.note: str = clustermetrics.note
                if clustermetrics.qcthres:
                    self.qcthres: float = clustermetrics.qcthres
                if clustermetrics.depth:
                    self.depth: float = clustermetrics.depth
                if clustermetrics.laterality:
                    self.laterality: str = clustermetrics.laterality
                if clustermetrics.labels:
                    self.labels: dict = clustermetrics.labels
                if clustermetrics.isiv:
                    self.isiv: dict = clustermetrics.isiv
                if clustermetrics.resp_neuro_df is not None:
                    self.resp_neuro_df: pd.DataFrame = clustermetrics.resp_neuro_df
                if clustermetrics.non_resp_df is not None:
                    self.non_resp_df: pd.DataFrame = clustermetrics.non_resp_df
            else:
                print("error loading previous analysis")
        except:
            print("No previous ClusterAnalysis to pull values from")

    def set_labels(
        self,
        labels: dict,
        depth: Optional[float] = None,
        laterality: Optional[str] = None,
    ) -> None:
        """
        this sets the labels for the integer stimuli for plotting, the depth of the
        probe used int the recording, and the laterality of the probe in the nervous
        system.

        Parameters
        ----------
        labels : dict
            Each stimulus should be a key with a dictionary of key:value pairs where the
            key is what is stored in trial groups and the value is the desired display
            value
        depth : float, optional
            A float depth of probe to be used for calculating waveform values.
            The default is None.
        laterality : str, optional
            Optional lateralization of the probe in the tissue. The default is None.

        Returns
        -------
        None
            stores values as labels, depth, laterality for use by other functions.

        """
        if self.eventTimes.get("ADC1", False) or self.eventTimes.get("ADC1tot", False):
            self.labels = labelGenerator(self.eventTimes)
        else:
            self.labels = labels
        current_depth = self.depth
        current_laterality = self.laterality
        if current_depth != depth and depth != None:
            print("Overwriting depth")
            self.depth = depth
        if current_laterality != laterality and laterality != None:
            print("Overwriting laterality")
            self.laterality = laterality
        filename = self.sp["filename"]
        np.save(filename + "labels.npy", labels, allow_pickle=True)

    def save_analysis(self, note=None, title="") -> None:
        """This save allows us to keep track of the responsive_neurons by saving a
        copy of a previous ClusterAnalysis Class. I'm working on making this save
        better using dataframes to store values. May deprecate this along with
        loadPrev()"""
        if note:
            self.note = note
        current_folder = os.path.abspath(os.getcwd())
        if self.sp["filename"] in current_folder:
            if os.path.basename(current_folder) == "pyanalysis":
                np.save(
                    self.sp["filename"] + title + "analysis.npy",
                    self,
                    allow_pickle=True,
                )
            else:
                os.chdir("pyanalysis")
                np.save(
                    self.sp["filename"] + title + "analysis.npy",
                    self,
                    allow_pickle=True,
                )
        else:
            (
                _,
                target_folder,
                _,
            ) = getdir()
            if os.path.basename(target_folder) != "pyanalysis":
                os.chdir("pyanalysis")
            np.save(
                self.sp["filename"] + title + "analysis.npy", self, allow_pickle=True
            )

    def firingratedf(self, window_dict=None, time_bin_size=0.05) -> None:
        """generates firing rate df based on the window given in `window_dict`
        based on the `time_bin_size` given. Default is 50 ms."""
        firing_rate_df = firingRateWin(
            self.sp, self.eventTimes, window_dict, timeBinSize=time_bin_size
        )
        self.firing_rate_df = firing_rate_df

    def plot_firingrate(
        self,
        graph="v",
        labels=None,
    ) -> None:
        """plots the firing rate df as either a violinplot `graph=v`
        or as a lineplot `graph=l`. `labels` allow for overriding
        internal labels"""
        if not labels:
            try:
                labels = self.labels
            except AttributeError:
                print("enter labels dict for appropriate labeling")
            try:
                response_df = self.resp_neuro_df
            except AttributeError:
                response_df = None
        plotFiringRate(
            self.firing_rate_df,
            graph=graph,
            response_df=response_df,
            labels=labels,
        )

    def get_waveforms(self, num_chans: int) -> None:
        """
        Obtains raw waveforms values from the binary file for each cluster stored
        internally in a dictionary

        Parameters
        ----------
        num_chans : int
            the number of channels in the recording to create the proper memory map

        Returns
        -------
        None
           Internally stores waveform data as a dictionary `wf`

        """
        wf = get_waveforms(self.sp, nCh=num_chans)
        self.wf = wf

    def waveform_vals(self) -> None:
        """This will collect true waveform values, including the duration of the
        action potential in samples and seconds, the depth as averaged by waveform
        amplitude (by pc feature space is another way to average, which I have not
        implemented) and the amplitudes"""
        self.shank_dict = None
        depth = self.depth
        laterality = self.laterality
        wf_vals = get_wf_values(
            self.wf, self.sp, dataOrder="F", depth=depth, laterality=laterality
        )
        self.max_waveform = wf_vals.max_waveforms
        self.waveform_dur = wf_vals.waveform_dur
        self.waveform_depth = wf_vals.final_depth
        self.waveform_amps = wf_vals.waveform_amps
        if wf_vals.shank_dict:
            self.shank_dict = wf_vals.shank_dict

    def plot_wfs(self, ind: bool = True) -> None:
        """
        plots the raw waveforms.

        Parameters
        ----------
        ind : bool, optional
            plots 300 waveforms of each cluster with the mean waveform
            in black. False will only display the mean waveform. The default is True.

        Returns
        -------
        None
            plots for each cluster in sp['cids']

        """
        plot_waveforms(self.wf, order="F", Ind=True)

    def gen_wfdf(self) -> None:
        """function to convert the waveform values as generated by
        `waveform_vals` into one dataframe with ids given with
        the "HashID" tag"""
        waveform_df = waveform_vals_DF(
            self.wf,
            self.sp,
            self.waveform_dur,
            self.waveform_depth,
            self.waveform_amps,
            self.shank_dict,
        )
        self.waveform_df = waveform_df

    def qcfn(self, isi=0.0005, ref_dur=0.0015) -> None:
        """qcfn will run the isolation distance of clusters (Harris 2001). It will
        also run the interspike interval violation caluclation based on Dan Hill's
        paper (2011). `isi` is the minimal interspike interval as suggested. The
        `ref_dur` is your hypothesized refractory period for your neurons currently
        set to 1.5 ms, but can be changed depending on neural population. qcValues
        also stores the simplified silhouette score (Huruschka 2004).

        Returns:
        qcValues:dict
        isiv: dict"""
        _, _, qcValues = masked_cluster_quality(self.sp)
        self.qc = qcValues
        isiv = isi_viol(self.sp, isi=isi, ref_dur=ref_dur)

        self.isiv = isiv

    def clu_zscore(
        self, time_bin_size=0.05, tg: bool = True, window=None
    ) -> tuple[dict, dict]:
        """
        creates z score values for each cluster for each stimulus based on baseline

        Parameters
        ----------
        time_bin_size : float, optional
            Size of time bin in seconds. The default is 0.05 (50 ms)
        tg : bool, optional
            Whether to separate by trial grouping for each stimulus The default is True.
        window : list[float], optional
            Can enter a list of lists for each stimulus with floats where each
            stimulus has a list of [start, end] for baseline and for event window. So,
            finally structure would be [[-1,-.1], [0, 4]]. The default is None.

        Returns
        -------
        allP: dict
            the score firing dictionary with each stimulus being a key and each value
            being an np.array of n_clusters x n_trial_groups x n_time_bins
        normVal: dict
            indicates baseline firing rates and std for each cluster

        """

        if type(time_bin_size) == float:
            time_bin_size = [time_bin_size]
        allP, normVal, window = clusterzscore(
            self.sp, self.eventTimes, time_bin_size, tg, window_list=window
        )
        self.allP: dict = allP
        self.time_bin: float = time_bin_size
        self.normVal: dict = normVal
        self.zwindow: list = window
        return allP, normVal

    def plot_z(self, labels=None, tg=True, plot=True) -> None:
        """plot_z plots of Z score data from clu_zscore. `tg` is the trial group
        flag. It automatically take time_bin_size self. Labels are what word
        should be used for graphing the trial groups. If you haven't set labels or
        want to overide then labels should be a nested dict"""
        if not labels:
            try:
                labels = self.labels
            except AttributeError:
                print("Enter a labels dict or add a temp labels dict to label graphs")

        time_bin_size = self.time_bin
        responsive_neurons, raw_responsive_neurons = plot_zscores(
            self.allP,
            self.normVal,
            self.eventTimes,
            self.zwindow,
            time_bin_list=time_bin_size,
            tg=tg,
            labels=labels,
            time_point=0,
            plot=plot,
        )
        self.responsive_neurons = responsive_neurons
        self.raw_responsive_neurons = raw_responsive_neurons

    def gen_zdf(self) -> None:
        """Will create a dataframe of zscored firing rates with HashID, and time bins
        numbered from 0 to n bins."""
        z_df = gen_zscore_df(self.sp, self.labels, self.allP)
        self.z_df = z_df

    def spike_raster(self, time_bin_size: float = 0.001, window_list=None) -> None:
        """
        Calculates raw firing rates based on time bins which can be consumed by other
        methods in the class

        Parameters
        ----------
        time_bin_size : TYPE, optional
            time bin size for raster plot calculation. The default is 0.001 (1ms)
        window_list : TYPE, optional
            A list of lists for windows for each stimulus. If not given
            the function will prompt the user. The default is None.

        Returns
        -------
        None
            Internally stores the psthvalues, time_bin, and raster_window for use with
            other functions.

        """
        if type(time_bin_size) == float:
            time_bin_size = [time_bin_size]
        psthvalues, window = psthfn.rasterPSTH(
            self.sp, self.eventTimes, time_bin_size, window_list
        )
        self.psthvalues: dict = psthvalues
        self.time_bin: float = time_bin_size
        self.raster_window: list = window

    def plot_spikes(
        self, labels: Optional[Union[dict, bool]] = None, tg=True, eb=True
    ) -> None:
        """
        function for plotting a smoothed firing rate and raster plot together. Will
        request gaussian smoothing filter value units of time_bin_size provided in
        `spike_raster`

        Parameters
        ----------
        labels : dict, optional
            optional overide of internal labels. None prompts to look
            for internal labels. False will cause it not to plot on the graph.
            The default is None.
        tg : bool
            Separates data by trial groups. The default is True.
        eb : bool
            Adds error bar shading to firing rate plot. The default is True.

        Returns
        -------
        None
            plots the firing rate and raster plot .

        """
        psthvalues = self.psthvalues
        eventTimes = self.eventTimes
        if labels is None:
            labels = self.labels
        plot_psth(
            psthvalues,
            eventTimes,
            labels=labels,
            groupSep=tg,
            eb=eb,
            raster_window=self.raster_window,
        )

    def plot_smfr(self, labels=None, tg=True, eb=True) -> None:
        """plots just the smoothed firing rate over each stimulus.
        Inputs:
            labels either a dict of labels, None to use internal labels or False to
        ignore. tg to separate by trial groups. eb for error bars"""
        psth = self.psthvalues
        events = self.eventTimes
        if labels is None:
            labels = self.labels
        plot_smfr(
            psth,
            events,
            labels=labels,
            groupSep=tg,
            eb=eb,
            raster_window=self.raster_window,
        )

    def plot_raster(self, tg=True) -> None:
        """plots a raster plot for each neuron for each stimulus.
        Inputs:
        labels either a dict of labels, None to use internal labels or False to
        ignore. tg indicates by trial group."""
        psth = self.psthvalues
        events = self.eventTimes
        plot_raster(
            psth,
            events,
            raster_window=self.raster_window,
            groupSep=tg,
        )

    def acg(self, ref_per: float = 0.002) -> None:
        """
        Creates autocorrelograms for each cluster.

        Parameters
        ----------
        ref_per : float, optional
            The refractory period of the neuronal population in seconds.
            The default is 0.002 (2ms).

        Returns
        -------
        None
            plots of ACGs.

        """
        plot_acgs(self.sp, refract_time=ref_per)

    def plot_pc(self) -> None:
        """
        plots the two highest value PCs to given a visualization of clustering quality.
        Better to use actual qc functions for qc cutoffs.

        Returns
        -------
        None
            plots 2d pc figures..

        """
        plot_pc(self.sp, nPCsPerChan=4, nPCchans=15)

    def neuro_corr(
        self, datatype="frraw", time_bin_size=0.05, tg=False, labels=None
    ) -> None:
        """This function takes in sp and eventTimes along with a datatype flag. The
        options for this flag are to be `frraw` which indicates the raw firing rate
        calculated from the psth functions. The `frbsl` flag takes a baseline as
        request from the user within the function. Finally the `frsm` flag will
        perform a smoothing operation again the Gaussian smoothing factor will be
        requested from the user during the function call. `time_bin_size` allows for
        setting time scale (50ms default). `tg` flag for trial groups and labels is the
        normal label dict.
        """
        if not labels:
            try:
                labels = self.labels
            except AttributeError:
                print("enter labels if you are planning on looking at trial groups")
        neuronCorr(
            self.sp,
            self.eventTimes,
            self.allP,
            self.normVal,
            self.zwindow,
            datatype=datatype,
            timeBinSize=time_bin_size,
            tg=tg,
            labels=labels,
        )

    def latency(
        self,
        time_bin_size: float,
        bsl_win: list[float],
        event_win: list[float],
        num_shuffles: int,
    ) -> None:
        """
        checks for the latency to fire after start of stimulus using two methods:
        chase et al for <2Hz neurons and Mormann et al for > 2Hz.

        Parameters
        ----------
        time_bin_size : float
            Size of the time bin to use given in seconds
        bsl_win : list
            baseline window to calculate baseline firing rate and to generate shuffles
        event_win : list
            window around stimulus to be analyzed with [start, end] format
        num_shuffles : int
            Generates random shuffles of baseline to allow for comparisons. This is the
            number of shuffles.

        Returns
        -------
        None
            stores to dictionarys, latency_vals which contains the latency firing rate
            for each cluster and the latency_shuffled for the shuffled baseline
            latencies.

        """
        latency_values, shuffled_values = latency_calculator(
            self.sp,
            self.eventTimes,
            timeBinSize=time_bin_size,
            bsl_win=bsl_win,
            event_win=event_win,
            num_shuffle=num_shuffles,
        )
        self.latency_vals = latency_values
        self.latency_shuffled = shuffled_values

    def trial_corrs(self, sm_params: Union[int, list]) -> None:
        trial_corr_df, _, _ = trial_corr(
            self.psthvalues, self.eventTimes, sm_param=sm_params
        )
        self.trial_corr_df = trial_corr_df

    def plot_cdf(self, unit_only=False, laterality=False) -> None:
        """plots cdf and pdf of spike depth by spike amplitude by firing rate.
        Requires depth of neurons. `unit_only` will label only `good units`.
        `laterality` is for multishank probes"""

        depth = self.depth
        sp = self.sp
        make_cdf(sp, depth, units_only=unit_only, laterality=laterality)

    def plot_drift(self) -> None:
        """Creates a drift map which marks out spikes of drift in red. Grayscale
        indicates the amplitude of spikes. Good for tracking drift, this does not
        do any drift corrections"""
        depth = self.depth
        spike_depths, spike_amps, _ = get_temp_pos(self.sp, depth)
        spike_times = self.spikeTimes
        plot_driftmap(spike_times, spike_amps, spike_depths)

    def plot_medlat_prevalence(self) -> None:
        """plot_prevalence returns number of lateral and medial neurons for multi-
        shank recordings"""
        med_neuron, lat_neuron = plotmedLat(self.sp, self.wf, self.shank_dict)
        self.med_count = med_neuron
        self.lat_count = lat_neuron

    def plot_depth_scatter(self, mark_units=False) -> None:
        """plots a scatter plot of the depth of neural units. If
        `mark_units` is `True` it will mark neurons labeled as
        'good' in red vs other neurons in black"""
        wf = self.wf
        sp = self.sp
        waveform_depths = self.waveform_depth
        plotDepthSpike(sp, wf, waveform_depths, units_marked=mark_units)

    def gen_respdf(self, qcthres: float, sil: float, isi=None) -> None:
        """takes the responsive_neurons dictionary from plot_z and converts to a
        pd.DataFrame. At the same time it can optionally take in a `qcthres` as
        a min isolation distance for cluster. Set to `0` to ignore as well as a
        raw refractory spike violation number as a float (ie 0.02 would be less
        2% violations). `isi=None` ignores violations."""
        try:
            isiv = self.isiv
        except AttributeError:
            isiv = None
            print("run qcfn to generate isiv if necessary")
        resp_neurons_df, non_resp_df = responseDF(
            self.responsive_neurons,
            isiv,
            self.qc,
            self.sp,
            self.labels,
            qcthres=qcthres,
            sil=sil,
            isi=isi,
        )
        self.resp_neuro_df = resp_neurons_df
        self.non_resp_df = non_resp_df

    def prevalence_calculator(self) -> None:
        """prevalence_calculator generates the counts for responsive neuron subtypes.
        Currently it just displays in terminal, but in the future I may store in a
        structure"""
        prevalence_calculator(self.resp_neuro_df)

    def gen_resp(self) -> None:
        """genResp loads sp['cids'] with only responsive neurons that passed qc.
        This allows for subanalysis of neurons. Can be reverted with `revertClu`"""
        sp = gen_resp(self.resp_neuro_df, self.sp)
        self.sp = sp

    def qc_only(self, qcthres: float, sil: float, isi: float) -> None:
        """This function ignores any responsiveness of neurons and instead only uses
        the isolation distance to mark units as high enough quality. `qcthres` is a
        float of the min isolation distance required. It automatically makes the
        dataframe of these units"""
        sp, quality_df = qc_only(
            self.qc, self.isiv, self.sp, qcthres=qcthres, sil=sil, isi=isi
        )
        self.sp = sp
        self.quality_df = quality_df

    def merge_dfs(self, dtype="resp"):
        """loads all dataframes together. requires specifying `resp` for fusing the
        responsive neuron dataframe or `qc` for just using qc cutoffs`. Requires that
        a waveform dataframe and the z_df to be generated before"""
        if dtype == "resp":
            df1 = self.resp_neuro_df
        elif dtype == "qc":
            df1 = self.quality_df
        wf_df = self.waveform_df
        z_df = self.z_df

        final_df = merge_datasets(self.sp, df1, wf_df, z_df, dtype=dtype)
        self.all_data_df = final_df

    def revert_cids(self) -> None:
        """After masking data based on unit quality or responsiveness this function will
        unmask the low quality or unresponsive data to go back to reanalyze all data"""
        cids = self._clusterIDs
        clu = self.clu
        sp = self.sp
        sp["clu"] = clu
        sp["cids"] = cids
        self.sp = sp
