# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 08:34:43 2023

@author: ZacharyMcKenzie

ClusterAnalysis is an oop class for analyzing neural data. It initialzed by load a spike
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

import numpy as np
import pandas as pd

# general functions--glue for the class
from misc.genhelpers import getFiles, loadPreviousAnalysis, getdir
from misc.cap_conversion import cap_conversion
from misc.label_generator import (
    labelGenerator,
    responseDF,
    genResp,
    qc_only,
    waveform_vals_DF,
)

# functions which calculate various metrics for the neural data
import analysis.psthfunctions as psthfn
from misc.getWaveForms import getWaveForms, getWaveFormVals
from qcfn.qcfns import maskedClusterQuality
from qcfn.isiVqc import isiV
from analysis.clusterzscore import clusterzscore
from analysis.firingratedf import firingRateWin
from analysis.prevalence_calculator import prevalence_calculator
from analysis.latency_calculator import latency_calculator

# plotting functions
from visualization_ca.psthviewer import plotPSTH
from visualization_ca.acg import plotACGs
from visualization_ca.plottingPCs import plotPCs
from visualization_ca.plotFiringRate import plotFiringRate
from visualization_ca.plotWaveforms import plotWaveforms
from visualization_ca.plotZscores import plotZscores
from visualization_ca.plotCDFs import makeCDF, getTempPos, plotDepthSpike
from visualization_ca.neuronprevalence import plotmedLat
from visualization_ca.detectdrift import plotDriftmap
from visualization_ca.neurocorrs import neuronCorr


class ClusterAnalysis:
    def __init__(self, sp: dict, eventTimes: dict):
        self.sp: dict = copy.deepcopy(sp)  # dict is mutable, so deep copy
        self.clu: np.array = sp["clu"].copy()
        self.clusterIDs: list = list(sp["cids"]).copy()
        self.spikeTimes = sp["spikeTimes"].copy()
        self.eventTimes: dict = copy.deepcopy(eventTimes)  # need to prevent overwriting
        self.filename: str = sp["filename"]
        self.allP = None
        self.zwindow = None
        self.normVal = None
        self.depth = None
        self.laterality = None
        self.resp_neuro_df = None
        self.non_resp_df = None

    """The repr will just print out the filename being analyzed--repr instead of str"""

    def __repr__(self):
        var_methods = dir(self)
        var = list(vars(self).keys())  # get our currents variables
        methods = list(set(var_methods) - set(var))
        final_methods = [method for method in methods if "__" not in method]
        return f"This is the analysis of {self.filename}.\n\nThe initialized variables are {var}\n\n The methods are {final_methods}"

    """I have the code cache wf and qcvalues. These are computationally 
    expensive calculations (RAM and CPU limited). So once we get these, they 
    shouldn't really change so it is better to load the files rather than 
    recalculate. I use a named tuple so if I decide to cache other files I can 
    easily grab those be modifying the underlying getFiles functions."""

    def get_files(self, title="") -> None:
        metrics = getFiles(self.filename)
        if metrics.firingrate:
            self.firingrate = metrics.firingrate
        if metrics.wf:
            self.wf: dict = metrics.wf
        if metrics.qcvalues:
            self.qc: dict = metrics.qcvalues
        if metrics.labels:
            self.labels: dict = metrics.labels
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
                if clustermetrics.resp_neuro_df:
                    self.resp_neuro_df: pd.DataFrame = clustermetrics.resp_neuro_df
                if clustermetrics.non_resp_df:
                    self.non_resp_df: pd.DataFrame = clustermetrics.non_resp_df
            else:
                print("error loading previous analysis")
        except:
            print("No previous ClusterAnalysis to pull values from")

    """labels is dict with format {str(int): "what you want"}. For baro it 
    works automatically based on 20mmHg/V. For other stimuli it must be entered
    by hand.
    It also allows to load laterality 'r' or 'l' for multi shank probes
    Finally it also load in depth of the bottom of the probe"""

    def set_labels(self, labels: dict, depth=None, laterality=None) -> None:
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

    """Function for converting event times based on the infusion of capsaicin
    requires time point"""

    def cap_conversion(self, time_pt: float) -> None:
        eventTimes, labels = cap_conversion(self.eventTimes, time_pt, self.sp)
        self.eventTimes = eventTimes
        self.labels = labels

    """This save allows us to keep track of the responsive_neurons by saving a 
    copy of a previous ClusterAnalysis Class. I'm working on making this save 
    better using dataframes to store values. May deprecate this along with 
    loadPrev()"""

    def save_analysis(self, note=None, title="") -> None:
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

    """plots firingrates as either lineplots or violinplots"""

    def firingratedf(self, window_dict=None, time_bin_size=0.05) -> None:
        firing_rate_df = firingRateWin(
            self.sp, self.eventTimes, window_dict, timeBinSize=time_bin_size
        )
        self.firing_rate_df = firing_rate_df

    def plot_firingrate(
        self,
        graph="v",
        labels=None,
    ) -> None:
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

    """getWaveForms will return the true waveforms (post phy curation) of, 
    which can then be analyzed for various metrics. This is RAM-limited since 
    it needs to load the whole binary file into memory. Please look at your RAM
    amount and if binary bile of raw waveforms is > 60% of your RAM this 
    function will likely fail to due to memeory limitations I have added some 
    protections, but may still fail."""

    def get_waveforms(self, num_chans: int) -> None:
        wf = getWaveForms(self.sp, nCh=num_chans)
        self.wf = wf

    """ This will collect true waveform values, including the duration of the 
    action potential in samples and seconds, the depth as averaged by waveform
    amplitude (by pc feature space is another way to average, which I have not
    implemented) and the amplitudes"""

    def waveform_vals(self) -> None:
        self.max_waveform = None
        self.waveform_dur = None
        self.waveform_depth = None
        self.waveform_amps = None
        self.shank_dict = None
        depth = self.depth
        laterality = self.laterality
        wf_vals = getWaveFormVals(
            self.wf, self.sp, dataOrder="F", depth=depth, laterality=laterality
        )
        self.max_waveform = wf_vals.max_waveforms
        self.waveform_dur = wf_vals.waveform_dur
        self.waveform_depth = wf_vals.final_depth
        self.waveform_amps = wf_vals.waveform_amps
        if wf_vals.shank_dict:
            self.shank_dict = wf_vals.shank_dict

    """`plot_wfs` will plot the raw waveforms. If `Ind` = `True` it will 
    display ~500 waveforms with the mean waveform in the middle. 
    If `Ind` is `False` it will only display the mean waveform."""

    def plot_wfs(self, ind=True) -> None:
        plotWaveforms(self.wf, order="F", Ind=True)

    def gen_wfdf(self):
        waveform_df = waveform_vals_DF(
            self.wf,
            self.sp,
            self.waveform_dur,
            self.waveform_depth,
            self.waveform_amps,
            self.shank_dict,
        )
        self.waveform_df = waveform_df

    """qcfn will run the isolation distance of clusters. It will also run the 
    interspike interval violation caluclation based on Dan Hill's paper. `isi` 
    is the minimal interspike interval as suggested. the `ref_dur` is your 
    hypothesized refractory period for your neurons currently set to 1.5 ms, 
    but can be changed depending on neural population"""

    def qcfn(self, isi=0.0005, ref_dur=0.0015) -> tuple[dict, dict]:
        unitQuality, contaminationRate, qcValues = maskedClusterQuality(self.sp)
        self.qc = qcValues
        isiv = isiV(self.sp, isi=isi, ref_dur=ref_dur)

        self.isiv = isiv
        return qcValues, isiv

    """cluZscore will calculate the zscored firing rates of neurons. There are
    two possible paramaters. timeBinSize which defaults to 10 ms, but can 
    be changed to 50 or 100 ms if desired. `tg` is the parameter for whether 
    to separate data by trial grouping. `False` means no sepearation by tg
    whereas `True` indicates to keep data separated. I added a window command
    which gives the user the option to enter a baseline window and a stimulus
    window if they use the same windows for their stimuli. Format is
    [[bslStart:float, bslStop:float], [stimStart:float, stimStop:float]]"""

    def clu_zscore(
        self, time_bin_size=0.05, tg=True, window=None
    ) -> tuple[dict, dict, list]:
        allP, normVal, window = clusterzscore(
            self.sp, self.eventTimes, time_bin_size, tg, window_list=window
        )
        self.allP: dict = allP
        self.time_bin: float = time_bin_size
        self.normVal: dict = normVal
        self.zwindow: list = window
        return allP, normVal, window

    """plot_z plots of Z score data from clu_zscore. `tg` is the trial group
    flag. It automatically take time_bin_size self. Labels are what word
    should be used for graphing the trial groups. If you haven't set labels or 
    want to overide then labels should be a dict with the following format 
    labels = {'value found in eventTimes': 'value you want to 
                               be displayed as str'}"""

    def plot_z(
        self, labels=None, tg=True, sorter_dict=None, time_point=0, plot=True
    ) -> None:
        if not labels:
            try:
                labels = self.labels
            except AttributeError:
                print("Enter a labels dict or add a temp labels dict to label graphs")

        time_bin_size = self.time_bin
        responsive_neurons, raw_responsive_neurons = plotZscores(
            self.allP,
            self.normVal,
            self.eventTimes,
            self.zwindow,
            timeBinSize=time_bin_size,
            tg=tg,
            labels=labels,
            sorter_dict=sorter_dict,
            time_point=time_point,
            plot=plot,
        )
        self.responsive_neurons = responsive_neurons
        self.raw_responsive_neurons = raw_responsive_neurons

    """spike_raster calculates psthvalues which can be used to create firing 
    rate and raster plots"""

    def spike_raster(self, time_bin_size=0.05) -> tuple[dict, list]:
        psthvalues, window = psthfn.rasterPSTH(self.sp, self.eventTimes, time_bin_size)
        self.psthvalues: dict = psthvalues
        self.time_bin: float = time_bin_size
        self.raster_window: list = window
        return psthvalues, window

    """plot_spikes will generate a 2 figure plot with firing rate on top and 
    raster on the bottom. `tg` is the trial group flag plots with `True` for
    separateing by trial groups. `eb` is the flag for including error shading
    for firing rate plot."""

    def plot_spikes(self, tg=True, eb=True) -> None:
        psthvalues = self.psthvalues
        eventTimes = self.eventTimes
        try:
            labels = self.labels
        except AttributeError:
            labels = None
        plotPSTH(
            psthvalues,
            eventTimes,
            labels=labels,
            groupSep=tg,
            eb=eb,
            raster_window=self.raster_window,
        )

    """This function plots ACG for each cluster. `ref_per` is the refractory
    period you want displayed on each graph. Default is 2ms"""

    def acg(self, ref_per=0.002) -> None:
        plotACGs(self.sp, refract_time=ref_per)

    """plot_pc plots only based on top two PCs. It will check for four pcs if 
    they exist and then after some math--see function for what is  happening--
    returns top two. Red is this cluster and black is spikes from other 
    clusters."""

    def plot_pc(self) -> None:
        plotPCs(self.sp, nPCsPerChan=4, nPCchans=15)

    """This function takes in sp and eventTimes along with a datatype flag. The
    options for this flag are to be `frraw` which indicates the raw firing rate
    calculated from the psth functions. The `frbsl` flag takes a baseline as 
    request from the user within the function. Finally the `frsm` flag will 
    perform a smoothing operation again the smoothing factor will be requested 
    from the user during the function call. `time_bin_size` allows for setting
    time scale. `tg` flag for trial groups and labels is the normal label dict.
    """

    def neuro_corr(
        self, datatype="frraw", time_bin_size=0.05, tg=False, labels=None
    ) -> None:
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

    def latency(self, time_bin_size: float, bsl_win: list, event_win: list) -> dict:
        latency_values = latency_calculator(
            self.sp,
            self.eventTimes,
            timeBinSize=time_bin_size,
            bsl_win=bsl_win,
            event_win=event_win,
        )
        self.latency = latency_values

    """plots cdf and pdf of spike depth by spike amplitude by firing rate. 
    Requires depth of neurons. `unit_only` will label only `good units`. 
    `laterality` is for multishank probes"""

    def plot_cdf(self, unit_only=False, laterality=False) -> None:
        depth = self.depth
        sp = self.sp
        makeCDF(sp, depth, units_only=unit_only, laterality=laterality)

    """Creates a drift map which marks out spikes of drift in red. Grayscale 
    indicates the amplitude of spikes."""

    def plot_drift(self) -> None:
        depth = self.depth
        spike_depths, spike_amps, _ = getTempPos(self.sp, depth)
        spike_times = self.spikeTimes
        plotDriftmap(spike_times, spike_amps, spike_depths)

    """plot_prevalence returns number of lateral and medial neurons for multi-
    shank recordings"""

    def plot_medlat_prevalence(self) -> None:
        med_neuron, lat_neuron = plotmedLat(self.sp, self.wf, self.shank_dict)
        self.med_count = med_neuron
        self.lat_count = lat_neuron

    def plot_depth_scatter(self, mark_units=False) -> None:
        wf = self.wf
        sp = self.sp
        waveform_depths = self.waveform_depth
        plotDepthSpike(sp, wf, waveform_depths, units_marked=mark_units)

    """takes the responsive_neurons dictionary from plot_z and converts to a 
    pd.DataFrame"""

    def gen_respdf(self, qcthres: float, isi=None) -> None:
        try:
            isiv = self.isiV
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
            isi=isi,
        )
        self.resp_neuro_df = resp_neurons_df
        self.non_resp_df = non_resp_df

    """prevalence_calculator generates the counts for responsive neuron subtypes. 
    Currently it just displays in terminal, but in the future I may store in a structure
    """

    def prevalence_calculator(self) -> None:
        prevalence_calculator(self.resp_neuro_df)

    """genResp loads sp['cids'] with only responsive neurons that passed qc. This allows
    for subanalysis of neurons. Can be reverted with `revertClu`"""

    def gen_resp(self) -> None:
        sp = genResp(self.resp_neuro_df, self.sp)
        self.sp = sp

    """I have a separate function that does this better, but if just a qc threshold is
    desired without thinking about use of responsivity this can be used instead"""

    def qc_only(self, qcthres: float) -> None:
        sp, quality_df = qc_only(self.qcvalues, self.sp, qcthres=qcthres)
        self.sp = sp
        self.quality_df = quality_df

    """This reverts back to the orginial cluster ids as well as the original
    clu."""

    def revert_cids(self) -> None:
        cids = self.clusterIDs
        clu = self.clu
        sp = self.sp
        sp["clu"] = clu
        sp["cids"] = cids
        self.sp = sp
