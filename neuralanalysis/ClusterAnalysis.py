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
from .misc_helpers.genhelpers import getFiles, loadPreviousAnalysis, getdir
from .misc_helpers.cap_conversion import cap_conversion
from .misc_helpers.label_generator import (
    labelGenerator,
    responseDF,
    genResp,
    qc_only,
    waveform_vals_DF,
)

# functions which calculate various metrics for the neural data
from .analysis import psthfunctions as psthfn
from .misc_helpers.getWaveForms import getWaveForms, getWaveFormVals
from .qcfn.qcfns import maskedClusterQuality
from .qcfn.isiVqc import isiV
from .analysis.clusterzscore import clusterzscore
from .analysis.firingratedf import firingRateWin
from .analysis.prevalence_calculator import prevalence_calculator
from .analysis.latency_calculator import latency_calculator

# plotting functions
from .visualization_ca.psthviewer import plotPSTH
from .visualization_ca.acg import plotACGs
from .visualization_ca.plottingPCs import plotPCs
from .visualization_ca.plotFiringRate import plotFiringRate
from .visualization_ca.plotWaveforms import plotWaveforms
from .visualization_ca.plotZscores import plotZscores
from .visualization_ca.plotCDFs import makeCDF, getTempPos, plotDepthSpike
from .visualization_ca.neuronprevalence import plotmedLat
from .visualization_ca.detectdrift import plotDriftmap
from .visualization_ca.neurocorrs import neuronCorr


class ClusterAnalysis:
    """ClusterAnalysis is a class which takes in `sp` spike properties data as well as
    `eventTimes` the stimulus data. It deep copies these values so that there won't be
    any accidental mutation. This is for one Phy set of files. Parameters are given with
    reasonable defaults as well as some optional parameters to make faster reuse, but if
    these optional parameters are not used, the user will be prompted with request for
    the necessary values."""

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

    def set_labels(self, labels: dict, depth=None, laterality=None) -> None:
        """labels is dict with format {'Stim': {str(float): "what you want"}}. For baro
        works automatically based on 20mmHg/V. For other stimuli it must be entered
        by hand.It also allows to load laterality 'r' or 'l' for multi shank probes
        Finally it also load in `depth` of the bottom of the probe"""
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
        """`get_waveforms` will return the true waveforms (post phy curation),
        which can then be analyzed for various metrics. This is RAM-limited since
        it needs to load the whole binary file into memory. Please look at your RAM
        amount and if binary bile of raw waveforms is > 60% of your RAM this
        function will likely fail due to memory limitations"""
        wf = getWaveForms(self.sp, nCh=num_chans)
        self.wf = wf

    def waveform_vals(self) -> None:
        """This will collect true waveform values, including the duration of the
        action potential in samples and seconds, the depth as averaged by waveform
        amplitude (by pc feature space is another way to average, which I have not
        implemented) and the amplitudes"""
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

    def plot_wfs(self, ind=True) -> None:
        """`plot_wfs` will plot the raw waveforms. If `ind` = `True` it will
        display ~500 waveforms with the mean waveform in the middle.
        If `ind` is `False` it will only display the mean waveform."""
        plotWaveforms(self.wf, order="F", Ind=True)

    def gen_wfdf(self):
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

    def qcfn(self, isi=0.0005, ref_dur=0.0015) -> tuple[dict, dict]:
        """qcfn will run the isolation distance of clusters. It will also run the
        interspike interval violation caluclation based on Dan Hill's paper. `isi`
        is the minimal interspike interval as suggested. the `ref_dur` is your
        hypothesized refractory period for your neurons currently set to 1.5 ms,
        but can be changed depending on neural population"""
        unitQuality, contaminationRate, qcValues = maskedClusterQuality(self.sp)
        self.qc = qcValues
        isiv = isiV(self.sp, isi=isi, ref_dur=ref_dur)

        self.isiv = isiv
        return qcValues, isiv

    def clu_zscore(
        self, time_bin_size=0.05, tg=True, window=None
    ) -> tuple[dict, dict, list]:
        """`clu_zscore` will calculate the zscored firing rates of neurons. There are
        two possible paramaters. `time_bin_size` which defaults to 50 ms, but can
        be changed to 10 or 100 ms if desired. `tg` is the parameter for whether
        to separate data by trial grouping. `False` means no sepearation by tg
        whereas `True` indicates to keep data separated. `window` is optional parameter
        which gives the user the option to enter a baseline window and a stimulus
        window if they use the same windows for their stimuli. Format is
        [[bslStart:float, bslStop:float], [stimStart:float, stimStop:float]]"""
        if type(time_bin_size) == float:
            time_bin_size = [time_bin_size]
        allP, normVal, window = clusterzscore(
            self.sp, self.eventTimes, time_bin_size, tg, window_list=window
        )
        self.allP: dict = allP
        self.time_bin: float = time_bin_size
        self.normVal: dict = normVal
        self.zwindow: list = window
        return allP, normVal, window

    def plot_z(self, labels=None, tg=True, time_point=0, plot=True) -> None:
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
        responsive_neurons, raw_responsive_neurons = plotZscores(
            self.allP,
            self.normVal,
            self.eventTimes,
            self.zwindow,
            time_bin_list=time_bin_size,
            tg=tg,
            labels=labels,
            time_point=time_point,
            plot=plot,
        )
        self.responsive_neurons = responsive_neurons
        self.raw_responsive_neurons = raw_responsive_neurons

    def spike_raster(self, time_bin_size=0.05, window_list=None) -> tuple[dict, list]:
        """spike_raster calculates psthvalues which can be used to create firing
        rate and raster plots. it takes in `time_bin_size` in seconds, ie the default
        is 50 ms, but if using for raster plot 1 ms (0.001) is much better because a 
        raster plot requires all bins to be 0 or 1."""
        if type(time_bin_size) == float:
            time_bin_size = [time_bin_size]
        psthvalues, window = psthfn.rasterPSTH(self.sp, self.eventTimes, time_bin_size,window_list)
        self.psthvalues: dict = psthvalues
        self.time_bin: float = time_bin_size
        self.raster_window: list = window
        return psthvalues, window

    def plot_spikes(self, labels=None, tg=True, eb=True) -> None:
        """plot_spikes will generate a 2 figure plot with firing rate on top and
        raster on the bottom. `labels` can either be None in which it will try to grab
        the internal labels from `set_labels` or it can be `False`, in which case it
        won't plot labels on the graph. `tg` is the trial group flag plots with `True`
        fo separating by trial groups. `eb` is the flag for including error shading
        for firing rate plot."""
        psthvalues = self.psthvalues
        eventTimes = self.eventTimes
        if labels is None:
            labels = self.labels
        plotPSTH(
            psthvalues,
            eventTimes,
            labels=labels,
            groupSep=tg,
            eb=eb,
            raster_window=self.raster_window,
        )

    def acg(self, ref_per=0.002) -> None:
        """This function plots ACG (autocorrelograms) for each cluster. `ref_per`
        is the refractory period to be displayed on each graph. Default is 2ms, but
        range could reasonably be 1-3 ms (0.001-0.003)"""
        plotACGs(self.sp, refract_time=ref_per)

    def plot_pc(self) -> None:
        """plot_pc plots only based on top two PCs. It will check for four pcs if
        they exist and then after some math--see function for what is happening--
        returns the top two. Red is this cluster and black is spikes from other
        clusters."""
        plotPCs(self.sp, nPCsPerChan=4, nPCchans=15)

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

    def latency(self, time_bin_size: float, bsl_win: list, event_win: list) -> dict:
        """calculates latency based on Chase 2007 and Mormann 2012. See function for
        full stats. Requires `time_bin_size` as time in seconds, `bsl_win` which is the
        window to look for the baseline in seconds [start, end], and an `event_win`
        which is the same, but for the stimulus time [start, end]"""
        latency_values = latency_calculator(
            self.sp,
            self.eventTimes,
            timeBinSize=time_bin_size,
            bsl_win=bsl_win,
            event_win=event_win,
        )
        self.latency_vals = latency_values

    def plot_cdf(self, unit_only=False, laterality=False) -> None:
        """plots cdf and pdf of spike depth by spike amplitude by firing rate.
        Requires depth of neurons. `unit_only` will label only `good units`.
        `laterality` is for multishank probes"""

        depth = self.depth
        sp = self.sp
        makeCDF(sp, depth, units_only=unit_only, laterality=laterality)

    def plot_drift(self) -> None:
        """Creates a drift map which marks out spikes of drift in red. Grayscale
        indicates the amplitude of spikes. Good for tracking drift, this does not
        do any drift corrections"""
        depth = self.depth
        spike_depths, spike_amps, _ = getTempPos(self.sp, depth)
        spike_times = self.spikeTimes
        plotDriftmap(spike_times, spike_amps, spike_depths)

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

    def gen_respdf(self, qcthres: float, isi=None) -> None:
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
            isi=isi,
        )
        self.resp_neuro_df = resp_neurons_df
        self.non_resp_df = non_resp_df


    def prevalence_calculator(self) -> None:
        """prevalence_calculator generates the counts for responsive neuron subtypes. 
        Currently it just displays in terminal, but in the future I may store in a 
        structure """
        prevalence_calculator(self.resp_neuro_df)

    

    def gen_resp(self) -> None:
        """genResp loads sp['cids'] with only responsive neurons that passed qc. 
        This allows for subanalysis of neurons. Can be reverted with `revertClu`"""
        sp = genResp(self.resp_neuro_df, self.sp)
        self.sp = sp

    def qc_only(self, qcthres: float) -> None:
        """This function ignores any responsiveness of neurons and instead only uses
        the isolation distance to mark units as high enough quality. `qcthres` is a 
        float of the min isolation distance required. It automatically makes the 
        dataframe of these units"""
        sp, quality_df = qc_only(self.qcvalues, self.sp, qcthres=qcthres)
        self.sp = sp
        self.quality_df = quality_df

    
    def revert_cids(self) -> None:
        """After masking data based on unit quality or responsiveness this function will
        unmask the low quality or unresponsive data to go back to reanalyze all data"""
        cids = self.clusterIDs
        clu = self.clu
        sp = self.sp
        sp["clu"] = clu
        sp["cids"] = cids
        self.sp = sp
