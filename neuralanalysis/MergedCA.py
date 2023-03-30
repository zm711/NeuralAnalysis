# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 08:37:47 2023

@author: ZacharyMcKenzie

MergedCA is a class which allows for analysis of multiple datasets. 
"""

from .ClusterAnalysis import ClusterAnalysis
from .analysis import psthfunctions as psthfn
from .analysis.clusterzscore import clu_z_score_merged
from .misc_helpers.mergeclufunctions import merge_df

from .visualization_ca.psthviewer import plot_psth


class MCA(ClusterAnalysis):
    """MCA is for assessing similar data sets across recordings to ensure that the same
    function paramaters are used. It takes all the same methods and attributes as the
    base `ClusterAnalysis` class, but has its own wrappers for working dataset by
    dataset. It takes in an optionally number of `ClusterAnalysis` objects to be
    initialized"""

    def __init__(self, *args: ClusterAnalysis):
        self.sp_list = list()
        self.event_list = list()
        self.filename_list = list()
        self.wf_list = list()
        self.depth_list = list()
        self.lat_list = list()
        self.label_list = list()
        self.waveform_list = list()
        self.resp_list = list()

        for cluster in args:
            self.sp_list.append(cluster.sp)
            self.event_list.append(cluster.eventTimes)
            self.filename_list.append(cluster.sp["filename"])
            try:
                self.wf_list.append(cluster.wf)
            except AttributeError:
                pass
            try:
                self.depth_list.append(cluster.depth)
            except AttributeError:
                pass
            try:
                self.lat_list.append(cluster.laterality)
            except AttributeError:
                pass
            try:
                self.label_list.append(cluster.labels)
            except AttributeError:
                pass
            try:
                self.waveform_list.append(cluster.waveform_df)
            except AttributeError:
                pass
            try:
                self.resp_list.append(cluster.resp_neuro_df)
            except AttributeError:
                pass
            try:
                self.non_resp_list.append(cluster.non_resp_df)
            except AttributeError:
                pass

    def __repr__(self):
        var_methods = dir(self)
        var = list(vars(self).keys())
        methods = list(set(var_methods) - set(var))
        final_methods = [method for method in methods if "__" not in method]
        return f"This is the analysis of {self.filename_list}.\n\nThe initialized variables are {var}\n\n The methods are {final_methods}"

    def m_spike_raster(self, time_bin_size=0.05, tg=True, ebval=True):
        """same as the standard spike_raster, but on multiple datasets and then with
        final merging of datasets together. Requires a `time_bin_size` default 50 ms
        as well as `tg` boolean to indicate split by trial groupings and then `ebval`
        which will include error bar shading"""
        sp_list = self.sp_list
        event_list = self.event_list
        for idx in range(len(sp_list)):
            sp_cur = sp_list[idx]
            eventTime_cur = event_list[idx]
            try:
                label_cur = self.label_list[idx]
            except IndexError:
                label_cur = None
            psthvalues, window = psthfn.rasterPSTH(
                sp_cur, eventTime_cur, timeBinSize=time_bin_size
            )

            plot_psth(
                psthvalues,
                eventTime_cur,
                labels=label_cur,
                groupSep=tg,
                eb=ebval,
                raster_window=window,
            )

    def m_acg(self, ref_period=0.002):
        sp_list = self.sp_list
        for idx in range(len(sp_list)):
            print(f"Analyising Data from {self.filename_list[idx]}")
            self.sp = sp_list[idx]

            super().ACG(self, ref_per=ref_period)

    def m_zscore(
        self,
        window_list,
        time_bin_list=[0.05],
        tg=True,
    ) -> None:
        """runs `clu_zscore` on multiple datasets. It can accept a `window_list` to
        allow for faster calculations otherwise it will prompt the user for values.
        `time_bin_size` as usual is set to default of 50 ms. `tg` is whether to analyze
        by trial grouping"""
        if window_list is None:
            window_list = [[-30, 10], [-10, 30]]
        allP, normVal, hash_idx = clu_z_score_merged(
            self.sp_list,
            self.event_list,
            time_bin_list=time_bin_list,
            window_list=window_list,
            tg=tg,
            label_list=self.label_list,
        )
        self.allP = allP
        self.normVal = normVal
        self.timeBin = time_bin_list
        self.eventTimes = self.event_list[0]
        self.zwindow = [window_list[1]]

    def merge_datasets(self) -> None:
        """This function takes any of the waveform data, responsive neuron data or non
        responsive neuron data and combines them into a pandas data frame with each unit
        being given a unique identity"""
        if len(self.waveform_df) > 0:
            m_waveform_df = merge_df(*self.waveform_list)
            self.m_waveform_df = m_waveform_df
        if len(self.resp_list) > 0:
            m_resp_df = merge_df(*self.resp_list)
            self.m_resp_df = m_resp_df
        if len(self.non_resp_list) > 0:
            m_non_resp_df = merge_df(*self.non_resp_list)
            self.m_non_resp_df = m_non_resp_df

    def qcfn(self):
        return "qcfn not possible in merged dataset"
