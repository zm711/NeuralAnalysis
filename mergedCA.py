# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 08:37:47 2023

@author: ZacharyMcKenzie

MergedCA is a class which allows for analysis of multiple datasets. 
"""

from ksanalysis import ClusterAnalysis
import psthfunctionszm as psthfn
from clusterzscorezm import clu_z_score_merged
from mergeclufunctions import merge_df

from psthviewer import plotPSTH


class MCA(ClusterAnalysis):
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

    def mSpiRas(self, timeBinSize=0.05, tg=True, ebval=True):
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
                sp_cur, eventTime_cur, timeBinSize=timeBinSize
            )

            plotPSTH(
                psthvalues,
                eventTime_cur,
                labels=label_cur,
                groupSep=tg,
                eb=ebval,
                raster_window=window,
            )

    def mACG(self, ref_period=0.002):
        sp_list = self.sp_list
        for idx in range(len(sp_list)):
            print(f"Analyising Data from {self.filename_list[idx]}")
            self.sp = sp_list[idx]

            super().ACG(self, ref_per=ref_period)

    def mZscore(
        self,
        window_list,
        time_bin_size=0.05,
        tg=True,
    ):
        if window_list is None:
            window_list = [[-30, 10], [-10, 30]]
        allP, normVal, hash_idx = clu_z_score_merged(
            self.sp_list,
            self.event_list,
            time_bin_size=time_bin_size,
            window_list=window_list,
            tg=tg,
        )
        self.allP = allP
        self.normVal = normVal
        self.timeBin = time_bin_size
        self.eventTimes = self.event_list[0]
        self.zwindow = [window_list[1]]

    def merge_datasets(self):
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

    def subClu(self):
        return "subClu not possible in merged dataset"

    def revertClu(self):
        return "revertClu not possible in merged dataset"

    def cluZscore(self):
        return "run mZscore instead for merged data"

    def spikeRaster(self):
        return "run mSpiRas instead for merged data"
