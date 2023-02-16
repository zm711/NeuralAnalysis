# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:09:30 2022

@author: ZacharyMcKenzie
"""


import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plotLatency(latencyfire, labels, allV=False):
    latency_df = convertLatency(latencyfire)
        
    if allV:
        plotLatencyCoreAll(latency_df, labels)
    else:
        plotLatencyCore(latency_df, labels)
        

def convertLatency(latencyfire):

    cluster_list=list()
    stim_list = list()
    trial_list = list()
    minspikes_list = list()
    
    for cluster in latencyfire.keys():
        for stim in latencyfire[cluster].keys():
            
            lf_sub = latencyfire[cluster][stim]
            for trial in sorted(lf_sub.keys()):
                minspikes = lf_sub[trial]['MinSpikes']
                
                cluster_list += (len(minspikes) * [cluster])
                stim_list += (len(minspikes) * [stim])
                trial_list += (len(minspikes) * [trial])
                minspikes_list +=minspikes
                
                
            latency_df = pd.DataFrame({
                'Clusters': cluster_list,
                'Stim': stim_list,
                'Trial Group': trial_list,
                'Spike Latency (s)': minspikes_list})
            
        return latency_df
    
    
def plotLatencyCoreAll(latency_df, labels=None):
    
    label=list()
    
    if labels:
    
        for (idx, values) in enumerate(latency_df['Trial Group'].unique()):
            label.append(labels[str(values)])
    else:
        for values in latency_df['Trial Group'].unique():
            label.append(values)
            
    
    for stim in latency_df['Stim'].unique():
        lat_sub = latency_df[latency_df['Stim']==stim]
        lat_sub = lat_sub.drop('Stim', axis='columns')
        
        plt.subplots(figsize= (10,8))
        
        
        ax = sns.violinplot(data=lat_sub, x='Clusters', y='Spike Latency (s)', hue='Trial Group')
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        #ax.set_ylim((0, subClusterDF['Spike Counts'].max()+subClusterDF['Spike Counts'].std() + 1))
        sns.stripplot(data=lat_sub, x='Clusters', y='Spike Latency (s)', hue='Trial Group')
        plt.title('Window Spike Latency Graph', weight='bold')
        plt.ylabel('Spike Latency (s)')
        handles, labelsa = ax.get_legend_handles_labels()
        ax.legend(handles, [label[0], label[1], label[2]], loc='upper left')
        sns.despine()
        plt.figure(dpi=1200)
        plt.show()
        
def plotLatencyCore(latency_df, labels = None):
    
    
    if labels:
        latency_df['Trial Group'] = latency_df['Trial Group'].apply(lambda x: labels[str(x)])
    
        
            
    for stim in latency_df['Stim'].unique():
        lat_sub = latency_df[latency_df['Stim']==stim]
        lat_sub = lat_sub.drop('Stim', axis='columns')
        
        for cluster in lat_sub['Clusters'].unique():
            
            lat_sub_clu = lat_sub[lat_sub['Clusters']==cluster]
            
            plt.subplots(figsize= (10,8))
            
            
            ax = sns.violinplot(data=lat_sub_clu, x='Trial Group', y='Spike Latency (s)', palette="RdPu")
            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
            #ax.set_ylim((0, subClusterDF['Spike Counts'].max()+subClusterDF['Spike Counts'].std() + 1))
            sns.stripplot(data=lat_sub_clu, x='Trial Group', y='Spike Latency (s)')
            plt.title('Window Spike Latency Graph', weight='bold')
            plt.ylabel('Spike Latency (s)')
            handles, labelsa = ax.get_legend_handles_labels()
           
            sns.despine()
            plt.figure(dpi=1200)
            plt.show()
            
            
            
