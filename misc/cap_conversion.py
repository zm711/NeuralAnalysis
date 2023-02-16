#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:55:33 2023

@author: zacharymckenzie
"""

import numpy as np

def cap_conversion(eventTimes:dict, time_pt:float, sp:dict)-> tuple[dict,dict]:
    
    filename = sp['filename']
    labels = dict() #create our final dict
    
    trial_groups_current = eventTimes['ADC1tot']['TrialGroup'].copy() #need to write so copy
    event_times_current = eventTimes['ADC1tot']['EventTime'] #need events for cuotff
    
    times_to_convert = event_times_current > time_pt #create our boolean
    
    #mutate in place. Add 20 since 20 is 100 mmHg
    np.place(trial_groups_current, times_to_convert, trial_groups_current+20)
    
    eventTimes['ADC1tot']['TrialGroup'] = trial_groups_current # load into data
    
    trials = set(trial_groups_current)
    #create new labels dict   
    for trial in trials:
        if trial <= 20:
            labels[str(trial)] = str(trial * 0.25 * 20) + " mmHg Pre-Capsaicin"
        else:
            labels[str(trial)] = str((trial-20) *0.25 * 20)+' mmHg Post-Capsaicin '
    #if person wants just peak adc data need to change that too
    if eventTimes.get('ADC1', 0):
        trial_groups_current = eventTimes['ADC1']['TrialGroup'].copy()
        event_times_current = eventTimes['ADC1']['EventTime']
        
        times_to_convert = event_times_current > time_pt
        
        np.place(trial_groups_current, times_to_convert, trial_groups_current+20)
        eventTimes['ADC1']['TrialGroup'] = trial_groups_current
        
    np.save(filename+'eventTimes.npy', eventTimes)    
    return eventTimes, labels
    
    
    
    
    
    
