# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:18:34 2023

@author: ZacharyMcKenzie


decorator funtions in order to allow people to change their z score cutoffs (or raw
cutoffs) without changing the overall plotZscores file. find your name and change your
values. This relies on they name of path so if using the server it should be fine. If
trying to work on your laptop you may need to temporary change the string with your
letters to something that appears are your path (if i were zm on my lap top I would 
I would check for zm)

inhib sets your inhib value -1 or -2 is appropriate
sustained is a list with sustained[0] = z score and sustained[1] bin count
onset is for onset and for onset-offset and is the same z score followed by bin count

for raw_count you need to give an actual count. So in the else portion of the loop it 
says you need a minimum of at least 75 spikes to count.
"""

import os


def z_score_cutoff(func):
    def cut_off(*args, **kwargs):

        if "zach" in os.getcwd().lower():
            new_func = func(
                *args, inhib=[-2, 3], sustained=[3.3, 5], onset=[4, 3], offset=[2.5, 3]
            )
        elif "charl" in os.getcwd().lower():
            new_func = func(
                *args, inhib=[-2, 3], sustained=[1.5, 10], onset=[3, 3], offset=[2.5, 3]
            )
        elif "lyub" in os.getcwd().lower():
            new_func = func(
                *args, inhib=[-2, 3], sustained=[3.1, 10], onset=[3, 3], offset=[2.5, 3]
            )
        else:
            new_func = func(
                *args, inhib=[-2, 3], sustained=[3.4, 10], onset=[3, 3], offset=[2.5, 3]
            )
        return new_func

    return cut_off


def raw_count(func):
    def cut_off_raw(*args, **kwargs):
        if "zach" in os.getcwd().lower():
            new_func = func(*args, sustained=75)
        elif "charl" in os.getcwd().lower():
            new_func = func(*args, sustained=75)
        elif "lyub" in os.getcwd().lower():
            new_func = func(*args, sustained=75)
        else:
            new_func = func(*args, sustained=75)
        return new_func

    return cut_off_raw


def sorter_dict_adder(func):
    def dict_adder(*args, **kwargs):
        if "charl" in os.getcwd().lower():
            new_func = func(
                *args,
                sorter_dict={
                    "Sustained": [50, 100],
                    "Onset": [50, 65],
                    "Onset-Offset": [50, 65, 90, 110],
                    "Relief": [100, 150],
                    "Inhib": [50, 67],
                },
                **kwargs
            )
        else:
            new_func = func(*args, sorter_dict=None, **kwargs)
        return new_func

    return dict_adder
