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


import yaml

"""change inhib, sustained, onset, offset with notes from above here."""


def z_score_cutoff(func):
    def cut_off(*args, **kwargs):
        with open("na_settings.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        zscore = config[0]
        new_func = func(
            *args,
            inhib=zscore["inhib"],
            sustained=zscore["sustained"],
            onset=zscore["onset"],
            offset=zscore["offset"]
        )

        return new_func

    return cut_off


def raw_count(func):
    def cut_off_raw(*args, **kwargs):
        with open("na_settings.aml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        raw = config[1]

        new_func = func(*args, sustained=raw["sustained"])

        return new_func

    return cut_off_raw


"""change me to change the time bins observed for determining
responsiveness see if "charl" example for set-up. Change
values in sorter_dict"""


def sorter_dict_adder(func):
    def dict_adder(*args, **kwargs):
        with open("na_settings.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        sorter_dict = config[3]

        new_func = func(*args, sorter_dict=sorter_dict, **kwargs)

        return new_func

    return dict_adder
