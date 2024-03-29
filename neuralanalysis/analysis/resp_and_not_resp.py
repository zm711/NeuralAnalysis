# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:48:29 2023

@author: ZacharyMcKenzie
"""

"""Clone the ClusterAnalysis object and split so that responsive vs non-responsive 
neurons can be analyzed. It automatically loads the responsive cluster ids into sp as 
well as loading the non-responsive cluster ids and returns a new ClusterAnalysis object
which can be analyzed with only the non-responsive Neurons"""

from ..ClusterAnalysis import ClusterAnalysis
import copy
from ..misc_helpers.label_generator import gen_resp


def resp_and_not_resp(
    myNeuron: ClusterAnalysis,
) -> tuple[ClusterAnalysis, ClusterAnalysis]:
    """This mutates in place the current `ClusterAnalysis` object
    to only list the responsive neurons and then returns a new
    `ClusterAnalysis` instance of the non_responsive neurons
    in case the user wants to compare these populations."""
    resp_neurons = myNeuron.resp_neuro_df
    non_resp = myNeuron.non_resp_df

    myNeuron_non_resp = copy.deepcopy(myNeuron)  # deep copy of all info

    sp_resp = gen_resp(resp_neurons, myNeuron.sp)  # generate new cids resp only
    sp_nonresp = gen_resp(non_resp, myNeuron_non_resp.sp)  # non-resp

    # load values into objects, return only new neuron
    myNeuron.sp = sp_resp
    myNeuron_non_resp.sp = sp_nonresp

    return myNeuron_non_resp
