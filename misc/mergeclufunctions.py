# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:11:16 2022

@author: ZacharyMcKenzie
"""


import numpy as np
import pandas as pd


def lookup_table_gen(sp_list: list) -> pd.DataFrame:
    filename_list = list()
    cids_list = list()
    hash_ids_final = list()

    for sp in sp_list:
        curr_filename = sp["filename"]
        curr_cids = sp["cids"]

        hash_ids = [hash(str(cid) + curr_filename) for cid in curr_cids]

        filename_list += [curr_filename] * len(hash_ids)
        cids_list += list(curr_cids)
        hash_ids_final += hash_ids

    id_table = pd.DataFrame(
        {"Filenames": filename_list, "Cluster IDs": cids_list, "HashID": hash_ids_final}
    )

    return id_table


def merge_df(*args: pd.DataFrame) -> pd.DataFrame:
    final_df = pd.DataFrame({})
    for idx in range(len(args)):
        if idx == 0:
            final_df = args[idx]
        else:
            final_df = final_df.merge(args[idx], left_on="HashID", right_on="HashID")

    return final_df
