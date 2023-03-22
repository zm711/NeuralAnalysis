# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:02:53 2023

@author: ZacharyMcKenzie
"""

import numpy as np
import hashlib
import pandas as pd
import os
from . import genhelpers as gh


def merge_datasets(sp, df1, df2, df3, dtype: str):
    final_df = pd.merge(
        df1, df2, left_on="HashID", right_on="HashID", suffixes=("", "_y")
    )
    final_df = pd.merge(
        final_df,
        df3,
        left_on="HashID",
        right_on="HashID",
        suffixes=("", "_y"),
    )
    final_df.drop(final_df.filter(regex="_y$").columns, axis=1, inplace=True)

    if sp is not None:
        filename = sp["filename"]
        if filename in os.getcwd():
            if "pyanalysis" not in os.getcwd():
                os.chdir("pyanalysis")
        else:
            _, curr_dir, _ = gh.getdir()
            os.chdir(curr_dir)
    final_df.to_csv(filename + dtype + "df.csv", index=False)
    return final_df


def gen_zscore_df(sp, labels, allP):
    filename = sp["filename"]
    allP_list = list()

    for stim in allP.keys():
        labels_stim = labels[stim]
        sub_keys = sorted([float(key) for key in labels_stim.keys()])
        names = ["x", "y", "z"]
        index = pd.MultiIndex.from_product(
            [range(s) for s in allP[stim].shape], names=names
        )
        df = pd.DataFrame({"A": allP[stim].flatten()}, index=index)["A"]
        df = df.unstack(level="z").swaplevel().sort_index()
        df.index.names = ["Trial Group", "IDs"]
        df = df.sort_values(by=["IDs", "Trial Group"])
        df = df.reset_index(level=[0, 1])
        df["Stim"] = stim
        df["Trial Group"] = df["Trial Group"].apply(
            lambda x: labels_stim[str(sub_keys[x])]
        )

        allP_list.append(df)

    final_df = pd.DataFrame(allP_list[0])

    for idx in range(1, len(allP_list)):
        final_df = pd.concat([final_df, allP_list[idx]], ignore_index=True)

    final_df["HashID"] = final_df["IDs"].apply(
        lambda x: hashlib.sha256((str(x) + filename).encode()).hexdigest()
    )

    return final_df
