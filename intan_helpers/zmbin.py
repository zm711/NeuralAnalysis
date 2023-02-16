# -*- coding: utf-8 -*-
#!/bin/env python3

"""
Created on Tue Jun 14 09:57:57 2022

@author: ZacharyMcKenzie
"""
import numpy as np
from intan_helpers import load_intan_rhd_format
from zmgenhelpers import getdirzm
import os
import glob


def binConvert() -> None:

    oldDir, currentDir, filename = getdirzm()

    filenamestr = filename + ".rhd"
    filenamebin = filename + ".bin"
    results = 0

    if (
        len(glob.glob("*.bin")) > 0
    ):  # check if binary file exists and if not generates it
        print("Binary file already generated")
    else:
        results = load_intan_rhd_format.read_data(filenamestr)
        amplifier_dataI = results["amplifier_data"]
        if amplifier_dataI[0][0] != np.int16:
            amplifier_dataI = np.array(amplifier_dataI, dtype=np.int16)

        filenamebin = open(filenamebin, "wb+")
        filenamebin.write(amplifier_dataI)
        filenamebin.close()

    # see if pyanalysis directory exists and if not makes it
    if os.path.isdir("pyanalysis"):
        os.chdir("pyanalysis")
    else:
        os.mkdir("pyanalysis")
        os.chdir("pyanalysis")

    if len(glob.glob("*intan.npy")) > 0:
        print("Data already exists please use loadintanvalues() to see")
    else:
        if results == 0:
            os.chdir("..")
            file_name = glob.glob("*.rhd")
            if len(file_name) == 1:
                results = load_intan_rhd_format.read_data(filenamestr)
                results.pop("amplifier_data", "Already Done")
            else:
                intan_list = list()
                results = dict()
                for file in file_name:
                    result_sub = load_intan_rhd_format.read_data(file)
                    result_sub.pop("amplifier_data")
                    intan_list.append(result_sub)

                for key in intan_list[0].keys():
                    if "data" in key:
                        for idx in range(len(intan_list)):
                            if idx == 0:
                                results[key] = intan_list[0][key]
                            else:
                                results[key] = np.hstack(
                                    (results[key], intan_list[idx][key])
                                )
                    elif type(intan_list[0][key]) == np.ndarray:
                        results[key] = np.concatenate(
                            list(sub[key] for sub in intan_list)
                        )
                    else:
                        results[key] = intan_list[0][key]

                    if "aux_input_data" in key:
                        results[key] = results[key].reshape((3, -1))

            os.chdir("pyanalysis")

        np.save(filenamestr[:-4] + "intan.npy", results, allow_pickle=True)
