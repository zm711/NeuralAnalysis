# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:03:45 2023

@author: ZacharyMcKenzie
"""

import neo
import os
from ..misc_helpers.genhelpers import getdir
import numpy as np


def process_stim(filename: str = "") -> None:
    if len(filename) == 0:
        print("No filename given please select folder containing .rhd file")
        _, _, filename = getdir()
        filename = filename = ".rhd"

    assert ".rhd" in filename, "please make sure the filename given is an .rhd"

    final_adc, digital_data, sample_freq = read_intan_neo(filename)

    value_matrix, values = preprocess_digital(digital_data)
    dig_channels = {}
    for idx, value in enumerate(values):
        dig_channels[idx] = {"native_channel_name": "DIG" + str(value)}

    intan_dict = {
        "board_adc_data": final_adc,
        "board_dig_in_data": value_matrix,
        "board_dig_in_channels": dig_channels,
        "frequency_parameters": {"amplifier_sample_rate": sample_freq},
    }

    os.chdir("pyanalysis")
    np.save(filename + ".intan.npy", intan_dict, allow_pickle=True)


def read_intan_neo(filename: str) -> tuple[np.array, np.array, float]:
    reader = neo.rawio.IntanRawIO(filename)
    print("Parsing header--this will take a while--")

    reader.parse_header()

    stream_list = list()
    for value in reader.header["signal_streams"]:
        stream_list.append(str(value[0]))

    adc_stream = [idx for idx, name in enumerate(stream_list) if "ADC" in name.upper()][
        0
    ]

    digital_stream = [
        idx for idx, name in enumerate(stream_list) if "DIGITAL" in name.upper()
    ][0]

    adc_data = reader.get_analogsignal_chunk(
        stream_index=adc_stream, channel_indexes=[0]
    )

    final_adc = np.squeeze(
        reader.rescale_signal_raw_to_float(
            adc_data, stream_index=adc_stream, dtype="float64"
        )
    )

    digital_data = np.squeeze(
        reader.get_analogsignal_chunk(stream_index=digital_stream, channel_indexes=[0])
    )

    for value in reader.header["signal_channels"]:
        sample_freq = value[2]
        break

    return final_adc, digital_data, sample_freq


def preprocess_digital(digital_data: np.array) -> tuple[np.array, np.array]:
    values = np.nonzero(np.unique(digital_data))[0]

    value_matrix = np.zeros((len(values), len(digital_data)), dtype=np.int16)
    for idx, value in enumerate(values):
        value_matrix[idx] = np.where(digital_data == value, 1, 0)

    return value_matrix, values
