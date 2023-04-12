# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:03:45 2023

@author: ZacharyMcKenzie

This will only work if Neo accepts my pull request. Otherwise I will generate a different work around.
"""

import neo
import os
from ..misc_helpers.genhelpers import getdir
from .intanutil.read_header import read_header
import numpy as np


def process_stim(filename: str = "") -> None:
    if len(filename) == 0:
        print("No filename given please select folder containing .rhd file")
        _, _, filename = getdir()
        filename = filename + ".rhd"

    assert ".rhd" in filename, "please make sure the filename given is an .rhd"

    final_adc, digital_data, sample_freq = read_intan_neo(filename)
    fid = open(filename, "rb")
    intan_header = read_header(fid)
    fid.close()
    intan_dict = dict(frequency_parameters=dict(amplifier_sample_rate=sample_freq))

    try:
        digital_data.shape
        value_matrix = preprocess_digital(digital_data, intan_header)
        intan_dict["board_dig_in_data"] = value_matrix
        intan_dict["board_dig_in_channels"] = intan_header["board_dig_in_channels"]
    except AttributeError:
        print("no digital data")

    try:
        final_adc.shape
        intan_dict["board_adc_data"] = final_adc
    except AttributeError:
        print("no adc data")

    os.mkdir("pyanalysis")
    os.chdir("pyanalysis")
    np.save(filename + ".intan.npy", intan_dict, allow_pickle=True)


def read_intan_neo(filename: str) -> tuple[np.array, np.array, float]:
    reader = neo.rawio.IntanRawIO(filename)
    print("Parsing header--this will take a while--")

    reader.parse_header()

    stream_list = list()
    for value in reader.header["signal_streams"]:
        stream_list.append(str(value[0]))

    adc_stream = [idx for idx, name in enumerate(stream_list) if "ADC" in name.upper()]

    digital_stream = [
        idx for idx, name in enumerate(stream_list) if "DIGITAL-IN" in name.upper()
    ]

    if len(adc_stream) != 0:
        adc_stream = adc_stream[0]
        adc_data = reader.get_analogsignal_chunk(
            stream_index=adc_stream, channel_indexes=[0]
        )

        final_adc = np.squeeze(
            reader.rescale_signal_raw_to_float(
                adc_data, stream_index=adc_stream, dtype="float64"
            )
        )
    else:
        final_adc = np.nan

    if len(digital_stream) == 0:
        try:
            digital_data = intan_neo_read_no_dig(reader)
        except:
            digital_data = np.nan
    else:
        digital_stream = digital_stream[0]
        digital_data = np.squeeze(
            reader.get_analogsignal_chunk(
                stream_index=digital_stream, channel_indexes=[0]
            )
        )

    for value in reader.header["signal_channels"]:
        sample_freq = value[2]
        break

    return final_adc, digital_data, sample_freq


def preprocess_digital(digital_data: np.array, header: dict) -> np.array:
    dig_in_channels = header["board_dig_in_channels"]
    values = np.zeros((len(dig_in_channels)), len(digital_data))

    for value in range(len(dig_in_channels)):
        values[value, :] = np.not_equal(
            np.bitwise_and(
                digital_data,
                (1 << dig_in_channels[value]["native_order"]),
            ),
            0,
        )

    return values


def intan_neo_read_no_dig(reader: neo.rawio.IntanRawIO) -> np.array:
    """if neo doesn't give easy access to digital stream then I recreate the function
    here"""
    digital_memmap = reader._raw_data["DIGITAL-IN"]  # directly grab memory map from neo
    dig_size = digital_memmap.size
    dig_shape = digital_memmap.shape
    # below we have all the shaping information necessary
    i_start = 0
    i_stop = dig_size
    block_size = dig_shape[1]
    block_start = i_start // block_size
    block_stop = i_stop // block_size + 1

    sl0 = i_start % block_size
    sl1 = sl0 + (i_stop - i_start)

    digital_data = np.squeeze(digital_memmap[block_start:block_stop].flatten()[sl0:sl1])

    return digital_data
