# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:34:57 2023

@author: ZacharyMcKenzie
"""

import numpy as np
import load_intan_rhd_format
from zmgenhelpers import getdirzm
from scipy import signal
import matplotlib.pyplot as plt
from stimulushelperszm import spike_prep
import statistics
import pandas as pd
import seaborn as sns


def plotVMR(filter_n: int):

    oldDir, currentDir, filename = getdirzm()

    filenamestr = filename + ".rhd"

    results = load_intan_rhd_format.read_data(filenamestr)

    for n in range(np.shape(results["board_adc_data"])[0]):

        board_adc_data = np.squeeze(results["board_adc_data"][n])

        sample_rate = int(results["frequency_parameters"]["amplifier_sample_rate"])

        baro_dig2 = np.array(
            np.logical_and(board_adc_data > 0.09, board_adc_data > 0), dtype=int
        )
        eventTimesDig_length, eventTimesDig, _ = spike_prep(baro_dig2, sample_rate)

        """my barostat stimuli are programmed as 20s. So anything below 15 is going
         to be an analog fluctation that must be removed. Do this for the start times
         and for the lengths"""

        eventStart = eventTimesDig[eventTimesDig_length > 15] * sample_rate
        eventLength = eventTimesDig_length[eventTimesDig_length > 15] * sample_rate

        """barostat runs at 1/20 Hz so the following filtering removes barostat, but
        we need to think about whether it should be 0.5 or 1.5 for example to clean up
        the signal the best"""

        """butter(order_number, frequency_filter, type_of_filter, output, sample_rate)"""

        sos = signal.butter(filter_n, 1, "highpass", output="sos", fs=sample_rate)
        filtered = signal.sosfilt(sos, board_adc_data)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

        ax1.plot(np.array(range(len(board_adc_data))) / sample_rate, board_adc_data)
        ax1.set_title("Raw Baro ADC Data", fontsize=8)
        ax1.set_ylabel("Volts")
        ax2.plot(np.array(range(len(filtered))) / sample_rate, filtered)
        ax2.set_title(f"{filter_n}th Order Bessel Filtered ADC Data", fontsize=8)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Volts")
        plt.tight_layout()
        sns.despine()
        plt.figure(dpi=1200)

        final_sum = np.zeros((len(eventStart),))
        trial_group = np.zeros((len(eventStart),))

        for event in range(len(eventStart)):

            start = int(eventStart[event])
            end = int(start + eventLength[event])

            trial_group[event] = (
                myround(statistics.mode(board_adc_data[start:end])) * 20  # 20mmHg/V
            )

            bsl = np.concatenate(
                (
                    filtered[start - 5 * sample_rate : start],  # 5(s) for bsl
                    filtered[end + 1 : end + 5 * sample_rate],
                )
            )

            event_bsl = np.sum(bsl[bsl < 0])  # this is our mean baseline value

            final_sum[event] = (
                np.sum(
                    filtered[start + sample_rate : end - sample_rate][
                        filtered[start + sample_rate : end - sample_rate] < 0
                    ]
                )
                / event_bsl
            )

        baro_auc = pd.DataFrame({"Trial Group": trial_group, "AUC": final_sum})

        fig2 = plt.subplots(figsize=(10, 8))
        sns.barplot(data=baro_auc, x="Trial Group", y="AUC", errorbar="sd", capsize=0.2)
        # plt.legend()
        sns.despine()
        plt.title(f"Animal from analog channel {n+1}", fontsize=8)
        plt.figure(dpi=1200)


"""myround allows use to convert to the nearest 0.25 * 20 is 5mmHg for barostat. I think
this about as much as I would trust the barostat. We could move this to 0.1 eventually
if we thing the barostat could do 2mmHg. But for the most part we will do changes of 10
or 15 mmHg which would be representable with 0.25"""


def myround(x, precision=2, base=0.25):
    return round(base * round(float(x) / base), precision)
