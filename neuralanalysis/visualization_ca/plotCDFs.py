# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:49:38 2022

@author: ZacharyMcKenzie


makeCDF is a function which allows for the generation of pdf and cdf plots of
spike depths, spike amplitudes, and spike rates to see spike properties as related
to the tissue.

INPUTS: sp: the dictionary of a kilosort run. It contains spike_times as well 
            as ycoords, temps are template waveforms, winv is the whitening
            matrix, spike_templates are the the templates that kilosort used to
            sort the spikes
        depth: this is the depth of the probe. If not given the code will ref
               everything to the probe instead of in absolute terms of the 
               tissue. Thus giving a depth is ideal. It should be an integer or
               a float
        unit_only: a boolean flag which will look only at spikes from clusters
                   that the user wants rather than all spikes. This relies on
                   cids in sp (ie sp['cids']) to be loaded with the desired
                   clusters. This can be done with the ClusterAnalysis class 
                   pretty easily.
               
OUTPUTS: CDF graph and PDF graph of the spikes.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def makeCDF(sp, depth=None, units_only=False, laterality=False) -> None:

    spike_times = sp["spikeTimes"]
    y_coords = np.sort(sp["ycoords"])
    probe_len = max(y_coords)
    y_set = sorted(list(set(y_coords)))
    pitch_end = y_set[-1] - y_set[-2]
    pitch_start = y_set[1] - y_set[0]
    pitch = min(pitch_start, pitch_end)

    spike_depths, spike_amps, spike_x = getTempPos(sp, depth)

    if units_only:
        spike_depths, spike_amps, spike_times, spike_x = desiredSpikes(
            sp, spike_depths, spike_amps, spike_times, spike_x
        )

    if laterality:

        lateral_x = np.logical_or(
            spike_x < 150, spike_x > 750
        )  # for H7 probe shankA then shankB
        depth_bins, amp_bins, recording_dur = genADBins(
            spike_amps[lateral_x], probe_len, pitch, spike_times[lateral_x], depth
        )
        pdfs, cdfs = computeWFamps(
            spike_amps[lateral_x],
            spike_depths[lateral_x],
            amp_bins,
            depth_bins,
            recording_dur,
        )
        plotPCDFs(pdfs, cdfs, depth_bins, amp_bins, title="Lateral Neurons")
        medial_x = np.logical_or(
            np.logical_and(spike_x > 450, spike_x < 750),
            np.logical_and(spike_x > 150, spike_x < 450),
        )  # for H7 shankB then shankA
        depth_bins, amp_bins, recording_dur = genADBins(
            spike_amps[medial_x], probe_len, pitch, spike_times[medial_x], depth
        )
        pdfs, cdfs = computeWFamps(
            spike_amps[medial_x],
            spike_depths[medial_x],
            amp_bins,
            depth_bins,
            recording_dur,
        )
        plotPCDFs(pdfs, cdfs, depth_bins, amp_bins, title="Medial Neurons")

    else:

        depth_bins, amp_bins, recording_dur = genADBins(
            spike_amps, probe_len, pitch, spike_times, depth
        )
        pdfs, cdfs = computeWFamps(
            spike_amps, spike_depths, amp_bins, depth_bins, recording_dur
        )
        plotPCDFs(pdfs, cdfs, depth_bins, amp_bins)


"""desiredSpikes uses the current cids which are based on sub-analysis criteria
in order to generate depths, amps, and times of only the desired spikes. It is
triggered by the units_only flag in the makeCDF function"""


def desiredSpikes(
    sp: dict,
    spike_depths: np.array,
    spike_amps: np.array,
    spike_times: np.array,
    spike_x: np.array,
) -> tuple[np.array, np.array, np.array, np.array]:

    cids = sp["cids"]
    clu = sp["clu"]

    """list comprehension to create boolean array if I come up with a better way
    I will switch but this works so I used it"""
    spike_index = np.array([True if x in cids else False for x in clu])

    corr_spike_depths = spike_depths[spike_index]
    corr_spike_amps = spike_amps[spike_index]
    corr_spike_times = spike_times[spike_index]
    corr_spike_x = spike_x[spike_index]

    return corr_spike_depths, corr_spike_amps, corr_spike_times, corr_spike_x


"""Don't use-- this is based on units rather than clusters. Need to fix
def genAD(waveAmps, waveFormDepth):
    spike_amps = waveAmps[:, 0]
    spike_depth = waveFormDepth
    
    return spike_amps, spike_depth

"""


"""getTempPos is a function which takes in sp and returns spike_depths and 
spike_amps for each spike regardless of cluster id. Thus it is a nSpikesx1 set 
values which can be used with spike_times and clu to determine some spike
properties

INPUTS: sp: dict of kilosort values
        depth: optional integer or float to reference to tissue rather than to
        the probe itself. In order to work it must be the value of the depth
        as given by linlab2, ie, the depth of the probe in the tissue
        
OUTPUTS: spike_depths: nSpikes x 1 giving the depths of each spike that kilosort
                       found
        spike_amps: nSpikes x 1 giving the amps of each spike that kilosort
                     found
"""


def getTempPos(sp: dict, depth=None) -> tuple[np.array, np.array, np.array]:

    temps = sp["temps"]  # these are our templates from Kilosort
    temp_scaling_amps = sp["tempScalingAmps"]  # these scale amplitudes
    winv = sp["winv"]  # whitening filtered applied by kilosort
    ycoords = sp["ycoords"]  # y-coords along the shanks
    spike_templates = sp["spikeTemplates"]  # template ids for each spike
    xcoords = sp["xcoords"]

    """First we unwhiten our templates by doing matrix multiplication with
    winv on the temps"""
    tempsUnW = np.zeros(np.shape(temps))

    for temp in range(np.shape(temps)[0]):
        tempsUnW[temp, :, :] = np.squeeze(temps[temp, :, :]) @ winv

    """Get the amplitude by doing max-min for each channel"""
    temp_chan_amps = tempsUnW.max(axis=1) - tempsUnW.min(axis=1)
    temp_amps_unscaled = temp_chan_amps.max(axis=1)

    """We 0 out any small amplitudes"""
    threshold_vals = temp_amps_unscaled * 0.3
    threshold_vals = np.expand_dims(
        threshold_vals, axis=1
    )  # need to expand to allow broadcasting

    thres_idx = temp_chan_amps < threshold_vals  # any values less than our threshold
    temp_chan_amps[thres_idx] = temp_chan_amps[thres_idx] = 0

    """This does a center of mass depth measurement"""
    template_depths = np.sum((temp_chan_amps * ycoords), axis=1) / np.sum(
        temp_chan_amps, axis=1
    )
    template_x = np.sum((temp_chan_amps * xcoords), axis=1) / np.sum(
        temp_chan_amps, axis=1
    )
    """depth is the depth of the probe so we can go from probe referenced values
    to actual depths in tissue. If we don't have this we keep it in probe terms"""

    if depth:
        template_depths_corr = depth - template_depths
    else:
        template_depths_corr = template_depths

    """Based on each template we give each spike it's depth"""
    spike_depths = template_depths_corr[spike_templates]

    """For each spike we calculate it's amplitude based on kilosort amp times
    the scaling factor to get the actual amplitude"""
    spike_amps = temp_amps_unscaled[spike_templates] * temp_scaling_amps
    spike_x = template_x[spike_templates]

    return spike_depths, spike_amps, spike_x


def genADBins(
    spike_amps: np.array,
    probe_len: float,
    pitch: float,
    spike_times: np.array,
    depth=None,
) -> tuple[np.array, np.array, float]:

    """Create depth bins based on size of the probe and create bins based on
    distance between the channels (pitch)"""
    depth_bins = np.linspace(0, probe_len, num=int((probe_len) / pitch))

    """if probe depth is given we need to correct to get absolute distance rather
    than probe distance"""
    if depth:
        dep_corr = depth - np.max(depth_bins)
        depth_bins = depth_bins + dep_corr

    """amplitude bins are based on a min of 800 or the max voltage of the spikes
    Then we create bins spaced at 30 apart. Could change the 30, but this was
    Nick's rec so keeping it for now"""
    amp_bin_max = np.min([np.max(spike_amps), 800])
    amp_bins = np.linspace(0, amp_bin_max, num=int(amp_bin_max / 30))

    """recording_dur gives us the time to convert to Hz later: we just start
    the time at 0 for these purposes though I would consider starting at 
    spike_times[0] instead since of 0 since spikes I care about"""
    recording_dur = spike_times[-1]

    return depth_bins, amp_bins, recording_dur


"""computes the pdf and cdf of amps by depth by spikes. All we do is quickly
iterate over our number of depth bins and generate a histogram of points. We
divide by recording_dur to switch between spike counts into Hz. Then our pdf
are just the counts loaded in the appropriate depth bin. To get our cdf I take
a copy of the histogram going in reverse and then do a cumulative summation
then reverse back to the same direction as the pdf and load into cdf. We return
the pdf and cdf."""


def computeWFamps(
    spike_amps: np.array,
    spike_depth: np.array,
    amp_bins: np.array,
    depth_bins: np.array,
    recording_dur: float,
) -> tuple[np.array, np.array]:

    n_d_bins = len(depth_bins) - 1  # number of depth bins
    n_a_bins = len(amp_bins) - 1  # number of amplitude bins sub 1 since the bins have
    # edges instead of centers

    pdfs = np.zeros((n_d_bins, n_a_bins))
    cdfs = np.zeros((n_d_bins, n_a_bins))

    for b in range(n_d_bins):
        """create our logical for counting amps based on depths"""
        depth_bins_logic = np.logical_and(
            spike_depth > depth_bins[b], spike_depth < depth_bins[b + 1]
        )
        h = np.histogram(spike_amps[depth_bins_logic], amp_bins)[0]
        h = h / recording_dur  # Hz conversion (ie spikes to spikes/time, ie hz)
        pdfs[b] = h  # just the raw counts
        rev_h = h[::-1].copy()  # make a deep copy and reverse order
        thiscdf = np.cumsum(rev_h)  # sum to create a cumulative sum
        cdfs[b] = thiscdf[::-1]

    return pdfs, cdfs


"""I use this function just to round my axes so they aren't long floats. Then I 
load the values into a DataFrame which play super nicely with seaborn. Then I run
the core plotting once for the pdf and once for the cdf"""


def plotPCDFs(
    pdfs: np.array, cdfs: np.array, depth_bins: np.array, amp_bins: np.array, title=""
) -> None:

    final_depth = ["%.1f" % float(x) for x in depth_bins[1:]]  # just rounding

    final_amp = ["%.2f" % float(x) for x in amp_bins[1:]]  # get rid of first bin (no 0)

    """Dataframe will allow seaborn to auto heatmap pretty nicely"""
    pdfsDF = pd.DataFrame(pdfs, columns=final_amp, index=final_depth)
    cdfsDF = pd.DataFrame(cdfs, columns=final_amp, index=final_depth)

    """set up to make histogram of just of the spike number vs depth"""
    pdfsDFhisto = pdfsDF.sum(axis=1)
    pdfsDFhisto.index.name = "Depth (µm)"
    pdfsDFhisto.name = "Spikes (Hz)"
    # cdfsDFhisto = cdfsDF.sum(axis=1)
    histoPDF = pdfsDFhisto.to_frame().reset_index()
    histoPDF["Depth (µm)"] = histoPDF["Depth (µm)"].astype(float).apply(lambda x: -x)

    counts = np.array(histoPDF["Spikes (Hz)"])
    depths = np.array(histoPDF["Depth (µm)"])
    # pdfhisto = pdfsDF.reset_index()
    # pdfhisto['Depth (µm)'] = pdfhisto['Depth (µm)'].astype(float).apply(lambda x: -x)
    plot_PDFhisto(counts, depths, title + "Histo PDF")

    PDF_CDFcore(cdfsDF, title + " cdf")
    PDF_CDFcore(pdfsDF, title + " pdf")


def PDF_CDFcore(im: pd.DataFrame, title: str) -> None:

    fig, ax = plt.subplots(figsize=(10, 8))

    """heatmap with vmin=0 since 0 Hz is the lowest possible value. I also format
    as %.2e in order to have scientific notation with 2 decimal places"""
    ax = sns.heatmap(
        data=im, vmin=0, cbar_kws={"label": "Firing Rate (Hz)", "format": "%.2e"}
    )

    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    # ax.set_xticklabels(amp_bins)
    # ax.set_yticklabels(depth_bins)

    """Makes labeling sparser for easier visualization"""

    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 2 == 0:  # every other label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    for ind, label in enumerate(ax.get_yticklabels()):
        if ind % 2 == 0:  # every other label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)

    plt.title(f"{title.upper()}")
    plt.tight_layout()
    plt.ylabel("Depth (µm)")
    plt.xlabel("Amplitude (µV)")
    plt.figure(dpi=1200)
    plt.show()


def plot_PDFhisto(counts: np.array, depths: np.array, title: str) -> None:

    fig, ax = plt.subplots(figsize=(10, 8))

    plt.barh(y=depths, width=counts, height=40, color="k")
    plt.title(f"{title.upper()}")
    plt.tight_layout()
    sns.despine()
    plt.ylabel("Depth (µm)")
    plt.xlabel("Spikes (Hz)")
    plt.figure(dpi=1200)
    plt.show()


"""makes a scatter plot of the clusters of analysis with depth on y axis
and spikes in hertz on the x-axis. It requires wf to get the clusterIDs which
have been determined for waveform_depths. Also needs waveform_depths to get 
'true' depths of each cluster. It needs sp to get clu and cids. If units_marked
is set to True it will use the currents cids (for whatever subanalysis is being)
done and will then mark those clusters in red rather then black"""


def plotDepthSpike(
    sp: dict, wf: dict, waveform_depths: np.array, units_marked=False
) -> None:

    clusterIDs = wf["F"]["ClusterIDs"]

    spike_times = sp["spikeTimes"]
    clu = sp["clu"]

    spike_count = np.zeros((len(clusterIDs), 1))
    for idx, cluster in enumerate(clusterIDs):
        spike_count[idx] = sum(clu[clu == cluster]) / spike_times[-1]

    spikes = np.squeeze(spike_count)

    if units_marked:
        cids = sp["cids"]
        desired_spikes = np.array([True if x in cids else False for x in clusterIDs])
        plotDepthSpikeCore(
            -waveform_depths,
            spikes=spikes,
            title="Scatter of Units Depth and Spikes(Hz)",
            desired_spikes=spikes[desired_spikes],
            desired_depths=-waveform_depths[desired_spikes],
        )
    else:

        plotDepthSpikeCore(
            -waveform_depths, spikes, title="Scatter of Units Depth and Spikes(Hz)"
        )


"""just makes a scatter"""


def plotDepthSpikeCore(
    depths: np.array,
    spikes: np.array,
    title="Scatter",
    desired_spikes=np.array([0, 0]),
    desired_depths=None,
) -> None:

    fig, ax = plt.subplots(figsize=(10, 8))

    plt.scatter(x=spikes, y=depths, alpha=0.7, color="k")
    if desired_spikes.any():
        plt.scatter(x=desired_spikes, y=desired_depths, color="r")
    plt.title(f"{title.upper()}")
    plt.tight_layout()
    sns.despine()
    plt.ylabel("Depth (µm)")
    plt.xlabel("Spikes (Hz)")
    if desired_spikes.any():
        plt.legend(["All Units", "Responsive Units"], frameon=False)

    plt.figure(dpi=1200)
    plt.show()
