# Visualization for the ClusterAnalysis

## `plot_spikes`

This function will generate a smoothed firing rate plot. It will request a smoothing value to apply a gaussian smoothing filter (std = nx6/5). It will also plot a raster plot below. It will generate a red line at the beginning of the stim time and different colored lines at the end depending on the stimulus. There are two flags `tg` for trial groupings which will split the data into multiple groups and `eb` for error bars. If `eb=True` there will be error shading around the firing rate.

![image](https://user-images.githubusercontent.com/92116279/219498714-f28a9beb-7720-4d09-8737-d8228d1b7606.png)


## `plot_z`

Creates z scored plots based on trial groups `tg=True` or without trial groups `tg=False`. With labels if given on in class or without if `None` or no attribute. 
Of note You may receive ` UserWarning: Attempting to set identical low and high xlims makes transformation singular; automatically expanding.`. This is because for neurons which are not z-scoreable (ie baseline std = 0) I plot as raw firing rate. This error can occur when you don't have any neurons in which case a blank graph will appear. See Below.

![image](https://user-images.githubusercontent.com/92116279/219498798-ddd01e73-8afe-4f7c-94e5-629e649bc769.png)
![image](https://user-images.githubusercontent.com/92116279/219796366-6feb66de-5d76-4df9-980c-1a1826b387d6.png)


## `plot_firingrate`
Creates a violinplot if `graph=v` otherwise it will generate a lineplot with each window provided and the trial groups. If `labels` provided as an attribute graph will be labeled otherwise it will just provide numbers. If the `neuro_resp_df` has been generated it will also provide a violin plot based on the type of responses.

![image](https://user-images.githubusercontent.com/92116279/219787327-4b395ac6-f93e-4440-b945-0f49dcfb7f10.png)


## `plot_pc`

Generates the top two dimensional pcs plot to show seperation of clusters. Of note it takes the top 4 spaces and condenses down into 2. This means that clusters not resolved in two dimensions may still be better resolved in another dimensions. Both kilosort and `qcfn` use more dimensions. So this more to make rep figures.

**Well resolved for two dimensions**


![image](https://user-images.githubusercontent.com/92116279/219499187-ecea29a8-114c-410d-8cbd-6cad136f11c1.png)

**Poorly resolved for two dimensions**


![image](https://user-images.githubusercontent.com/92116279/219499238-4222b306-315e-4ce7-abf9-d30e1db392bd.png)


## `plot_wfs`

Has one parameter flag `ind` which if set to `True` will return individual waveforms with the mean overlaid. `ind =False` will return just the mean waveform.

![image](https://user-images.githubusercontent.com/92116279/219499991-bf7e59a0-e0d0-419a-a749-ee4d7d6d4bc5.png)


## `plot_drift`

This is a function to check for drift during the course of the recording. It marks any drift detection with red dots. May need to change the drift cutoffs in the `detectdrift.py` file for your use case.

![image](https://user-images.githubusercontent.com/92116279/219500047-625a17c3-f657-4cb1-af83-eeacc0f371d1.png)

## `neuro_corr`

Does pairwise time bin correlation of neurons with lots of possibilities triggered by the `datatype`. Options include `frraw` which is just raw firing rate, `frsm` which is a smoothed firing rate, `zscore` which correlates z score responses. `time_bin_size` is normal flag. `tg` for trial groups. `labels` for labeling the graphs.

Ex with `datatype='zscore'`

![image](https://user-images.githubusercontent.com/92116279/219500830-ec336eba-91ef-4df1-996f-417fc8626cda.png)

## `acg`

Plots the autocorrelogram of a neuron's spikes to its other spikes to help visualization refractory period violations. This is graphical representation of the `isiV` value from the `qcfn`. Red lines are set by the flag `ref_dur` which is the refractory period in seconds. Depending on neuron type this could be between 0.001 and 0.003 so I set default to 0.002. 

![image](https://user-images.githubusercontent.com/92116279/219501400-ba2c2d8e-9f39-4ef2-9a3f-de27ac71c36d.png)

## `plot_cdf`

This is a function for plotting a cumulative distribution style figure of neural spiking by depth, amplitude and firing rate. It's just a way to visualize the data. `units_only` is a boolean to mark only responsive units. `laterality` will generate separate medial and lateral plots for stack H7 probes. It also generates a pdf version as well as some histograms.

![image](https://user-images.githubusercontent.com/92116279/219502120-24b19bcb-6877-4a91-9b8d-dadc276e8c4e.png)
![image](https://user-images.githubusercontent.com/92116279/219502176-06e2f2d9-90ae-45e2-9580-2a375237cb93.png)
![image](https://user-images.githubusercontent.com/92116279/219502181-95f59dc9-8c16-46b4-9ea1-e86cefa612c4.png)

