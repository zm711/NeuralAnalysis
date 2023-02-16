# Visualization for the ClusterAnalysis

## `plot_spikes`

This function will generate a smoothed firing rate plot. It will request a smoothing value to apply a gaussian smoothing filter (std = nx6/5). It will also plot a raster plot below. It will generate a red line at the beginning of the stim time and different colored lines at the end depending on the stimulus. There are two flags `tg` for trial groupings which will split the data into multiple groups and `eb` for error bars. If `eb=True` there will be error shading around the firing rate.

![image](https://user-images.githubusercontent.com/92116279/219498714-f28a9beb-7720-4d09-8737-d8228d1b7606.png)


## `plot_z`

Creates z scored plots based on trial groups `tg=True` or without trial groups `tg=False`. With labels if given on in class or without if `None` or no attribute. 

![image](https://user-images.githubusercontent.com/92116279/219498798-ddd01e73-8afe-4f7c-94e5-629e649bc769.png)


## `plot_pc`

Generates the top two dimensional pcs plot to show seperation of clusters. Of note it takes the top 4 spaces and condenses down into 2. This means that clusters not resolved in two dimensions may still be better resolved in another dimensions. Both kilosort and `qcfn` use more dimensions. So this more to make rep figures.

Well resolved for two dimensions
![image](https://user-images.githubusercontent.com/92116279/219499187-ecea29a8-114c-410d-8cbd-6cad136f11c1.png)

Poorly resolved for two dimensions
![image](https://user-images.githubusercontent.com/92116279/219499238-4222b306-315e-4ce7-abf9-d30e1db392bd.png)


## `plot_wfs`

Has one parameter flag `ind` which if set to `True` will return individual waveforms with the mean overlaid. `ind =False` will return just the mean waveform.

![image](https://user-images.githubusercontent.com/92116279/219499991-bf7e59a0-e0d0-419a-a749-ee4d7d6d4bc5.png)


## `plot_drift`

This is a function to check for drift during the course of the recording. It marks any drift detection with red dots. May need to change the drift cutoffs in the `detectdrift.py` file for your use case.

![image](https://user-images.githubusercontent.com/92116279/219500047-625a17c3-f657-4cb1-af83-eeacc0f371d1.png)


