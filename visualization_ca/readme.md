# Visualization for the ClusterAnalysis

## `plot_spikes`

This function will generate a smoothed firing rate plot. It will request a smoothing value to apply a gaussian smoothing filter (std = nx6/5). It will also plot a raster plot below. It will generate a red line at the beginning of the stim time and different colored lines at the end depending on the stimulus. There are two flags `tg` for trial groupings which will split the data into multiple groups and `eb` for error bars. If `eb=True` there will be error shading around the firing rate.

![image](https://user-images.githubusercontent.com/92116279/219498714-f28a9beb-7720-4d09-8737-d8228d1b7606.png)


## `plot_z`

Creates z scored plots based on trial groups `tg=True` or without trial groups `tg=False`. With labels if given on in class or without if `None` or no attribute. 

![image](https://user-images.githubusercontent.com/92116279/219497968-70699e6b-3816-463a-8dea-3a9e4b9dd932.png)
