 # ClusterAnalysis
 
 ## Initialize the class
 The class is initialized by the spike data `sp` and the stimuli data `eventTimes`.
 
 ```python
 from neuralanalysis.ClusterAnalysis import ClusterAnalysis
 myNeuron = ClusterAnalysis(sp, eventTimes)
 ```
 **I will use myNeuron as the name for an instance of `ClusterAnalysis` for the rest of this doc**
 
 ## First methods
 
 ### Setting important recording values
 There are a few values beneficial for this type of analysis which cannot be predicted when loading the data. First trial groups are loaded as numeric values so likely for graphing it is necessary to map the numeric values to a stimulus value. This can be done with a nested-dictionary. I turn the trial groups into strings so the keys for this dictionary should be strings of the values used in `trialGroup` portion of `eventTimes`. The first set of keys are stimuli `Stim` and the values for each
 `Stim` are a set of `key:value` pairs converting from numeric to stimulus name.
 
 ```python
 my_stim = {'Stim':
           {
            '1.0': '180 Degrees',
            ..., 
            '10.0': '270 Degrees'
            }
            } 
 ```
 
In addition the depth of the probe if measured can be factored into the analysis (*e.g.* 500 um or 1000 um). If depth is included then all graphing and outputs will be relative to this probe measure. (*e.g.* a probe inserted 1000um registering a spike at 400um would mean that relative to the insertion into the tissue the spike is 600um deep, +/- measurement error etc). Finally since most of the nervous system is bilateral indicating whether the recording was done on the 'l' or 'r' may be useful. So running the `set_labels` methods of `ClusterAnalysis` allows for inputting these values. *In addition I am working on a universal way to determine shank id based on the weighted density of the spike amplitudes, (less accurate, but faster)*.
 
 ```python
 myNeuron.set_labels(labels=my_stim, depth = 1000, laterality='l')
 ```
 
 ### Generating raw waveform data
 Raw waveform data as opposed to the templates from `phy` can be beneficial for assessing peak-trough duration, amplitude etc. We can load this data from the `.bin` file that had been generated for kilosort based on our post-curation neural data. Of note this is a slow, RAM hungry (5-7gb RAM use) process that performs a memory map of the binary file). Since Kilosort is written in Matlab, the function also assumes that the `.bin` file was generated in Matlab, which will be Fortran ordered rather than the NumPy standard of `C`. So I load the structure in the `F` key of the `wf` attribute of kilosort. If you inspect `wf` you will see it also has a `C` key. This was historical since I had experimented with generating the `.bin` file in python. If you see nonsense values it could be that your file is using  `C` order. In `getWaveForms.py` I have commented out creating a `C`-ordered memory map, but this can be run by removing the comment hashes. Then this method can be run to see if this is the issue. This means that down stream in the `waveform_vals()` method the order would need to be switched to `C`. The one optional paramater is `num_chan` this is because kilosort gives only the number of useful channels rather than total channels. And the raw `.bin` final is organized by total number of channels. Thus to organize the memory map the `num_chans` must be entered as an `int`.
 
 ```python
 myNeuron.get_waveforms(num_chans=64) # 64 channel recording probe.
 ```
 
 ### Generating quality metrics
 Based on Nick Steinmetz's `sortingQuality` repo the `qcfn` method returns the isolation distance (Harris et al 2001) based on mahalobnis distance (which is dependent on the covariance matrix so it is less prone to being influenced by less important dimensions as Euclidean is) between pc values of clusters and the interspike interval violations (Schmitzer-Tobert et al. 2005, Hill 2011, Hill 2012). In short we can approximate the separation of units in our recording based on the distance in the pc spaces used by kilosort. These values are stored as `qc` in a dictionary with `uQ` the isolation distance and `cR` the contamination rate (see function for explanation of this value). I find that `uQ` unit quality is more consistent so for downstream methods I use that.
 
 ISI violations depend on the existence of neural refractory periods. It is based on Nick's Matlab code based on Hill et al 2011, and adapted based on UltraMegaSort from Hill 2012. In short it generates a false positive rate of spikes along with the raw number of violations. The benefit of false positive rate is it accounts for changes in spike number, but use of raw number of refractory period violations is relatively common (Chirila et al. 2023, Emanuel et al. 2021). this is stored as `isiv` with keys related to `fp` and `nViol` which is the fraction rate of violations. Of note the original method as proposed by Hill 2011 could lead to complex numbers (due to 'hidden' correlations between neurons), but I take the Hill 2011 approach with a flag that returns `fp=5` to indicate that the unit is failing the assumptions (d/t correlations or bursting activity). Downstream I just use raw violations, but I could include the false positive rate in a future addition.
 
 **There are no hard cutoffs for either value. Appropriate cutoffs must be determined based on your analysis**
 
 ```python
 myNeuron.qcfn()
 ```
 
 ## If loading a previous analysis
 Although the class requires `sp` and `eventTimes` for each initialization. There are load previous and save methods which can reduce the number of times methods must be run (although these methods save to the drive so if storage limited they are not required). Once waveform data and qcvalues have been generated files are saved and can be loaded for future analysis. This is accomplished with the `get_files` methods. Of note there is the `title` flag. This flag is to line up with the `title` flag in the `save_analysis` method. If some sort of subanalysis (for instance use of a different qc threshold) could be saved as separate files each given a unique title value.
 
 ```python
 myNeuron.get_files(title='')
 ```
 ## Analyzing Data
 
All analysis is split among generating values which are stored as class attributes and plotting functions which use these attributes for plotting and generate dataframe functions to allow for additional analysis or export. But to do your own analyses based on these values just access the appropriate attributes (list in ClusterAnalysisAttribute document). Methods where I often inspect values return from the class into the terminal, but most methods just store their returns internally in the class as attributes.
 
 ### Firing Rate Data
 
 Spike counts are the fundamental neural data for *in vivo* analysis. In order to generate these counts we need a `time_bin_size` given in seconds. 10-50 milliseconds works pretty well, but for slower neurons longer time bins provides more smoothing of the data and smaller time bins provides more 0 count bins (looking at firing rate works better with smoothing whereas raster plots only work when each bins has values of 0 or 1). This function generates the `psthvalues` attribute of the `ClusterAnalysis` class which is organized as a dictionary of neurons with each neuron having a `'BinnedArray'` with the matrix of firing rates give as an `nEvents x nTimeBins`. Additionally this function will ask window info for each stimulus. The window info should be given as start,end. So for -10 before stimulus to 10 after for a 10 second stimulus I would write -10,20 when prompted. There is a window option in the api call if you plan to always use the same windows.
 
 ```python
 psthvalues, windowlst = myNeuron.spike_raster(time_bin_size=0.05) # 50 millisecond example
 ```
 
 ### Z scored Data
 
 Data can also be z scored to allow for normalization of data. This requires that the std != 0, but to account for this an np.array called `normVal` is also produced which indicates the baseline mean and std or np.nan if un-z scoreable. *reminder Z score = x-mu/std* The return is `allP` a structure of the z scores stored as a dictionary of stimuli, followed by a matrix of the z score data. If `tg` is `True` then it will be `nUnits x nTrialGroups x nTimeBins` otherwise it will be `nUnits x nTimeBins`. `time_bin_size` like above is size of the time_bins. I also give an optional chance to input a `window_list` which is formatted as nested lists where each stimulus require two lists of time. Since this is complicated I explain below:
 
 #### Windows
 Each stimulus needs the baseline period to generate the baseline mean and std. [bslStart, bslEnd]. These times are in relation to the event onset. So do to do the 2 seconds before the stimuli onsets would be [-2,0]. To do 500 to 100 milliseconds would be [-.5, -.1]. Then each stimulus also needs a window. Since some neurons have after-discharges this value can be longer or shorter than the actual stimlus. If I am doing a 5 second stim I could analyze the first 2 seconds [0, 2] or the last 2 [3, 5] or [0, 7] for full stimulus but also with the 2 seconds after.
 
 So the final window_list would be [[bslS, bslE], [start, end]] or with multiple different stimuli it's best to set to `None` and let the function prompt each stimulus itself **IE when in doubt just enter `None` and let the function prompt you**
 ###
 
 Returns `allP`, the dictionary of z scores, `normVal` a dictionary with mean and std of the baseline/neuron or nan, `window` used for analysis. Attributes are `allP`, `normVal`, `zwindow`
 
 ```python
 allP, normVal, window= myNeuron.clu_zscore(time_bin_size = 0.05, tg=True, window=None)
 ```

### Raw Firing Rate Data
To collect raw firing rates over specific time windows the `firingratedf` function can be used. It will quickly calculate firing rates across trial groups over the windows set in the `window_dict`. As is usual a `time_bin_size` must be given. 
#### `window_dict`
This is a structure in time (s) given as:

```python
    window_dict = {
        "Rest": [-2, -1],
        "Onset": [0, 1],
        "Sustained": [0, 5],
        "Offset": [4,5],
        }
```
Once this value has been prepped the function will generate raw firing rates for each time period for each neuron stored as an attribute data frame `firing_rate_df`.


### Latency Data
Generating latency values can be a bit tricky since they rely on a lot of assumptions about the underlying distribution of the spiking data. There are multiple strategies. I'm working on generating a permutation based code, but currently I've implemented two different latency calculations. One is based on Chase and Young 2007 where they use a poisson distribution to check for the first spike which causes deviation from the base rate, lambda. Since many neurons I'm interested inhave low baseline rates and likley do not follow a poisson I also implement a strategy adapted from Mormann et al. 2008 taking first spike among events and then taking the median among each trial. The decision point in my code is currently firing rates of less than 2 Hz (as suggested by Mormann et al). Stored as an attribute in the class under `ClusterAnalysis.latency` in the form of a nested dict with np.arrays inside.

```python
myNeuron.latency(time_bin_size=0.05, bsl_win=[[-10,-5]], event_win=[[0,10]]) # nested lists are required
```

### Neuron metrics (depth etc)
Once the raw waveform data has been generated we can look for depth, duration, amplitude, and for multishank (only one implementation currently dual stacked H7 Cambridge Neurotech) position. The functions perform a lot of matrix math and list comprehensions to find the different values. In short we generate the depth based on the weigthed average of the spike amplitudes on the different electrodes of the probe. Amplitude is determined by finding the maximal waveform and then taking the difference from its min and max values. The peak-trough duration is determined by finding the index of minimum value and then finding the number of samples between the min and max value. For medial-lateral position the weighted average of the x coords are given and then organized into shank 1,2,3,4 and depending on lateraility of the probe position give a medial or lateral identity. These values are stored as attributes in the class: `max_waveform`, `waveform_dur` (the peak-trough duration), `waveform_depth` (corrected for shank if depth give in `set_labels`), `waveform_amps`, `shank_dict` (for stack H7 only currently).

```python
myNeuron.waveform_vals()
myNeuron.gen_wfdf()
```
The dataframe is stored as an attribute `waveform_df`.

### Responsive Neurons
Currently I am using user defined responsive neuron properties. Cutoffs vary (Emmanuel et al. 2021 uses 2.58 for the 99% CI, Chirila et al 2023 uses clustering to generate profiles). For historic reasons this function is a sub function stored in the `plot_z` function. It can be used to just generate the responsive neurons by setting the `plot` flag to `False`. With that out of the way, I use a decorator found in `plot_z_settings.py` to allow for multiple people in the same group to set their own z score cutoffs (filepath is `/visualization_ca/plot_z_settings.py`). So that file can be re-formated with desired cutoffs. If only one person is using this code delete the conditional logic and just put in the values. `inhib` for inhibitory, `onset` for onset neurons, `sustained` for sustained neurons, `offset` for near end of stimulus and `relief` for after discharge. The first value is desired z score and the second number is the number of bins required to count as the type given by the keyword. So this will depend on the `time_bin_size` set in `clu_zscore`. Once this decorator is set the method can be run. `plot_z` has a few flags, most of them are optional to override the attributes set. So `labels` will pull from the attribute `labels` but can be overridden, `tg` is a boolean for trial group (it should match with what was done in `clu_zscore`), `time_pt` is historic and will be removed soon. In order to control how neurons are classified as responsive the `na_settings.yaml` is used. This file is generated in the `pyanlaysis` folder if it does not exist, but can be edited for future analyses. Settings are below. First the desired z scores for a subtype are given. The first value is the value needed and the second number is the number of timebins with this value required. Then the `sorter_dict` values are percent of the stimulus to be used.
```yaml
- zscore:
    inhib:
    - -2
    - 3
    offset:
    - 2.5
    - 3
    onset:
    - 4
    - 3
    sustained:
    - 3.3
    - 5
- raw:
    sustained:
    - 75
- sorter_dict:
    Inhib:
    - .2
    Onset:
    - .1
    Onset-Offset:
    - .2
    - .2
    Relief:
    - .2
    Sustained:
    - 1
```
The `zero_point` is the time_bin in which the stimulus starts. I prefer calculating with some baseline in `allP` meaning that the first bin with stimulus is not 0 (start of baseline), but for example bin 20. The `event_len` will be the final bin of the stimulus in this case, eg bin 80. For relief in this example I start at bin 80 and go to the final length of bins for example 100. For onset, I do only a subset of bins etc. For two sub periods you just generate a list of 4 numbers. Currently 4 time points (ie two time periods) are the max you can do for the sorter dict.

#### Running `plot_z`. Once this is done for generating responsive neurons just run as 
```python
myNeuron.plot_z(labels = None, tg=True, time_pt=0, plot=False)
```

### Creating the responsive neuron dataframe
Since it is easier to work with a dataframe for mining this data I have the method `gen_respdf` which allows for quality control and response check. It will set two new attributes `resp_neuro_df` which has the responsive neurons, along with which stimuli trial groups they responded to and which time of response and `non_resp_df` which includes neurons that made quality cutoffs but did not respond to the stimuli. 

#### Quality Cutoffs
As stated above there are no hard values or rules for these cutoffs. Of note the python isolation distance is always lower than Matlab's due to the way the values are calculated, but they are correlated. So if reading a paper based on matlab code just test lower values for python. Another type of cutoff often used is isi violations. I tend to use isolation distance, but this function has an optional `isi` flag which requires a fraction rate you'd tolerate (ie 0.02 would be anything less than a 2% violation rate. In the future I may switch this to check for the false positive rate (Dan Hill 2011). To shutoff off quality metrics set `qcthres = 0` and leave `isi=None`. Otherwise pick one or both to use.

##### Common Error in qc
If the error len(noise) != len(qc) this is due to the way `sp` is generated from phy curation and how noise vs not noise are labeled. So if this occurs it means that your `qcvalues` no longer align with your clusters defined as `noise` by Phy. Before reporting a bug try rerunning `qcfn()` to realign the lengths of these variables. This is because we save a copy of `qcvalues.npy` in the working directory and if more curation occurs we need to overwrite this file rather than load it. To be explicit in what to run if this error is seen:
```python
myNeuron.qcfn()
```
#### Running the function
```python
myNeuron.gen_respdf(qcthres=10, isi=None)
```

#### Accessing the attributes
```python
responsive_df = myNeuron.resp_neuro_df
non_responsive_df = myNeuron.non_responsive_df
```
#### Mining the responsive neurons
As a dataframe it is easy to just pandas logic to see various metrics. For example to count the number of unique units one could do
```python
len(responsive_df['IDs'].unique())
```
Or to see raw counts of which neurons fell into the cutoffs you used one could try
```python
responsive_df['Sorter'].value_counts()
```

#### Calculating the Prevalence
Once that has been run you can quickly see the numbers by running `prevalence_calculator`. The assumptions of this calculator are that a neuron that is onset and sustained is just sustained since it has a peak and a sustained period and that a neuron that is onset-offset and onset is just onset. Basically this is a a square is a rectangle, but a rectangle is not a square situation where we tried to create priority classification for each neuron. 

### Subanalysis
Often times we want to look at subsets of neurons that have high quality or that respond to stimuli so we have three additional options

#### `qc_only`
This method does that same as `gen_respdf` except it only uses `qcthres` as its cutoff. This allows for assessing all neurons which meet your threshold. 

#### `gen_resp`
If you want to only look at responsive neurons or only plot responsive neurons you can use `gen_resp` which internally tells all the plotting functions and calculating functions to ignore non responsive neurons. This is great for prepping for a merged analysis, which will be discussed in another section.

#### `revert_cids`
this reverses the results of `qc_only` or `gen_resp` and unmasks back to the raw phy curated data. If you decide you don't want to do a subanalysis or want to try other `qcthres` or `isi` fractions use this.

### Saving
Finally, to save a bunch of different values of an analysis use the `save_analysis` method. This take the parameter `title` which can be used if you plan to save multiple iterations. You would then use the same `title` when using the `get_files` method. The save will occur in the pyanalysis folder and so look there when prompted in `get_files`. Always save the analysis at least once (space permitting) to allow you to quickly recollect old data like the `labels`, `depth`, etc. These values can always be written over if you change analysis, but it is helpful to reload these values.


### The \_\_repr\_\_
Since *in vivo* analysis often involves working with multiple files I provide a \_\_repr\_\_ which provides some helpful information. It will print the `filename` stored in the instance of the class. Then it will store the attributes currently stored within the class, which can help for pulling out values for *post-hoc* and then finally it will print a list of the methods in the class. In case the way I spelled a method seems confusing this will print out the spelling etc of the methods.

```python

>>> myNeuron
>>> This is the analysis of filename.2.17.23

>>> The initialized variables are ['sp', 'clu', 'clusterIDs', 'spikeTimes', 'eventTimes', 'filename', 
'allP', 'zwindow', 'normVal', 'depth', 'laterality', 'resp_neuro_df', 'non_resp_df']

>>> The methods are ['set_labels', 'gen_wfdf', 'waveform_vals', 'plot_spikes', 'plot_firingrate', 
'qcfn', 'get_files', 'gen_resp', 'spike_raster', 'plot_wfs',          
'firingratedf', 'clu_zscore', 'prevalence_calculator', 'cap_conversion', 'save_analysis', 
'plot_depth_scatter', 'plot_medlat_prevalence', 'get_waveforms', 'plot_pc', 'plot_cdf', 
'revert_cids', 'acg', 'plot_drift', 'gen_respdf', 'plot_z', 'neuro_corr', 'latency', 'qc_only']
```
