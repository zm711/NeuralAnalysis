# ClusterAnalysis
Pipeline for analyzing kilosort/phy data
some functions based off of Nick Steinmetz's matlab code, others original


## Requirements

I've tested python 3.8-3.10. I've also tested some other packages and these are my current to recreate. I work in spyder so I generate a conda env with its own spyder I had stability issues so I'm sticking with 5.3.3 for now. Using the packages below will prevent any compatibility issues.

```sh
 conda create -n spyderweb -c conda-forge python=3.10 spyder=5.3.3 numpy=1.23 pandas=1.5 scipy=1.10 matplotlib=3.63 h5py=3.8 seaborn=0.12 scikit-learn=1.12 cython=0.29 sympy=1.11 numba=1.23
 ```
 ## General Notes

1. When I orginally designed this class it was largely based on dictionaries, but I find that dataframes are actually better for a lot of analyses, but this requires translation functions from my original dictionary structures (which all my plotting functions were writtened based on--although I often convert to dataframes within the plotting functions themselves, so maybe eventually I'll remove the dictionaries altogether). 

2. The goal will be to stack all analyses into large final dataframes, which are indexed by what I call the HashID. Since kilosort and phy always just give the same numbers between recordings I take the hash of the filename with the cluster number to generate a unique id for each neuron for each recording. This allows me to interact with multiple datasets while keeping track of unique ids. Haven't had any hash collisions yet.

3. With these caveats I save both the dictionaries and dataframes so either structure can be used for post-hoc analyses

 ## Inputs for the Class
 
 ### Neural Data
 This pipline is based specifically on kilosort/phy outputs. These include a series of numpy (.npy) files which are generated by kilosort and edited by phy. As long as phy is the final output then my `loadsp` function should load all data into a dictionary which I call `sp` for spike properties. Within the `ksanalysis.py` file all necessary keys can be found, but a few important ones include: `spikeTimes` are the list of spike occurrences in seconds, `clu` is the list of curated cluster ids for each spike, `cids` are the list of all possible ids after curation. 
 
 ### Stimulus Data
 Many *in vivo* experiments are based on recording neural activity in response to stimulus data. Stimulus data also needs to be loaded as a specific dictionary structure. Since I use Intan for recording data and stimuli, I use their python code for extracting stimulus data and generating this dictionary. If another data recording system is used the general format required is: eventTimes['stim channel']['EventTime']: np.array (of events), eventTimes['stim channel']['Length']: np.array (of lengths for each event), eventTimes['stim channel']['TrialGroup']: np.array of the different degree of stim (for example changing light orientation or changing pressure of stimuli) and eventTimes['stim channel']['Stim']: str ( the name of the stimulus for plotting). 
 
 #### Intan_helpers Folder
 Includes functions to run intan data automatically along with my functions for prepping the stimulus data from the .rhd file. To generate an appropriate `eventTimes` for the class reading through these functions will be key. For the most up to date intan functions they offer their functions as a zip file although of note if downloading from their website I changed the import structure to fit with my pipeline (ie from intan_helpers.file import function)
 
 ## Initialize the class
 The class is initialized by the spike data (sp) and the stimuli data (eventTimes).
 
 ```python
 from ClusterAnalysis import ClusterAnalysis
 myNeuron = ClusterAnalysis(sp, eventTimes)
 ```
 **I will use myNeuron as the name for an instance of `ClusterAnalysis` for the rest of this doc**
 
 ## First methods
 
 ### Setting important recording values
 There are a few values beneficial for this type of analysis which cannot be predicted when load the data. First trial groups are loaded as numeric values so likely for graphing it is necessary to map the numeric values to a stimulus value. This can be done with a dictionary. my_stim = {'1.0': '180 Degrees',..., '10.0': '370 Degrees'}. In addition the depth of the probe if measured can be factored into the analysis (500 um or 1000 um). Finally since most of the nervous system is bilateral indicating whether the recording was done of the 'l' or 'r' may be useful. So running the `set_labels` methods of `ClusterAnalysis` allows for inputting these values.
 
 ```python
 myNeuron.set_labels(labels = my_stim, depth = 1000, laterality='l')
 ```
 
 ### Generating raw waveform data
 Raw waveform data as opposed to the templates from phy can be beneficial for assessing peak-trough duration, amplitude etc. We can load this data from the `.bin` file that had been generated for kilosort based on our post-curation neural data. Of note this is a slow, RAM hungry process that performs a memory map of the binary file. It may need to be done on a server or a high-RAM workstation.
 
 ```python
 myNeuron.get_waveforms()
 ```
 
 ### Generating quality metrics
 Based on Nick Steinmetz's sortingQuality repo the `qcfn` method returns the isolation distance (Harris et al 2001) based on mahalobnis distance between pc values of clusters and the interspike interval violations (Schmitzer & Tobin et al. 2006). In short we can approximate the separation of units in our recording based on the distance in the pc spaces used by kilosort. Since Euclidean distance is prone to be influenced by less important features use of mahalobnis which uses the covariance matrix is less prone to these errors. ISI violations depend on the existence of neural refractory periods. 
 
 **There are no hard cutoffs for either value. Appropriate cutoffs must be determined based on your analysis**
 
 ```python
 myNeuron.qcfn()
 ```
 
 ## If loading a previous analysis
 Although the class requires `sp` and `eventTimes` for each initialization. There are load previous and save methods which can reduce the number of times methods must be run. Once waveform data and qcvalues have been generated files are saved and can be loaded for future analysis. This is accomplished with the `get_files` methods. Of note there is the `title` flag. This flag is to line up with the `title` flag in the `save_analysis` method. If some sort of subanalysis (for instance use of a different qc threshold) could be saved as separate files each given a unique title value.
 
 ```python
 myNeuron.get_files(title='')
 ```
 ## Analyzing Data
 
 All analysis is split amongst generating values which are stored as class attributes and plotting functions which use these attributes for plotting. But to do your own analyses based on these values just access the appropriate attributes (list at bottom of this document). Methods where I often inspect values return from the class, but most methods just store as attributes.
 
 ### Firing Rate Data
 
 Spike counts are the fundamental neural data for in vivo analysis. In order to generate these counts we need a `time_bin_size` given in seconds. 10-50 milliseconds work pretty well, but for slower neurons longer time bins provides more smoothing of the data and smaller time bins provides more 0 count bins. This function generates the `psthvalues` attribute of the `ClusterAnalysis` class which is organized as a dictionary of neurons with each neuron having a 'BinnedArray' with the matrix of firing rates give as an nEvents x nTimeBins.
 
 ```python
 psthvalues, windowlst = myNeuron.spike_raster(time_bin_size=0.05) # 50 millisecond example
 ```
 
 ### Z scored Data
 
 Data can also be z scored to allow for normalization of data. This requires that the std != 0, but to account for this an np.array called `normVal` is also produced which indicates the baseline mean and std or np.nan if un-z scoreable. *reminder Z score = x-mu/std* The return is `allP` a structure of the z scores stored as a dictionary of stimuli, followed by a matrix of the z score data. If `tg` is `True` then it will be nUnits x nTrialGroups x nTimeBins otherwise it will be nUnits x nTimeBins. `time_bin_size` like above is size of the time_bins. I also give an optional chance to input a `window_list` which is formatted as nested lists where each stimulus require two lists of time. Since this is complicated I explain below:
 
 #### Windows
 Each stmiulus needs the baseline period to generate the baseline mean and std. [bslStart, bslEnd]. These times are in relation to the event onset. So do to do the 2 seconds before the stimuli onsets would be [-2,0]. To do 500 to 100 milliseconds would be [-.5, -.1]. Then each stimulus also needs a window. Since some neurons have after-discharges this value can be longer or shorter than the actual stimlus. If I am doing a 5 second stim I could analyze the first 2 seconds [0, 2] or the last 2 [3, 5] or [0, 7] for full stimlus but the 2 seconds after.
 
 So the final window_list would be [[bslS, bslE], [start, end]] or with multiple different stimuli it's best to set to `None` and let the function prompt each stimuli itself
 ###
 
 Returns allP, the dictionary of z scores, normVal a dictionary with mean and std of the baseline/neuron or nan, window used for analysis. Attributes are `allP`, `normVal`, `zwindow`
 
 ```python
 allP, normVal, window= myNeuron.clu_zscore(time_bin_size = 0.05, tg=True, window=None)
 ```

### Latency Data
generating latency values can be tricky. There are multiple strategies. I'm working on generating a permutation based code, but currently I've implemented two different latency styles. For Chase and Young 2007 they use a poisson distribution to check for the first spike which causes deviation from the base rate, lambda. Since many neurons I'm interested have low baseline rates and likly do not follow poisson I also implement a strategy adapted from Mormann et al. 2008 taking first spike amongst events and then taking the median amongst each trial. Stored as an attribute in the class under `ClusterAnalysis.latency` in the form of a nested dict with np.arrays inside.

```python
myNeuron.latency(time_bin_size=0.05, bsl_win=[[-10,-5]], event_win=[[0,10]]) # nested lists are required
```

### Neuron metrics (depth etc)
Once the raw waveform data has been generated we can look for depth, duration, amplitude, and for multishank (only one implementation currently dual stacked H7 Cambridge Neurotech) position. The functions perform a lot of matrix math and list comprehensions to find the different values. These values are stored as attributes in the class: `max_waveform`, `waveform_dur` (the peak-trough duration), `waveform_depth` (corrected for shank if depth give in `set_labels`), `waveform_amps`, `shank_dict` (for stack H7 only currently).

```python
myNeuron.waveform_vals()
myNeuron.gen_wfdf()
```
The dataframe is stored as an attribute `waveform_df`.

### Responsive Neurons
Currently I am using user defined responsive neuron properties. Cutoffs vary (Emmanuel et al. 2021 uses 2.58 for the 99% CI, Chirila et al 2023 uses clustering to generate profiles). For historic reasons this stored in the `plot_z` function, which can be used to just generate the responsive neurons by setting the `plot` flag to `False`. With that out of the way, I use a decorator found in `z_score_decorator.py` to allow for multiple people in the same group set their own z score cutoffs. So that file can be re-formated with desired cutoffs. If only one person is using this code delete the conditional logic and just put in the values. `inhib` for inhibitory, `onset` for onset neurons, `sus` for sustained neurons, `offset` for near end of stimulus and `relief` for after discharge. The first value is desired z score and the second number is the number of bins required to count. So this will depend on the `time_bin_size` set in `clu_zscore`. Once this decorator is set the method can be run. `plot_z` has a few flags, most of them are optional to override the attributes set. So `labels` will pull from the attribute `labels` but can be overridden, `tg` is for trial group (it should match with what was done in `clu_zscore`, and `time_pt` is required for use with the default `sorter_dict` this is not recommend since the default is based on my analysis and not the current analysis. So the key parameter is setting a `sorter_dict`.

#### Creating a sorter_dict
this is a dictionary of the time periods that should be analyzed (in time bins) for each time of response. The general structure would follow:

```python
sorter_dict = {
                "sustained": [zero_point, event_len],
                "relief": [event_len, len(time_bins)],
                "onset": [zero_point, zero_point + time_point],
                "onset-offset": [
                    zero_point,
                    zero_point + time_point,
                    event_len - time_point,
                    event_len + time_point,
                ],
                "inhib": [zero_point, zero_point + time_point],
            }
            
```

The `zero_point` is the time_bin in which the stimulus starts. I prefer plotting with some baseline meaning that the first bin with stimulus is not 0, but for example bin 20. The event_len will be the final bin of the stimulus in this case, eg bin 80. For relief in this example I start at bin 80 and go to the final length of bins for example 100. For onset, I do only a subset of bins etc. For two sub periods you just generate a list of 4 numbers. Currently 4 time points (ie two time periods) are the max you can do for the sorter dict.

#### Running `plot_z`. Once this is done for generating responsive neurons just run as 
```python
myNeuron.plot_z(labels = None, tg=True, sorter_dict=sorter_dict, time_pt=0, plot=False)
```

### Creating the responsive neuron dataframe
Since it is easier to work with a dataframe for mining this data I have the method `gen_respdf` which allows for quality control and response check. It will set two new attributes `resp_neuro_df` which has the responsive neurons, along with which stimuli trial groups they responded to and which time of response and `non_resp_df` which includes neurons that made quality cutoffs but did not respond to the stimuli. 

#### Quality Cutoffs
As stated above there are no hard values or rules for these cutoffs. Of note the python isolation distance is always lower than Matlab's due to the way the values are calculated, but they are nicely correlated. So if reading a paper based on matlab code just test lower values for python. Another type of cutoff often used is isi violations. I tend to use isolation distance, but this function has an optional `isi` flag which requires a fraction rate you'd tolerate (ie 0.02 would be anything less than a 2% violation rate. To shutoff off quality metrics set `qcthres = 0` and leave `isi=None`. Otherwise pick one or both to use.

##### Common Error in qc
If the error len(noise) != len(qc) this is due to the way `sp` is generated from phy curation and how noise vs not noise are labeled. So if this occurs it means that your `qcvalues` no longer align with your noise. Before reporting a bug try rerunning `qcfn()` to realign the lengths of these variables. 
to be explicit:
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


### Plotting
plotting functions are explained in the visualization_ca folder.

# MCA
This is the merged cluster analysis. It is still very much in beta, but the goal will be to merge multiple recordings together and analyze them in parallel when similar conditions are being used across experiments
