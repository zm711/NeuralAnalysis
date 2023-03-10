
## Quick List of Attributes
Reminder that some of these require various methods to be called first
As always these are accessed with `.` notation thus `var = myNeuron.var`

`sp`: spike properties

`eventTimes`: stimulus properties

`clu`: array of curated cluster ids

`spikeTimes`: array of spike times

`clusterIDs`: the original cluster ids. Should never be overwritten.

`filename`: filename

`wf`: raw waveform dictionary, keys are `['F']` (fortran ordered) followed by `['ClusterIDs']`, `[spikeTimeKeeps']`

`allP`: dict of z scored by stimulus

`zwindow`: window used for allP

`normVal`: mean/std/nan for baseline for z scored data

`depth`: real depth of the probe

`laterality`: left side or right side of animal

`responsive_neurons`: dictionary of responsive neuron **indices not ids**. By using these as indices into `sp['cids']` the correct ids can be obtained

`resp_neuro_df`: DataFrame of responsive neurons, what and how they are responsive

`non_resp_df`: DataFrame of non_responsive neurons

`labels`: dictionary to translate numeric to string labels

`qc`: contains dictionary of isolation distance `['uQ']` and contamination rate `['cR']` explanation in the `qcfn` code

`isiv` dictionary of isiv `['fp']` is a statistic metric for violation rate. Other metrics are just total and fractional violation rate

`psthvalues`: dictionary of firing rate/ time bin

`raster_window` window used

`time_bin` time bin size used

`latency` dictionary of latency value for first spike after stimulus
