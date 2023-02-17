# Generating Stimulus Data

## Parts of stimulus data

### Channel
Intan allows for a variety of digital and analog channels. Most amplifiers/DACs will accept digital and analog inputs. The first layer of creating our stimulus dictionary
is to have a separate key for each channel, eg `DIG1`, `DIG2`, `ADC1` etc. For Intan datasets the `load_intan_rhd_format` can be run, which will generate `amplifier_data`
which is the recording data, but will also have `board_dig_in_data` which is a numpy matrix of the digital channels as well as `board_adc_in_data` which is an numpy matrix 
of the analog data.

#### Digital
Since digital channels are just booleans based on their own TTL logic they are pretty easy to process. Then input should be an (nSamples,)
np.array. For how to process data like this you can look in intan_helpers in the `stimulushelpers.py` file at the `calculate_binary` function. This requires that the boolean
logic be `1` and `0`s. It checks for both unsigned and signed ints. Similar code could be written for a `bool` style with `True` and `False` instead.

#### Analog
ADC channels will typically require some kind of processes. Although it would be possible to analyze this type of data using a GLM (see Pillow Lab Princeton for educational material)
our strategy is to digitize these events. We use a threshold system to check for changes above the cutoffs of our analog voltages and start are onset time based on this value.
My pipeline is currently based on this type of digitization of the ADC signals rather than trying to use a GLM.

### Onsets
`eventTimes['DIG1']['EventTime']`
Each stimulus will have a series of events so each array will contain `nEvents`. This is an np.array of all events for that stimulus regardless of intensity, positioning etc. This is just the start time of the stimulus.

### Lengths
`eventTimes['DIG1']['Lengths']`
Each stimumlus will have a series of events `nEvents`. Each `event` will have corresponding length stored in this np.array. Likely will be similar lengths for this allows for variability in stimuli as recording by the DAC/amplifier

### Trial Groups
`eventTimes['DIG1']['TrialGroup']
Each stimulus will have a series of events `nEvents`. Each `event` could have a different intensity, orientation, positioning etc. If all stimuli were exactly the same this could be an np.array of 1s of len `nEvents`. Otherwise use of ints or floats can be used to distinguish the values. E.g. 0 for 0 degrees, 1 for 90 degrees, 2 for 180 degrees, 3 for 270 degrees. In the `ClusterAnalysis` class a `labels` attribute allows for translating between the numbers and the labels. I use numbers here since I often sort based on the numbering (if 1 is low intensity for 5 being high intensity it is easier to sort than sorting strings which are variable '1mW' '5mW')

### Stim Name
`eventTimes['DIG1']['Stim']
Just a string of the stim name for plotting. Not crucial, but for multiple stimuli it is nicer to keep track of instead of just printing 'DIG1' vs 'DIG2'

### Rest
I don't use this any more, but currently my pipeline creates this. Not necessary if making your own functions.

