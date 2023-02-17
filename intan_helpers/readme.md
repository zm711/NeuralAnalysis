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
Each stimulus will have a series of events so each array will contain `nEvents`. 

