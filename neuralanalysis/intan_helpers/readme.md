# Generating Stimulus Data

## Parts of stimulus data

### Channel
Intan allows for a variety of digital and analog channels. Most amplifiers/DACs will accept digital and analog inputs. The first layer of creating our stimulus dictionary
is to have a separate key for each channel, eg `DIG1`, `DIG2`, `ADC1` etc. For Intan datasets the `load_intan_rhd_format` can be run, which will generate `amplifier_data`
which is the recording data, but will also have `board_dig_in_data` which is a numpy matrix of the digital channels as well as `board_adc_in_data` which is an numpy matrix of the analog data. Of note most stimulus recording have a `sample rate` which is the number of samples/sec (1k-30k are typical depending on use case), so data
is typically organized in structures of `nSamples`.

#### Digital
Since digital channels are just booleans based on TTL logic they are pretty easy to process. The input should be an array of `nSamples` or in the case of
multiple digital inputs it may be `nChannels x nSamples` np.array. For Intan, there is a dict with channel information so that each row can be identified with its appropriate channel. For how to process data like this you can look in intan_helpers in the `stimulushelpers.py` file at the `calculate_binary` function. This requires that the boolean logic be `1` and `0`s. It checks for both unsigned and signed ints. Similar code could be written for a `bool` style with `True` and `False` instead.

#### Analog
ADC channels (analog to digital converted) will typically require some kind of processing. Although it would be possible to analyze this type of data using a GLM (see Pillow Lab Princeton for educational material) our strategy is to digitize these events. We use a threshold system to check for changes above the cutoffs of our analog voltages and start our onset time based on this value. My pipeline is currently based on this type of digitization of the ADC signals rather than trying to use a GLM. I also have my barostat stimulation analysis pipeline where there are `20mmHg/1 V`. Thus I digitize my adc with the logic provided in `intan_helpers/stimulussetup.py` in the `barostat_stim_setup` function.

### Onsets
`eventTimes['DIG1']['EventTime']`
Each stimulus will have a series of `events` with a total number `nEvents`. The onsets needs to be an np.array of the start time (in seconds) for all `events` for that stimulus (`DIG1` in the example above) regardless of intensity, positioning etc. My pipeline assumes time rather than samples so technically all values are divided by `sample rate`.

### Lengths
`eventTimes['DIG1']['Lengths']`
Each stimulus will have a series of `events` with number `nEvents`. Each `event` will have corresponding length (seconds) stored in this np.array as a float. Likely a particular stimulus will have similar lengths, but by calculating the length for each allows for variability in stimuli as recording by the DAC/amplifier--'perfect' stimuli could still have variability in the 10-100 millisecond range. 

### Trial Groups
`eventTimes['DIG1']['TrialGroup']`
Each stimulus will have a series of `events` with number `nEvents`. Each `event` could have a different intensity, orientation, positioning etc. If all stimuli were exactly the same this could be an np.array of 1s of len `nEvents`. Otherwise use of ints or floats can be used to distinguish the values. E.g. 0 for 0 degrees, 1 for 90 degrees, 2 for 180 degrees, 3 for 270 degrees. In the `ClusterAnalysis` class a `labels` attribute allows for translating between the numbers and the labels. I use numbers here since I often sort based on the numbering (if 1 is low intensity for 5 being high intensity it is easier to sort than sorting strings which are variable '1mW' '5mW')

### Stim Name
`eventTimes['DIG1']['Stim']`
Just a string of the stim name for plotting. Not crucial, but for multiple stimuli it is nicer to keep track of instead of just printing 'DIG1' vs 'DIG2'. Of note this **is necessary** for my pipeline, so if you want to stay anonymous just write 'DIG1' for example.

### Rest
I don't use this any more, but currently my pipeline creates this. Not necessary if making your own functions.

## Important Alignment Caveat
I typically plug my stimuli directly into the intan recording amplifier to reduce lag. When controlling stimuli via Matlab/Python through a DAC to Hardware and then to Intan or whatever recorder is being used could actually lead to offsets approaching 500 milliseconds between onsets and offsets of stimuli. If this is the case either find a more direct way to have stimuli communicate with the recording apparatus or perform a regression to align timing of stimuli. 
