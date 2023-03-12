# Easy Start
If using `Phy` input output files and using `Intan` as your recording equipment. I have one helper function which will automatically load everything.

```python
import neuralanlysis.full as na
sp, eventTimes = na.loadKS()
```

## Collect Neural Data `sp` and Stimulus Data `eventTimes`
This function will ask for the root directory for the `Phy` files and generate `sp` and then after it will ask for the location of `eventTimes.npy`. If
this is the first time running this function it will fail to find the file and then ask for where the `*.rhd` (Intan raw file) is located. From there it
will generate the necessary `intan` file and save it (*i.e.* it will not save `amplifier_data`, but will save the `dig_in` and `adc` data as file in case
you want to access it again later). From this file it will generate `eventTimes` and prompt you for the `Stimulus` name. These values can be loaded into
`ClusterAnalysis` objects.

## Quick load into `ClusterAnalysis`
A quick way to do everything if using `Phy` and Intan would be to just unpack the values directly into an instance of `ClusterAnalysis`:

```python
import neuralanalysis.full as na
my_neuron = na.ClusterAnalysis(*na.loadKS())
```
