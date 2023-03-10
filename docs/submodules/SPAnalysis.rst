# SPAnalysis
This is the spike property only analysis. This is a quick way to load the Kilosort/Phy output numpy structures to examine the raw data. Also has the capabilities to generate qcvalues, isiv's, and raw waveform data. **It does not require stimulus data to function**

#### Method Return philosophy
For this class object I always return the attribute to make it easier to do *post-hoc* inspection. If you are simultaneously working on a `ClusterAnalysis` with the same data you can easily load the `return` to the appropriate attribute. *e.g.*
```python
myNeuron.wf = wf
```
or if concerns about just using reference a deep copy could be used.
```python
myNeuron.wf = copy.deepcopy(wf)
```

## Initializing
A file path can optionally be given otherwise the `SPAnalysis` class can be initialized with no inputs:

```python
spikes = SPAnalysis()
```

To load the `sp` dictionary into an instance of the class the `loadsp` method can be called.

```python
sp = spikes.loadsp()
```
## QC

The same `qcfn` is available. It will store and return `qcvalues` in `self.qc` as well as `isiv` in `isiv`. `isi` is the minimum interspike interval of the neurons and the `ref_dur` is the refractory period as above. 
```python
qcvalues, isiv = spikes.qcfn()
```

## Waveforms

Raw waveforms can be generated using the `get_waveforms` method call. It requires `num_chans` as above (see explanation above).

## Plotting Functions

For images and explanations in depth of plotting see the `readme.md` in the `visualization_ca` folder. In short the following plotting can be used
`plot_wfs`, `acg`, `plot_pc`, `plot_drift`, and `plot_cdf`.
