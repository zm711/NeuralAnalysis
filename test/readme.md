# Testing

I'm including a couple test files. One `sp.npy` will have a prepackaged sp file set up with all the appropriate dictionary structures and one `eventTimes.npy` an example of how stimulus data should be set up. Running the test.py file will automatically load these files and create an instance of the ClusterAnalysis class. It will attempt various assertions which should all pass. 

## Initial Notes
Raw files are all >5gb and I routinely work on >20gb raw files. So I can't share those here. Since all storage is done in `.npy` files we require a lot of pickling. Since pickle is unsafe I also don't want to require `allow_pickle` to occur during testing. Internal pickle is used, but for external files please don't allow pickle. 

in `test.py` I create a small sample `sp` and `eventTimes`. Unfortunately I can't load all the values into `sp` (we are missing the pc features `pcFeat` and `pcFeatInd`) as well as the whitening matrix `winv`. Finally I am not currently loading `temps` which are the kilosort templates that are automatically loaded when you have your own generated files. 

## Functions That Will Not Work

### `get_waveforms`
This requires the raw 5-20 gb `.bin` file. Since I can't include it, we can't run this function in the test file.

### `qcfn`
This requires the raw pc_files which would require pickling which isn't safe for testing in files that you don't know about. 



## Results

Running clu_zscore with baseline window -2,-.1, stimulus window = -2,12 and running plot_z will generate:

![image](https://user-images.githubusercontent.com/92116279/219795937-2fc37781-70ba-4a0b-abb7-3983ff958555.png)
![image](https://user-images.githubusercontent.com/92116279/219795950-4feb8a07-76a2-4c0a-b11f-bd9acb36f0d7.png)

```python
np.shape(allP) should be (6,2,280).
np.sum(allP['Dig2']) == -30.93797084395051
```
