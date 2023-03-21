# Subanalyzing Data

`ClusterAnalysis` works on a no data deletion policy for running subanalysis. Instead the class internally masks data which does not qualify for the subanalysis.
This can be done based on responsiveness or based on quality metrics. This masking is done by saving an initial list of all `cids` when initializing an instance
of the `ClusterAnalysis` class which is never touched.

## Function Inputs
All methods in `ClusterAnalysis` and `MCA` use `sp['cids']` in order to iterate through the clusters which need to be analyzed. This means that by changing the
values found in `sp['cids']` we control which clusters/units are being analyzed. Thus when performing a subanalysis we change the np.array in `sp['cids']` instead
of touching any raw data. Thus `sp['spikeTimes']`, `sp['clu']`, etc are never altered. These are the permanent input data for any analysis. 

## QC Values
Since each unit is analyzed for isolation distance, simplified silhouette score, and interspike violations we can actually generate a list of `cids` which are of appropriate quality. This is
done with the `qc_only` method which will return a dataframe and will also automatically reload `sp['cids']` with the ids which are of appropriate desired quality. Parameters are `qcthres` given as a float > 0. `sil` is a value between `[-1, 1]` (reminder -1 indicates bad clustering and 1 indicates good clusters). and `isi` which is the fraction violation rate (*i.e.* 0.02 would be accepting a 2% violation rate).

## Responsiveness
If a user wants to perform the subanalysis based upon user defined resonsiveness this is also easily accomplished. In this case the `na_settings.yaml` file should be
edited to include the desired z scores and number of times bins of those scores. Once this setting has been adjusted the user can generate z scores with `clu_zscore` 
followed by `plot_z(plot=False)`. Finally `gen_respdf` will only take ids of responsive neurons. Of note this method takes optional qc values at the same time `qcthres`
`sil` and `isi` so that all responsive neurons can be taken or only responsive neurons of high enough quality.

## Reverting
Since we do not alter the raw data `spike_times`, `clu`, etc. In order to revert back to analyze all spikes. all clusters the `revert_cids` method can be used. This
will grab the class attribute `clusterIDs` which has the original ids and will load these back into `sp` allowing for the mask to be deleted.
