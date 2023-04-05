# NeuralAnalysis
Pipeline for analyzing post-Phy Data (ie, kilosort data, spyking circus data, spikeinterface (if exported to phy)).
some functions based off of Nick Steinmetz's matlab code (qcfn, psthfns, psthviewer, isiv, loadsp), others are based on reading of the literature which I've indicated in 
function comments and in this docs folder. 

There is a submodule `SPAnalysis` for just looking at spike data. Then there are two other submodules `ClusterAnalysis` and `MCA` for analyzing one dataset of spike and stimulus data or multiple datasets, respectively. 

Currently I have helpers built in for analyzing Intan, but I'm actively working on `Neo` support, which thus may become a dependency in the future.

## Requirements

I've tested python 3.8-3.10. I've also tested some other packages and these are my current to recreate. I work in spyder so I generate a conda env with its own spyder I had stability issues so I'm sticking with 5.3.3 for now. Using the packages below will prevent any compatibility issues.

```bash
 conda create -n neuralanalysis -c conda-forge python=3.10 spyder=5.3.3 numpy pandas scipy matplotlib h5py seaborn scikit-learn cython sympy numba
 pip install neo
 ```
 **I'm going to create an environment `yaml` soon--ie don't use the yaml yet.**
 **Also this isn't packaged yet so only with git clone can this package be used**


## todo
fix mca when different stimuli used
