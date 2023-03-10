# NeuralAnalysis
Pipeline for analyzing kilosort/phy data
some functions based off of Nick Steinmetz's matlab code (qcfn, psthfns, psthviewer, isiv, loadsp), others are based on reading of the lit which I've tried to indicate in their functions as comments and here.


## Requirements

I've tested python 3.8-3.10. I've also tested some other packages and these are my current to recreate. I work in spyder so I generate a conda env with its own spyder I had stability issues so I'm sticking with 5.3.3 for now. Using the packages below will prevent any compatibility issues.

```sh
 conda create -n neuralanalysis -c conda-forge python=3.10 spyder=5.3.3 numpy=1.23 pandas=1.5 scipy=1.10 matplotlib h5py=3.8 seaborn=0.12 scikit-learn cython=0.29 sympy=1.11 numba
 ```
 **I'm going to great an environment `yaml` soon**
 
### Caching Functions
The counting algorithms used by `psthfns` are sped up using jit and cached in the `__pycache__` folder for future use. Since they are the core of many functions this is a desired behavior. It can be turned off by looking for `@jit(nopython=True, cache=True)` and changing `cache` to `False`


 
 .

### Plotting
plotting functions are explained in the visualization_ca folder.



# MCA
This is the merged cluster analysis. It is still very much in beta, but the goal will be to merge multiple recordings together and analyze them in parallel when similar conditions are being used across experiments
