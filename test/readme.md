# Testing

I'm including a couple test files. One `sp.npy` will have a prepackaged sp file set up with all the appropriate dictionary structures and one `eventTimes.npy` an example of how stimulus data should be set up. Running the test.py file will automatically load these files and create an instance of the ClusterAnalysis class. It will attempt various assertions which should all pass. 

## Testing different functions yourself
To test the class. Unfortunately the nature of the `.npy` dictionary structure requires pickling. Raw files are typically. The raw file for generating the data are typically minimum of 15gb, so I can't provide the raw intan file or the binary file used for kilosort. Instead I'm including the preprocessed files
```python
import numpy as np
from ClusterAnalysis import ClusterAnalysis
sp = np.load('sp.npy', allow_pickle=True)[()]
eventTimes = np.load('eventTimes.npy', allow_pickle=True)[()]
```
