import numpy as np
from ClusterAnalysis import ClusterAnalysis

sp = np.load('sp.npy', allow_pickle=True)[()]
eventTimes = np.load('eventTimes.npy', allow_pickle=True)[()]

myNeuron = ClusterAnalysis(sp, eventTimes)

myNeuron.clu_zscore()
myNeuron.plot_z(labels=None, tg=True, sorter_dict=None, time_point=0, plot=True)

allP = myNeuron.allP

assert(type(allP)==dict)
