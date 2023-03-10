 # An introduction to NeuralAnalysis Module
 
 ## General Notes

1. When I orginally designed this class it was largely based on dictionaries, but I find that dataframes are actually better for a lot of analyses, but this requires translation functions from my original dictionary structures (which all my plotting functions were writtened based on--although I often convert to dataframes within the plotting functions themselves, so maybe eventually I'll remove the dictionaries altogether). 

2. The goal will be to stack all analyses into large final dataframes, which are indexed by what I call the HashID. Since kilosort and phy always just give the same numbers between recordings I take the hash of the filename with the cluster number to generate a unique id for each neuron for each recording. This allows me to interact with multiple datasets while keeping track of unique ids. Haven't had any hash collisions yet.

3. With these caveats I save both the dictionaries and dataframes so either structure can be used for post-hoc analyses
