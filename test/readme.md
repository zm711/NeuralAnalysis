# Check_install.py

This file can be downloaded and run to visualize what some of the graphing functions etc will look like and to make sure all packages were correctly downloaded.

Running clu_zscore with baseline window -2,-.1, stimulus window = -2,12 and running plot_z will generate:

![image](https://user-images.githubusercontent.com/92116279/219795937-2fc37781-70ba-4a0b-abb7-3983ff958555.png)
![image](https://user-images.githubusercontent.com/92116279/219795950-4feb8a07-76a2-4c0a-b11f-bd9acb36f0d7.png)

```python
np.shape(allP) should be (6,2,280).
np.sum(allP['Dig2']) == -30.93797084395051
```
The `acg` figure for `Neuron 5` should be:

![image](https://user-images.githubusercontent.com/92116279/219798237-3f878480-8344-490d-ba19-5a5811ea1a3a.png)

# Testing

Testing currently with pytest--working on setting up a Github Action, but I mainly test the mathematical functions/algos as I'm tweaking things to make sure I don't break anything
