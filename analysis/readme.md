# `psthfns`

## Core function
`time_stamps_to_bins`
This function takes in an array of spike_times labeled as time_stamps and an array of events labeled as `reference_points`. It also takes in `start` and `stop`.
From this information it uses `np.histogram` to calculate the counts of spikes which occurred within the user specified time from the events. This function is
numba jit compiled with `nopython=True` and `cache=True`. This is because this is the most used function in the whole pipeline so saving a compiled cache will
speed up all future analysis. O(nxm) speed

# `cluzscore`

This is just wrapper to access `psthfuntions` with both a baseline time period and the stimulus time periods. The baseline is used to create a mean and std for a
neuron which is then used to z-score each neuron. Neurons which are non-zscoreable are saved as raw spikes/second. this function creates `normVal` in order to keep
track of these means and stds or nan if non-zscoreable.

# `latency_calculator`

Uses Chase and Young 2007 and Mormann et al. 2008 to calculate latency to spike in relationship to Stimuli. Briefly Chase and Young suggest use of a Poisson distribution
and scanning along time bins to determine the first statistical deviation from this distribution. Mormann 2008 notes that low firing rate neurons often deviate from
Poisson and notes that a low firing rates likely indicate that the first spike is actually in response to the stimulus. So for neurons <2Hz we instead check for the
first spike and take the median value of these first spikes. It excludes neurons which have too much variability.

# `prevalence_calculator`
Just looks at the resp_df and gives numbers based on a hierarchy of responses. Logic is basically sustained-onset/offset-onset; inhibitory; relief. A new can only be one
of the first three, but the other two categories are not mutually exclusive.

# `firingratedf`
Same as `cluzscore`, but instead does a series of sub windows with the raw firing rate rather than z scored.
