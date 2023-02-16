# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:41:01 2023

@author: ZacharyMcKenzie
"""

cids1 = sp['cids']
cids2 = sp2['cids']
cids3 = sp3['cids']
cids4 = sp4['cids']

wf1 = myNeuron.wf
wf2 = myNeuron_02.wf
wf3 = myNeuron_3.wf
wf4 = myNeuron_4.wf

desired_spikes1 = np.array([True if x in cids1 else False for x in wf1['F']['ClusterIDs']])
desired_spikes2 = np.array([True if x in cids2 else False for x in wf2['F']['ClusterIDs']])
desired_spikes3 = np.array([True if x in cids3 else False for x in wf3['F']['ClusterIDs']])
desired_spikes4 = np.array([True if x in cids4 else False for x in wf4['F']['ClusterIDs']])

depth1 = myNeuron.waveform_depth
depth2 = myNeuron_02.waveform_depth
depth3 = myNeuron_3.waveform_depth
depth4 = myNeuron_4.waveform_depth


fin_depth1 = depth1[desired_spikes1]
fin_depth2 = depth2[desired_spikes2]
fin_depth3 = depth3[desired_spikes3]
fin_depth4 = depth4[desired_spikes4]

desired_spike_fin = np.concatenate((desired_spikes1, desired_spikes2, desired_spikes3, desired_spikes4))

fin_depth = np.concatenate((fin_depth1, fin_depth2, fin_depth3, fin_depth4))


fig, ax = plt.subplots(figsize=(10,8))
ax.scatter((range(len(fin_depth_total))), -fin_depth_total, color='black', alpha=0.7)
ax.scatter(np.array(range(len(fin_depth_total)))[desired_spike_fin],-fin_depth_total[desired_spike_fin], color='red')
ax.set_ylabel('Depth (um)')
ax.set_xlabel('Unit Number (n=482 total)')
ax.legend(['Non-Responsive, Low Quality n=434','Responsive n=48'])
sns.despine()

plt.figure(dpi=1200)

fin_depth_total = np.concatenate((depth1, depth2, depth3, depth4))
