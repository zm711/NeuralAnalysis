# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:02:57 2023

@author: ZacharyMcKenzie
"""

import numpy as np
from sklearn import decomposition, cluster, mixture
import matplotlib.pyplot as plt
import seaborn as sns
import time

n_list = list()
variance = list()


if len(np.shape(allP["Barostat"])) == 2:
    X = allP["Barostat"].copy()
else:
    X = (
        allP["Barostat"]
        .copy()
        .reshape(
            np.shape(allP["Barostat"])[0],
            allP["Barostat"].shape[1] * allP["Barostat"].shape[2],
        )
    )


for n in range(1, min(np.shape(X))):
    pc = decomposition.PCA(n_components=n)
    pc.fit(X)

    n_list.append(n)
    variance.append(sum(pc.explained_variance_ratio_))
    if sum(pc.explained_variance_ratio_) >= 0.90:
        n_comp = n
        break

test = pc.components_[0]
test2 = pc.components_[1]

fig10, (ax10, ax11) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax10.plot(test, color="black")
ax11.plot(test2, color="black")
sns.despine()
plt.figure(dpi=1200)

plt.scatter(n_list, variance)
test = "pca"
if test == "pca":
    pca = decomposition.PCA(n_components=n_comp)
    pca.fit(X)
    x_pca = pca.transform(X)
    # test = pca.components_
elif test == "nmf":

    """test non-negative matrix"""
    nmf = decomposition.NMF(n_components=n_comp)
    y = np.where(X < 0, 0, X.copy())
    nmf.fit(X.copy())
    x_pca = nmf.transform(X)

elif test == "ica":
    ica = decomposition.FastICA(n_components=n_comp)
    ica.fit(X)
    x_pca = ica.transform(X)

start = time.time()
start = 200
end = 600
for n in range(2, 8):
    n_clu = n
    kmeans = cluster.KMeans(n_clusters=n_clu, random_state=0, n_init="auto").fit(x_pca)
    gmm = mixture.GaussianMixture(
        n_components=n, random_state=0, covariance_type="diag"
    ).fit(x_pca)
    hierarch = cluster.AgglomerativeClustering(n_clusters=n_clu, linkage="ward").fit(
        x_pca
    )
    spectral = cluster.SpectralClustering(
        n_clusters=n_clu,
        random_state=0,
    ).fit(x_pca)
    dbscan = cluster.DBSCAN(eps=0.005).fit(x_pca)
    # kmeans.labels_
    # kmeans.cluster_centers_

    fig, ax = plt.subplots(n_clu, 1, sharex=True)
    for labels in kmeans.labels_:
        curr_data = np.mean(X[kmeans.labels_ == labels], axis=0)

        ax[labels].plot(
            range(np.shape(X)[1]),
            curr_data,
            color=["blue", "red", "green", "orange", "black", "pink", "yellow"][labels],
        )
        ax[labels].axvline(start, color="k")
        ax[labels].axvline(end, color="k")
        plt.suptitle("K-means")

    fig2, ax2 = plt.subplots(n_clu, 1, sharex=True)
    labels = gmm.predict(x_pca)
    for label in labels:
        curr_data = np.mean(X[labels == label], axis=0)

        ax2[label].plot(
            range(np.shape(X)[1]),
            curr_data,
            color=["blue", "red", "green", "orange", "black", "pink", "yellow"][label],
        )
        ax2[label].axvline(start, color="k")
        ax2[label].axvline(end, color="k")
        plt.suptitle("GMM")

    fig3, ax3 = plt.subplots(n_clu, 1, sharex=True)
    labels = hierarch.labels_
    for label in labels:
        curr_data = np.mean(X[labels == label], axis=0)

        ax3[label].plot(
            range(np.shape(X)[1]),
            curr_data,
            color=["blue", "red", "green", "orange", "black", "pink", "yellow"][label],
        )
        ax3[label].axvline(start, color="k")
        ax3[label].axvline(end, color="k")
        plt.suptitle("Hierarchical Clustering")

    fig4, ax4 = plt.subplots(n_clu, 1, sharex=True)
    labels = spectral.labels_
    for label in labels:
        curr_data = np.mean(X[labels == label], axis=0)

        ax4[label].plot(
            range(np.shape(X)[1]),
            curr_data,
            color=["blue", "red", "green", "orange", "black", "pink", "yellow"][label],
        )
        ax4[label].axvline(start, color="k")
        ax4[label].axvline(end, color="k")
        plt.suptitle("Spectral Clustering")

    fig5, ax5 = plt.subplots(n_clu, 1, sharex=True)
    labels = dbscan.labels_
    for label in labels:
        curr_data = np.mean(X[labels == label], axis=0)

        ax5[label].plot(
            range(np.shape(X)[1]),
            curr_data,
            color=["blue", "red", "green", "orange", "black", "pink", "yellow"][label],
        )
        ax5[label].axvline(start, color="k")
        ax5[label].axvline(end, color="k")
        plt.suptitle("DBSCAN")

print(f"Final time is {time.time()-start} seconds")
