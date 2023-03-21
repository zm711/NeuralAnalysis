# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:03:08 2023

@author: ZacharyMcKenzie

In short use simplified to reduce from O(n**2) down to O(n). Data is already organized
into this_cluster and nearest_other clusters so no need to search through k clusters. In
short we measure each a(i)=d(i,C_i) where C_i=centroid of i followed by b(i) =d(i, C_j) 
where C_j=centroid of nearest neighboring cluster. We then take the silhouette score for
all i  such that s(i) = b(i)-b=a(i)/(max(a(i), b(i))). Finally I take the mean of s.
"""


import numpy as np
from scipy.spatial.distance import cdist


def silhouette_score(
    fetThisCluster: np.array,
    fetOtherClusters: np.array,
):
    """calculate the simplified silhouette score based on Hruschka et al. 2004 currently
    with euclidean distance since malahobnis fails for singular matrices"""
    md_self = np.squeeze(
        cdist(
            fetThisCluster,
            np.reshape(np.mean(fetThisCluster, axis=0), (1, -1)),
            "euclidean",
        )
    )
    md_other = np.squeeze(
        cdist(
            fetThisCluster,
            np.reshape(np.mean(fetOtherClusters, axis=0), (1, -1)),
            "euclidean",
        )
    )
    md_len = len(md_self)
    md_other = md_other[:md_len]
    sil_scores = (md_other - md_self) / np.maximum(md_self, md_other)

    mean_sil_score = np.mean(sil_scores)
    return mean_sil_score
