from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn import cluster

np.set_printoptions(precision=4, suppress=True)


def kolmogorov_smirnov_matrix(spectra: pd.DataFrame):
    ks_stat = np.zeros(shape=(9, 9))
    ks_p = np.zeros(shape=(9, 9))
    for i, j in combinations(range(9), 2):
        ks_calc = ks_2samp(spectra.iloc[:, i], spectra.iloc[:, j])
        ks_stat[i, j] = ks_calc.statistic
        ks_stat[j, i] = ks_calc.statistic
        ks_p[i, j] = ks_calc.pvalue
        ks_p[j, i] = ks_calc.pvalue

    print(ks_stat)

    print(ks_p)

    return ks_stat, ks_p


def kolmogorov_smirnov_clusters(ks):
    cluster_centers_indices, labels = cluster.affinity_propagation(ks)
    n_clusters_ = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % n_clusters_)
    print(cluster_centers_indices)
    print(labels)

    return cluster_centers_indices, labels
