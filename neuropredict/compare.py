
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import friedmanchisquare, rankdata, norm
from scipy.special import gammaln

def pairwise(accuracy_balanced, method_names, out_results_dir, num_repetitions):
    """
    Produces a matrix of pair-wise significance tests,
        where each cell [i, j] answers the question:
        is method i significantly better than method j?

        The result would be based on a test of choice.
        The default test would be non-parametric Friedman test.

    """

    bal_acc_transposed = accuracy_balanced.T # num_datasets x num_rep (each CV rep is considered a performance measurement on a new dataset, although not independent from each other)

    num_datasets = len(method_names)
    median_bal_acc = np.nanmedian(accuracy_balanced, axis=0)
    ranks = np.rank(median_bal_acc)

    critical_dist = compute_critical_dist(ranks)

    signif_matrix = np.full([num_datasets, num_datasets], np.nan)
    for m1, method_name in enumerate(method_names):
        for m2 in range(m1+1, num_datasets+1, 1):
            signif_matrix[m1, m2] = check_if_better(ranks[m1], ranks[m2], critical_dist)

    return signif_matrix


def compute_critical_dist(ranks):
    ""

    pass


def check_if_better(rank_one, rank_two, critical_dist):
    "Checks whether rank1 is greater than rank2 by at least critical dist"

    is_better = rank_one - rank_two >= critical_dist

    return is_better


def vertical_nemenyi_plot(data, num_reps,
                          alpha = 0.05,
                          cmap = plt.cm.Greens):
    """Vertical Nemenyi plot to compare model ranks and show differences."""

    return

