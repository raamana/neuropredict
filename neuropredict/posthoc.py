from __future__ import print_function
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter

def display_confusion_matrix(cfmat, class_labels,
                             title='Confusion matrix',
                             display_perc=False,
                             cmap=plt.cm.Blues):
    """
    Display routine for the confusion matrix.
    Entries in confusin matrix can be turned into percentages with `display_perc=True`.
    """

    plt.imshow(cfmat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    if display_perc:
        cfmat = cfmat.astype('float') / cfmat.sum(axis=1)[:, np.newaxis]

    # trick from sklearn
    thresh = cfmat.max() / 2.
    for i, j in itertools.product(range(cfmat.shape[0]), range(cfmat.shape[1])):
        plt.text(j, i, cfmat[i, j],
                 horizontalalignment="center",
                 color="white" if cfmat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')

    return


def summarize_misclassifications(misclf_stats):
    "Summary of most/least frequently mislcassified subjects for further analysis"

    pass

def visualize_metrics():
    "Distribution plots of various metrics"

    pass


def stat_comparison(clf_results):
    "Non-parametric statistical comparison of different feature sets"

    pass



if __name__ == '__main__':
    pass