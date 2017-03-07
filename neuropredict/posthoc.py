from __future__ import print_function
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import itertools
from collections import Counter

common_fig_size = [9, 9]


def feature_importance_map(feat_imp, method_labels, base_output_path,
                           show_distr = False,
                           plot_title = 'feature importance',
                           feat_labels_given=None):
    """

    Generates a map/barplot of feature importance.


    feat_imp must be a list of length num_datasets,
        each an ndarray of size [num_repetitions,num_features[idx]]
        where num_features[idx] refers to the dimensionality of n-th dataset.

    metho_names must be a list of titles of the same length as feat_imp.
    feat_labels must be a list of same size as feat_imp,
        each element being another list of labels corresponding to num_features[idx].

    show_distr, if True, plots the distribution (over different trials of cross-validation)
        of feature importance for each feature.

    """

    num_datasets = len(feat_imp)

    if num_datasets > 1:
        fig, ax = plt.subplots(int(np.ceil(num_datasets/2.0)), 2,
                               sharey=True,
                               figsize=[12, 9])
        ax = ax.flatten()
    else:
        fig, ax_h = plt.subplots(figsize=[12, 9])
        ax = [ax_h] # to support indexing

    for dd in range(num_datasets):

        num_features = feat_imp[dd].shape[1]
        feat_ticks = range(num_features)
        if feat_labels_given is None:
            feat_labels = [ "f{}".format(ix) for ix in feat_ticks]
        else:
            feat_labels = feat_labels_given[dd]
            assert len(feat_labels)==num_features

        plt.sca(ax[dd])
        if show_distr:
            line_coll = ax[dd].violinplot(feat_imp[dd], widths=0.8, bw_method=0.2,
                                      showmedians=True, showextrema=False,
                                      positions=feat_ticks)
            jet = cm.get_cmap('hsv', num_features)
            for cc, ln in enumerate(line_coll['bodies']):
                ln.set_facecolor(jet(cc))
                #ln.set_label(feat_labels[cc])
        else:
            median_feat_imp = np.median(feat_imp[dd], axis=0)
            stdev_feat_imp  = np.nanstd(feat_imp[dd], axis=0)
            barwidth = 8.0 / num_features
            rects = ax[dd].bar(range(num_features), median_feat_imp, width=barwidth, yerr=stdev_feat_imp)

        #plt.legend(loc=2, ncol=num_datasets)

        ax[dd].tick_params(axis='both', which='major', labelsize=15)
        ax[dd].grid(axis='y', which='major')

        ax[dd].set_xticks(feat_ticks)
        ax[dd].set_xlim(np.min(feat_ticks) - 1, np.max(feat_ticks) + 1)
        ax[dd].set_xticklabels(feat_labels, rotation=45)  # 'vertical'
        ax[dd].set_title(method_labels[dd])


    if num_datasets < len(ax):
        fig.delaxes(ax[-1])

    # plt.xlabel(xlabel, fontsize=16)
    plt.suptitle(plot_title, fontsize=16)
    fig.tight_layout()

    base_output_path.replace(' ', '_')
    pp1 = PdfPages(base_output_path + '.pdf')
    pp1.savefig()
    pp1.close()


    return


def display_confusion_matrix(cfmat_array, class_labels,
                             method_names, base_output_path,
                             title='Confusion matrix',
                             cmap=plt.cm.Greens):
    """
    Display routine for the confusion matrix.
    Entries in confusin matrix can be turned into percentages with `display_perc=True`.

    Use a separate method to iteratve over multiple datasets.
    confusion_matrix dime: [num_classes, num_classes, num_repetitions, num_datasets]

    """

    num_datasets = cfmat_array.shape[3]
    num_classes = cfmat_array.shape[0]
    assert num_classes == cfmat_array.shape[1], \
        "Invalid dimensions of confusion matrix. " \
        "Need [num_classes, num_classes, num_repetitions, num_datasets]"

    np.set_printoptions(2)
    for dd in range(num_datasets):
        output_path = base_output_path + '_' + method_names[dd]
        output_path.replace(' ', '_')

        # mean confusion over CV trials
        avg_cfmat = np.mean(cfmat_array[:, :, :, dd], 2)

        # percentage confusion relative to class size
        clsiz_elemwise = np.transpose(np.matlib.repmat(np.sum(avg_cfmat, axis=1), num_classes, 1))
        cfmat = np.divide(avg_cfmat, clsiz_elemwise)
        # human readable in 0-100%, 3 deciamls
        cfmat = 100 * np.around(cfmat, decimals=3)

        fig, ax = plt.subplots(figsize=common_fig_size)

        im = plt.imshow(cfmat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=45)
        plt.yticks(tick_marks, class_labels)

        # trick from sklearn
        thresh = 100.0 / num_classes  # cfmat.max() / 2.
        for i, j in itertools.product(range(num_classes), range(num_classes)):
            plt.text(j, i, "{}%".format(cfmat[i, j]),
                     horizontalalignment="center", fontsize=14,
                     color="tomato" if cfmat[i, j] > thresh else "teal")

        plt.tight_layout()
        plt.ylabel('True class')
        plt.xlabel('Predicted class')

        fig.tight_layout()

        pp1 = PdfPages(output_path + '.pdf')
        pp1.savefig()
        pp1.close()

    return


def summarize_misclassifications(num_times_misclfd, num_times_tested, method_labels, outpath):
    """
    Summary of most/least frequently mislcassified subjects for further analysis

    """

    # TODO capture all the constants in various methods to a single cfg file
    num_bins = 20
    count_thresh = 0.6
    highlight_thresh50 = 0.5
    highlight_thresh75 = 0.75

    num_datasets = len(num_times_tested)
    perc_misclsfd = [None]*num_datasets
    for dd in range(num_datasets):
        perc_misclsfd[dd] = dict()
        for sid in num_times_misclfd[dd].keys():
            if num_times_misclfd[dd][sid] > 0 and num_times_tested[dd][sid] > 0:
                perc_misclsfd[dd][sid] = np.float64(num_times_misclfd[dd][sid]) / np.float64(num_times_tested[dd][sid])

    # plot histograms per dataset
    if num_datasets > 1:
        fig, ax = plt.subplots(int(np.ceil(num_datasets/2.0)), 2,
                               sharey=True,
                               figsize=[12, 9])
        ax = ax.flatten()
    else:
        fig, ax_h = plt.subplots(figsize=[12, 9])
        ax = [ax_h] # to support indexing

    for dd in range(num_datasets):
        plt.sca(ax[dd])
        ax[dd].hist(perc_misclsfd[dd].values(), num_bins)

        cur_ylim = ax[dd].get_ylim()
        line50, = ax[dd].plot([highlight_thresh50, highlight_thresh50],
                              cur_ylim, 'b')
        line75, = ax[dd].plot([highlight_thresh75, highlight_thresh75],
                              cur_ylim, 'r--')
        ax[dd].set_ylim(cur_ylim)
        ax[dd].set_title(method_labels[dd])
        ax[dd].set_ylabel('number of subjects')
        ax[dd].set_xlabel('percentage of misclassification')

        most_freq_misclfd = [ sid for sid in perc_misclsfd[dd].keys() if perc_misclsfd[dd][sid] > count_thresh ]
        txt_path = '_'.join([outpath, method_labels[dd], 'ids_most_frequent.txt'])
        with open(txt_path, 'w') as mfm:
            mfm.writelines('\n'.join(most_freq_misclfd))

    if num_datasets < len(ax):
        fig.delaxes(ax[-1])

    fig.tight_layout()

    pp1 = PdfPages(outpath + '_frequency_histogram.pdf')
    pp1.savefig()
    pp1.close()


    return


def visualize_metrics(metric, labels, output_path, num_classes=2, metric_label='balanced accuracy'):
    """

    Distribution plots of various metrics such as balanced accuracy!

    metric is expected to be ndarray of size [num_repetitions, num_datasets]

    """

    num_repetitions = metric.shape[0]
    num_datasets = metric.shape[1]
    assert len(labels) == num_datasets, "Differing number of features and labels!"
    method_ticks = 1.0 + np.arange(num_datasets)

    fig, ax = plt.subplots(figsize=common_fig_size)
    line_coll = ax.violinplot(metric, widths=0.8, bw_method=0.2,
                              showmedians=True, showextrema=False,
                              positions=method_ticks)

    jet = cm.get_cmap('hsv', num_datasets)
    for cc, ln in enumerate(line_coll['bodies']):
        ln.set_facecolor(jet(cc))
        ln.set_label(labels[cc])

    plt.legend(loc=2, ncol=num_datasets)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(axis='y', which='major')

    lower_lim = np.float64(0.9 / num_classes)
    upper_lim = 1.01
    step_tick = 0.1
    ax.set_ylim(lower_lim, upper_lim)

    ax.set_xticks(method_ticks)
    ax.set_xlim(np.min(method_ticks) - 1, np.max(method_ticks) + 1)
    ax.set_xticklabels(labels, rotation=45)  # 'vertical'

    ax.set_yticks(np.arange(lower_lim, upper_lim, step_tick))
    ax.set_yticklabels(np.arange(lower_lim, upper_lim, step_tick))
    # plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(metric_label, fontsize=16)

    fig.tight_layout()

    pp1 = PdfPages(output_path + '.pdf')
    pp1.savefig()
    pp1.close()

    return


def stat_comparison(clf_results):
    "Non-parametric statistical comparison of different feature sets"

    # TODO implement

    pass


if __name__ == '__main__':
    pass
