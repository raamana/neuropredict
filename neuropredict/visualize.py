from __future__ import print_function, division

__all__ = ['feature_importance_map', 'confusion_matrices',
           'freq_hist_misclassifications', 'metric_distribution',
           'compare_misclf_pairwise_parallel_coord_plot', 'compare_misclf_pairwise', ]

import itertools
import warnings
from sys import version_info

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages

if version_info.major==2 and version_info.minor==7:
    import config_neuropredict as cfg
elif version_info.major > 2:
    from neuropredict import config_neuropredict as cfg
else:
    raise NotImplementedError('neuropredict supports only 2.7 or Python 3+. Upgrade to Python 3+ is recommended.')

def feature_importance_map(feat_imp,
                           method_labels,
                           base_output_path,
                           feature_names = None,
                           show_distr = False,
                           plot_title = 'feature importance'):
    """
    Generates a map/barplot of feature importance.

    feat_imp must be a list of length num_datasets,
        each an ndarray of size [num_repetitions,num_features[idx]]
        where num_features[idx] refers to the dimensionality of n-th dataset.

    metho_names must be a list of strings of the same length as feat_imp.
    feature_names must be a list (of ndarrays of strings) same size as feat_imp,
        each element being another list of labels corresponding to num_features[idx].

    show_distr, if True, plots the distribution (over different trials of cross-validation)
        of feature importance for each feature.

    """

    num_datasets = len(feat_imp)

    if num_datasets > 1:
        fig, ax = plt.subplots(num_datasets, 1,
                               sharex=True,
                               figsize=[9, 12])
        ax = ax.flatten()
    else:
        fig, ax_h = plt.subplots(figsize=[9, 12])
        ax = [ax_h] # to support indexing

    for dd in range(num_datasets):

        num_features = feat_imp[dd].shape[1]
        feat_ticks = range(num_features)
        if feature_names is None:
            feat_labels = [ "f{}".format(ix) for ix in feat_ticks]
        else:
            feat_labels = feature_names[dd]
            assert len(feat_labels)==num_features

        plt.sca(ax[dd])
        # violin distribution or stick bar plot?
        if show_distr:
            line_coll = ax[dd].violinplot(feat_imp[dd],
                                          positions=feat_ticks,
                                          widths=0.8, bw_method=0.2,
                                          vert=False,
                                          showmedians=True, showextrema=False)
            cmap = cm.get_cmap(cfg.CMAP_FEAT_IMP, num_features)
            for cc, ln in enumerate(line_coll['bodies']):
                ln.set_facecolor(cmap(cc))
                #ln.set_label(feat_labels[cc])
        else:
            median_feat_imp = np.median(feat_imp[dd], axis=0)
            stdev_feat_imp  = np.nanstd(feat_imp[dd], axis=0)
            barwidth = 8.0 / num_features
            rects = ax[dd].barh(feat_ticks, median_feat_imp,
                               height=barwidth, xerr=stdev_feat_imp)

        #plt.legend(loc=2, ncol=num_datasets)

        ax[dd].tick_params(axis='both', which='major', labelsize=10)
        ax[dd].grid(axis='x', which='major')

        ax[dd].set_yticks(feat_ticks)
        ax[dd].set_ylim(np.min(feat_ticks) - 1, np.max(feat_ticks) + 1)
        ax[dd].set_yticklabels(feat_labels) #, rotation=45)  # 'vertical'
        ax[dd].set_title(method_labels[dd])


    if num_datasets < len(ax):
        fig.delaxes(ax[-1])

    plt.xlabel('feature importance', fontsize=14)
    # plt.suptitle(plot_title, fontsize=16)
    fig.tight_layout()

    base_output_path.replace(' ', '_')
    pp1 = PdfPages(base_output_path + '.pdf')
    pp1.savefig()
    pp1.close()


    return


def confusion_matrices(cfmat_array, class_labels,
                       method_names, base_output_path,
                       cmap=cfg.CMAP_CONFMATX):
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

        fig, ax = plt.subplots(figsize=cfg.COMMON_FIG_SIZE)

        im = plt.imshow(cfmat, interpolation='nearest', cmap=cmap)
        plt.title(method_names[dd])
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


def compute_pairwise_misclf(cfmat_array):
    "Merely computes the misclassification rates, for pairs of classes."

    num_datasets = cfmat_array.shape[3]
    num_classes  = cfmat_array.shape[0]
    num_misclf_axes = num_classes * (num_classes - 1)

    avg_cfmat  = np.full([num_datasets, num_classes, num_classes], np.nan)
    misclf_rate= np.full([num_datasets, num_misclf_axes], np.nan)
    for dd in range(num_datasets):
        # mean confusion over CV trials
        avg_cfmat[dd, :, :] = np.mean(cfmat_array[:, :, :, dd], 2)
        # percentage confusion relative to class size
        clsiz_elemwise = np.transpose(np.matlib.repmat(np.sum(avg_cfmat[dd, :, :], axis=1), num_classes, 1))
        avg_cfmat[dd, :, :] = np.divide(avg_cfmat[dd, :, :], clsiz_elemwise)
        # making it human readable : 0-100%
        avg_cfmat[dd, :, :] = 100 * np.around(avg_cfmat[dd, :, :], decimals=3)

        count = 0
        for ii, jj in itertools.product(range(num_classes), range(num_classes)):
            if ii != jj:
                misclf_rate[dd,count] = avg_cfmat[dd, ii, jj]
                count = count + 1

    return avg_cfmat, misclf_rate


def compare_misclf_pairwise_parallel_coord_plot(cfmat_array, class_labels, method_labels, out_path):
    """
    Produces a parallel coordinate plot (unravelling the cobweb plot) 
    comparing the the misclassfication rate of all feature sets 
    for different pairwise classifications.

    Parameters
    ----------
    cfmat_array
    class_labels
    method_labels
    out_path

    Returns
    -------

    """

    num_datasets = cfmat_array.shape[3]
    num_classes = cfmat_array.shape[0]
    assert num_classes == cfmat_array.shape[1], \
        "Invalid dimensions of confusion matrix. " \
        "Required dims: [num_classes, num_classes, num_repetitions, num_datasets]"

    num_misclf_axes = num_classes * (num_classes - 1)

    out_path.replace(' ', '_')

    avg_cfmat, misclf_rate = compute_pairwise_misclf(cfmat_array)

    misclf_ax_labels = list()
    for ii, jj in itertools.product(range(num_classes), range(num_classes)):
        if ii != jj:
            misclf_ax_labels.append("{} --> {}".format(class_labels[ii], class_labels[jj]))

    fig = plt.figure(figsize=cfg.COMMON_FIG_SIZE)
    ax = fig.add_subplot(1, 1, 1)

    cmap = cm.get_cmap(cfg.CMAP_DATASETS, num_datasets)

    misclf_ax_labels_loc = list()
    handles = list()

    misclf_ax_labels_loc = range(1,num_misclf_axes+1)
    for dd in range(num_datasets):
        h = ax.plot(misclf_ax_labels_loc, misclf_rate[dd, :], color=cmap(dd))
        handles.append(h[0])

    ax.legend(handles, method_labels)
    ax.set_xticks(misclf_ax_labels_loc)
    ax.set_xticklabels(misclf_ax_labels)
    ax.set_ylabel('misclassification rate (in %)')
    ax.set_xlabel('misclassification type')
    ax.set_xlim([0.75, num_misclf_axes+0.25])

    fig.tight_layout()

    pp1 = PdfPages(out_path + '.pdf')
    pp1.savefig()
    pp1.close()

    return


def compare_misclf_pairwise_barplot(cfmat_array, class_labels, method_labels, out_path):
    """
    Produces a bar plot comparing the the misclassfication rate of all feature sets for different pairwise
    classifications.
    
    Parameters
    ----------
    cfmat_array
    class_labels
    method_labels
    out_path

    Returns
    -------

    """

    num_datasets = cfmat_array.shape[3]
    num_classes  = cfmat_array.shape[0]
    assert num_classes == cfmat_array.shape[1], \
        "Invalid dimensions of confusion matrix. " \
        "Required dims: [num_classes, num_classes, num_repetitions, num_datasets]"

    num_misclf_axes = num_classes*(num_classes-1)

    out_path.replace(' ', '_')

    avg_cfmat, misclf_rate = compute_pairwise_misclf(cfmat_array)

    misclf_ax_labels = list()
    for ii, jj in itertools.product(range(num_classes), range(num_classes)):
        if ii != jj:
            misclf_ax_labels.append("{} --> {}".format(class_labels[ii], class_labels[jj]))

    fig = plt.figure(figsize=cfg.COMMON_FIG_SIZE)
    ax = fig.add_subplot(1, 1, 1)

    cmap = cm.get_cmap(cfg.CMAP_DATASETS, num_datasets)

    misclf_ax_labels_loc = list()
    handles = list()
    for mca in range(num_misclf_axes):
        x_pos = np.array(range(num_datasets)) + mca * (num_datasets + 1)
        h = ax.bar(x_pos, misclf_rate[:,mca], color=cmap(range(num_datasets)))
        handles.append(h)
        misclf_ax_labels_loc.append(np.mean(x_pos))

    ax.legend(handles[0], method_labels)
    ax.set_xticks(misclf_ax_labels_loc)
    ax.set_xticklabels(misclf_ax_labels)
    ax.set_ylabel('misclassification rate (in %)')
    ax.set_xlabel('misclassification type')

    fig.tight_layout()

    pp1 = PdfPages(out_path + '.pdf')
    pp1.savefig()
    pp1.close()

    return


def compare_misclf_pairwise(cfmat_array, class_labels, method_labels, out_path):
    """
    Produces a cobweb plot comparing the the misclassfication rate of all feature sets for different pairwise
    classifications.
    
    Parameters
    ----------
    cfmat_array
    class_labels
    method_labels
    out_path

    Returns
    -------

    """

    num_datasets = cfmat_array.shape[3]
    num_classes  = cfmat_array.shape[0]
    assert num_classes == cfmat_array.shape[1], \
        "Invalid dimensions of confusion matrix. " \
        "Required dims: [num_classes, num_classes, num_repetitions, num_datasets]"

    num_misclf_axes = num_classes*(num_classes-1)

    out_path.replace(' ', '_')

    avg_cfmat, misclf_rate = compute_pairwise_misclf(cfmat_array)

    misclf_ax_labels = list()
    for ii, jj in itertools.product(range(num_classes), range(num_classes)):
        if ii != jj:
            misclf_ax_labels.append("{} --> {}".format(class_labels[ii], class_labels[jj]))

    theta = 2 * np.pi * np.linspace(0, 1 -1.0/num_misclf_axes, num_misclf_axes)

    fig = plt.figure(figsize=[9, 9])
    cmap = cm.get_cmap(cfg.CMAP_DATASETS, num_datasets)

    ax = fig.add_subplot(1, 1, 1, projection='polar')
    # clock-wise
    ax.set_theta_direction(-1)
    # starting at top
    ax.set_theta_offset(np.pi / 2.0)

    lw_polar = 2.5
    for dd in range(num_datasets):
        ax.plot(theta, misclf_rate[dd,:], color= cmap(dd), linewidth=lw_polar)
        # connecting the last axis to the first to close the loop
        ax.plot([theta[-1], theta[0]],
                [misclf_rate[dd, -1], misclf_rate[dd, 0]],
                color=cmap(dd), linewidth=lw_polar)

    ax.set_thetagrids(theta * 360/(2*np.pi),
                      labels=misclf_ax_labels,
                      va = 'top',
                      ha = 'center')

    tick_perc = [ '{:.2f}%'.format(tt) for tt in ax.get_yticks() ]
    ax.set_yticklabels(tick_perc)
    # ax.set_yticks(np.arange(100 / num_classes, 100, 10))
    leg = ax.legend(method_labels, loc=2)
    # setting colors manually as plot has been through arbitray jumps
    for ix, lh in enumerate(leg.legendHandles):
        lh.set_color(cmap(ix))

    fig.tight_layout()

    pp1 = PdfPages(out_path + '.pdf')
    pp1.savefig()
    pp1.close()

    return


def compute_perc_misclf_per_sample(num_times_misclfd, num_times_tested):
    "Utility function to compute subject-wise percentage of misclassification."

    num_samples   = len(num_times_tested[0].keys())
    num_datasets  = len(num_times_tested)
    perc_misclsfd = [None]*num_datasets
    never_tested  = list() # since train/test samples are same across different feature sets
    for dd in range(num_datasets):
        perc_misclsfd[dd] = dict()
        for sid in num_times_misclfd[dd].keys():
            if num_times_tested[dd][sid] > 0:
                perc_misclsfd[dd][sid] = np.float64(num_times_misclfd[dd][sid]) / np.float64(num_times_tested[dd][sid])
            else:
                never_tested.append(sid)

    never_tested = list(set(never_tested))

    return perc_misclsfd, never_tested, num_samples, num_datasets


def freq_hist_misclassifications(num_times_misclfd, num_times_tested, method_labels, outpath,
                                 separate_plots = False):
    """
    Summary of most/least frequently misclassified subjects for further analysis

    """

    num_bins = cfg.MISCLF_HIST_NUM_BINS
    count_thresh = cfg.MISCLF_PERC_THRESH

    def annnotate_plots(ax_h):
        "Adds axes labels and helpful highlights"

        highlight_thresh50 = 0.5
        highlight_thresh75 = 0.75

        cur_ylim = ax_h.get_ylim()
        # line50, = ax_h.plot([highlight_thresh50, highlight_thresh50],
        #                     cur_ylim, 'k', linewidth=cfg.MISCLF_HIST_ANNOT_LINEWIDTH)
        # line75, = ax_h.plot([highlight_thresh75, highlight_thresh75],
        #                     cur_ylim, 'k--', linewidth=cfg.MISCLF_HIST_ANNOT_LINEWIDTH)
        line_thresh, = ax_h.plot([count_thresh, count_thresh],
                            cur_ylim, 'k--', linewidth=cfg.MISCLF_HIST_ANNOT_LINEWIDTH)
        ax_h.set_ylim(cur_ylim)
        ax_h.set_ylabel('number of subjects')
        ax_h.set_xlabel('percentage of misclassification')

    # computing the percentage of misclassification per subject
    perc_misclsfd, never_tested, num_samples, num_datasets = compute_perc_misclf_per_sample(num_times_misclfd, num_times_tested)

    if len(never_tested) > 0:
        warnings.warn(' {} subjects were never selected for testing.'.format(len(never_tested)))
        nvpath = outpath + '_never_tested_samples.txt'
        with open(nvpath,'w') as nvf:
            nvf.writelines('\n'.join(never_tested))

    # plot frequency histogram per dataset
    if num_datasets > 1 and separate_plots:
        fig, ax = plt.subplots(int(np.ceil(num_datasets/2.0)), 2,
                               sharey=True,
                               figsize=[12, 9])
        ax = ax.flatten()
    else:
        fig, ax_h = plt.subplots(figsize=[12, 9])

    for dd in range(num_datasets):
        # calculating percentage of most frequently misclassified subjects in each dataset
        most_freq_misclfd = [sid for sid in perc_misclsfd[dd].keys() if perc_misclsfd[dd][sid] > count_thresh]
        perc_most_freq_misclsfd = 100*len(most_freq_misclfd) / num_samples
        this_method_label = "{} - {:.1f}% ".format(method_labels[dd], perc_most_freq_misclsfd)
        if dd == 0:
            this_method_label = this_method_label + 'most frequently misclassfied'

        if num_datasets > 1 and separate_plots:
            ax_h = ax[dd]
            plt.sca(ax_h)
            ax_h.hist(perc_misclsfd[dd].values(), num_bins)  # label = method_labels[dd]
        else:
            # TODO smoother kde plots?
            ax_h.hist(perc_misclsfd[dd].values(), num_bins,
                      histtype = 'stepfilled', alpha = cfg.MISCLF_HIST_ALPHA,
                      label = this_method_label)

        if num_datasets > 1 and separate_plots:
            ax_h.set_title(this_method_label)
            annnotate_plots(ax_h)
        else:
            if dd == num_datasets-1:
                ax_h.legend(loc=2)
                annnotate_plots(ax_h)

        txt_path = '_'.join([outpath, method_labels[dd], 'ids_most_frequent.txt'])
        with open(txt_path, 'w') as mfm:
            mfm.writelines('\n'.join(most_freq_misclfd))

    if separate_plots and num_datasets < len(ax):
        fig.delaxes(ax[-1])

    fig.tight_layout()

    pp1 = PdfPages(outpath + '_frequency_histogram.pdf')
    pp1.savefig()
    pp1.close()


    return


def metric_distribution(metric, labels, output_path, num_classes=2, metric_label='balanced accuracy'):
    """

    Distribution plots of various metrics such as balanced accuracy!

    metric is expected to be ndarray of size [num_repetitions, num_datasets]

    """

    num_repetitions = metric.shape[0]
    num_datasets = metric.shape[1]
    assert len(labels) == num_datasets, "Differing number of features and labels!"
    method_ticks = 1.0 + np.arange(num_datasets)

    fig, ax = plt.subplots(figsize=cfg.COMMON_FIG_SIZE)
    line_coll = ax.violinplot(metric, widths=0.8, bw_method=0.2,
                              showmedians=True, showextrema=False,
                              positions=method_ticks)

    cmap = cm.get_cmap(cfg.CMAP_DATASETS, num_datasets)
    for cc, ln in enumerate(line_coll['bodies']):
        ln.set_facecolor(cmap(cc))
        ln.set_label(labels[cc])

    plt.legend(loc=2, ncol=num_datasets)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(axis='y', which='major')

    lower_lim = np.round(np.min([ np.float64(0.9 / num_classes), metric.min() ]), 3)
    upper_lim = np.round(np.max([ 1.01, metric.max() ]), 3)
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

    # TODO later: as the implementation will need significant testing!

    pass


if __name__ == '__main__':
    pass
