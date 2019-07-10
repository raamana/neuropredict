from __future__ import print_function, division

__all__ = ['feature_importance_map', 'confusion_matrices',
           'freq_hist_misclassifications', 'metric_distribution',
           'compare_misclf_pairwise_parallel_coord_plot', 'compare_misclf_pairwise', ]

import itertools
import warnings
from sys import version_info

import matplotlib.pyplot as plt
import numpy
import numpy.matlib # to force
import numpy as np
import scipy.stats
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages

if version_info.major > 2:
    from neuropredict import config_neuropredict as cfg, rhst
    from neuropredict.utils import chance_accuracy
else:
    raise NotImplementedError('neuropredict requires Python 3+.')


def feature_importance_map(feat_imp,
                           method_labels,
                           base_output_path,
                           feature_names = None,
                           show_distr = False,
                           plot_title = 'feature importance',
                           show_all = False):
    """
        Generates a map/barplot of feature importance.

    feat_imp must be a list of length num_datasets,
        each an ndarray of size [num_repetitions,num_features[idx]]
        where num_features[idx] refers to the dimensionality of n-th dataset.

    metho_names must be a list of strings of the same length as feat_imp.
    feature_names must be a list (of ndarrays of strings) same size as feat_imp,
        each element being another list of labels corresponding to num_features[idx].
        
    Parameters
    ----------
    feat_imp : list
        List of numpy arrays, each of length num_features
    method_labels : list
        List of names for each method (or feature set).
    base_output_path : str
    feature_names : list
        List of names for each feature.
    show_distr : bool
        plots the distribution (over different trials of cross-validation)
        of feature importance for each feature.
    plot_title : str
        Title of the importance map figure.
    show_all : bool
        If true, this will attempt to show the importance values for all the features. 
        Be advised if you have more than 50 features, the figure would illegible.
        The default is to show only few important features (ranked by their median importance), when there is more than 25 features.

    Returns
    -------

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

        scaled_imp = feat_imp[dd]
        # some models do not provide importance values
        is_nan_imp_values = np.isnan(scaled_imp.flatten())
        if np.all(is_nan_imp_values):
            print('unusable feature importance values for {} : '
                  'all NaNs!\n Skipping it.'.format(method_labels[dd]))
            continue

        num_features = feat_imp[dd].shape[1]
        if feature_names is None:
            feat_labels = [ "f{}".format(ix) for ix in num_features]
        else:
            feat_labels = feature_names[dd]
            if len(feat_labels)<num_features:
                raise ValueError('Insufficient number of feature labels.')

        usable_imp, freq_sel, median_feat_imp, stdev_feat_imp, conf_interval \
            = compute_median_std_feat_imp(scaled_imp)

        if num_features > cfg.max_allowed_num_features_importance_map:
            print('Too many (n={}) features detected for {}.\n'
                  'Showing only the top {} to make the map legible.\n'
                  'Use the exported results to plot make importance maps.'
                  ''.format(num_features, method_labels[dd],
                            cfg.max_allowed_num_features_importance_map))
            sort_indices = np.argsort(median_feat_imp)[::-1] # ascending order, then reversing
            selected_idx_display = sort_indices[:cfg.max_allowed_num_features_importance_map]
            usable_imp_display = [ usable_imp[ix] for ix in selected_idx_display ]
            selected_feat_imp  = median_feat_imp[selected_idx_display]
            selected_imp_stdev = stdev_feat_imp[selected_idx_display]
            selected_conf_interval = conf_interval[selected_idx_display]
            selected_feat_names = feat_labels[selected_idx_display]
            effective_num_features = cfg.max_allowed_num_features_importance_map
        else:
            selected_idx_display = None
            usable_imp_display = usable_imp
            selected_feat_imp = median_feat_imp
            selected_imp_stdev= stdev_feat_imp
            selected_conf_interval=conf_interval
            selected_feat_names = feat_labels
            effective_num_features = num_features

        feat_ticks = range(effective_num_features)

        plt.sca(ax[dd])
        # checking whether all features selected equal number of times (needed for violing pl
        # violin distribution or stick bar plot?
        if show_distr:
            line_coll = ax[dd].violinplot(usable_imp_display,
                                          positions=feat_ticks,
                                          widths=0.8, bw_method=0.2,
                                          vert=False,
                                          showmedians=True, showextrema=False)
            cmap = cm.get_cmap(cfg.CMAP_FEAT_IMP, effective_num_features)
            for cc, ln in enumerate(line_coll['bodies']):
                ln.set_facecolor(cmap(cc))
                #ln.set_label(feat_labels[cc])
        else:
            barwidth = max(0.05, min(0.9, 8.0 / effective_num_features))
            rects = ax[dd].barh(feat_ticks, selected_feat_imp,
                               height=barwidth, xerr=selected_conf_interval)

        ax[dd].tick_params(axis='both', which='major', labelsize=10)
        ax[dd].grid(axis='x', which='major')

        ax[dd].set_yticks(feat_ticks)
        ax[dd].set_ylim(np.min(feat_ticks) - 1, np.max(feat_ticks) + 1)
        ax[dd].set_yticklabels(selected_feat_names) #, rotation=45)  # 'vertical'
        ax[dd].set_title(method_labels[dd])
        print()


    if num_datasets < len(ax):
        fig.delaxes(ax[-1])

    plt.xlabel('feature importance', fontsize=14)
    # plt.suptitle(plot_title, fontsize=16)
    fig.tight_layout()

    base_output_path.replace(' ', '_')
    pp1 = PdfPages(base_output_path + '.pdf')
    pp1.savefig()
    pp1.close()
    plt.close()

    return


def mean_confidence_interval(data, confidence=0.95):
    """Computes mean and CI

    From: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data/

    """

    arr = 1.0*np.array(data, dtype=float)
    n  = len(arr)
    mu = np.mean(arr)
    se = scipy.stats.sem(arr)
    h  = se * scipy.stats.t._ppf((1+confidence)/2., n-1)

    return mu, h


def compute_median_std_feat_imp(imp,
                                ignore_value=cfg.importance_value_to_treated_as_not_selected,
                                never_tested_value=cfg.importance_value_never_tested,
                                never_tested_stdev=cfg.importance_value_never_tested_stdev):
    "Calculates the median/SD of feature importance, ignoring NaNs and zeros"

    num_features = imp.shape[1]
    usable_values = list()
    freq_selection = list()
    conf_interval = list()
    median_values = list()
    stdev_values = list()
    for feat in range(num_features):
        index_nan_or_0 = np.logical_or(np.isnan(imp[:, feat]),
                                       np.isclose(ignore_value, imp[:, feat],
                                                  rtol=1e-4, atol=1e-5))
        index_usable = np.logical_not(index_nan_or_0)
        this_feat_values = imp[index_usable, feat].flatten()
        if len(this_feat_values) > 0:
            usable_values.append(this_feat_values)
            freq_selection.append(len(this_feat_values))
            median_values.append(np.median(this_feat_values))
            stdev_values.append(np.std(this_feat_values))
            mean_, CI_sym = mean_confidence_interval(this_feat_values)
            conf_interval.append(CI_sym)
        else:  # never ever selected
            usable_values.append(None)
            freq_selection.append(0)
            median_values.append(never_tested_value)
            stdev_values.append(never_tested_stdev)
            conf_interval.append(never_tested_stdev)

    return usable_values, np.array(freq_selection), \
           np.array(median_values), np.array(stdev_values), np.array(conf_interval)


def confusion_matrices(cfmat_array, class_labels,
                       method_names, base_output_path,
                       cmap=cfg.CMAP_CONFMATX):
    """
    Display routine for the confusion matrix.
    Entries in confusin matrix can be turned into percentages with `display_perc=True`.

    Use a separate method to iteratve over multiple datasets.
    confusion_matrix dime: [num_classes, num_classes, num_repetitions, num_datasets]

    Parameters
    ----------
    cfmat_array
    class_labels
    method_names
    base_output_path
    cmap

    Returns
    -------

    """

    num_datasets = cfmat_array.shape[3]
    num_classes = cfmat_array.shape[1]
    if num_classes != cfmat_array.shape[2]:
        raise ValueError("Invalid dimensions of confusion matrix.\n"
                         "Need [num_classes, num_classes, num_repetitions, num_datasets]")

    np.set_printoptions(2)
    for dd in range(num_datasets):
        output_path = base_output_path + '_' + method_names[dd]
        output_path.replace(' ', '_')

        # mean confusion over CV trials
        avg_cfmat = mean_over_cv_trials(cfmat_array[:, :, :, dd], num_classes)

        # avg_cfmat = np.mean(cfmat_array[:, :, :, dd], 0)
        #
        # # percentage confusion relative to class size
        # class_size_elementwise = np.transpose(np.matlib.repmat(np.sum(avg_cfmat, axis=1), num_classes, 1))
        # cfmat = np.divide(avg_cfmat, class_size_elementwise)
        # # human readable in 0-100%, 3 deciamls
        # cfmat = 100 * np.around(cfmat, decimals=cfg.PRECISION_METRICS)

        fig, ax = plt.subplots(figsize=cfg.COMMON_FIG_SIZE)

        im = plt.imshow(avg_cfmat, interpolation='nearest', cmap=cmap)
        # TODO need a utility compress a super long method name (>50 chars) to < 50 chars
        #   with ... in the middle
        plt.title(method_names[dd])
        plt.colorbar(im, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=45)
        plt.yticks(tick_marks, class_labels)

        thresh = np.percentile(avg_cfmat, 50)
        for i, j in itertools.product(range(num_classes), range(num_classes)):
            plt.text(j, i, "{:.{prec}f}%".format(avg_cfmat[i, j], prec=cfg.PRECISION_METRICS),
                     horizontalalignment="center", fontsize='large',
                     color="blue" if avg_cfmat[i, j] > thresh else "yellow")

        plt.tight_layout()
        plt.ylabel('True class')
        plt.xlabel('Predicted class')

        fig.tight_layout()

        pp1 = PdfPages(output_path + '.pdf')
        pp1.savefig()
        pp1.close()

    plt.close()

    return


def mean_over_cv_trials(conf_mat_array, num_classes):
    """
    Common method to average over different CV trials,
        to ensure it is done over the right axis
        (the first one - axis=0, column 1) for all confusion matrix methods.

    """

    if conf_mat_array.shape[1] != num_classes or \
                    conf_mat_array.shape[2] != num_classes or \
                    len(conf_mat_array.shape) != 3:
        raise ValueError('Invalid shape of confusion matrix array! '
                         'It must be num_rep x {nc} x {nc}'.format(nc=num_classes))

    # can not expect nan's here; If so, its a bug somewhere else
    avg_cfmat = np.mean(conf_mat_array, axis=0)

    # percentage confusion relative to class size
    class_size_elementwise = np.transpose(np.matlib.repmat(np.sum(avg_cfmat, axis=1), num_classes, 1))
    avg_cfmat_perc = np.divide(avg_cfmat, class_size_elementwise)
    # making it human readable : 0-100%
    avg_cfmat_perc100 = 100 * np.around(avg_cfmat_perc, decimals=cfg.PRECISION_METRICS)

    return avg_cfmat_perc100


def compute_pairwise_misclf(cfmat_array):
    "Merely computes the misclassification rates, for pairs of classes."

    num_datasets = cfmat_array.shape[3]
    num_classes = cfmat_array.shape[1]
    if num_classes != cfmat_array.shape[2]:
        raise ValueError("Invalid dimensions of confusion matrix.\n"
                         "Need [num_classes, num_classes, num_repetitions, num_datasets]")

    num_misclf_axes = num_classes * (num_classes - 1)

    avg_cfmat  = np.full([num_datasets, num_classes, num_classes], np.nan)
    misclf_rate= np.full([num_datasets, num_misclf_axes], np.nan)
    for dd in range(num_datasets):
        # mean confusion over CV trials
        avg_cfmat[dd, :, :] = mean_over_cv_trials(cfmat_array[:, :, :, dd], num_classes)

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
    num_classes = cfmat_array.shape[1]
    if num_classes != cfmat_array.shape[2]:
        raise ValueError("Invalid dimensions of confusion matrix.\n"
                         "Need [num_classes, num_classes, num_repetitions, num_datasets]")

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

    plt.close()

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
    num_classes = cfmat_array.shape[1]
    if num_classes != cfmat_array.shape[2]:
        raise ValueError("Invalid dimensions of confusion matrix.\n"
                         "Need [num_classes, num_classes, num_repetitions, num_datasets]")

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

    plt.close()

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
    num_classes = cfmat_array.shape[1]
    if num_classes != cfmat_array.shape[2]:
        raise ValueError("Invalid dimensions of confusion matrix.\n"
                         "Need [num_classes, num_classes, num_repetitions, num_datasets]")

    num_misclf_axes = num_classes*(num_classes-1)

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

    for dd in range(num_datasets):
        ax.plot(theta, misclf_rate[dd,:], color= cmap(dd),
                linewidth=cfg.LINE_WIDTH)
        # connecting the last axis to the first to close the loop
        ax.plot([theta[-1], theta[0]],
                [misclf_rate[dd, -1], misclf_rate[dd, 0]],
                color=cmap(dd), linewidth=cfg.LINE_WIDTH)

    lbl_handles = ax.set_thetagrids(theta * 360/(2*np.pi),
                      labels=misclf_ax_labels,
                      va = 'top',
                      ha = 'center',
                      fontsize=cfg.FONT_SIZE)

    ax.grid(linewidth=cfg.LINE_WIDTH)
    tick_perc = [ '{:.2f}%'.format(tt) for tt in ax.get_yticks() ]
    ax.set_yticklabels(tick_perc, fontsize=cfg.FONT_SIZE)
    # ax.set_yticks(np.arange(100 / num_classes, 100, 10))
    plt.tick_params(axis='both', which='major')

    # putting legends outside the plot below.
    fig.subplots_adjust(bottom=0.2)
    leg = ax.legend(method_labels, ncol=2, loc=9,
                    bbox_to_anchor=(0.5, -0.1))

    # setting colors manually as plot has been through arbitray jumps
    for ix, lh in enumerate(leg.legendHandles):
        lh.set_color(cmap(ix))
    leg.set_frame_on(False)  # making leg background transparent

    fig.tight_layout()

    out_path.replace(' ', '_')
    # fig.savefig(out_path + '.png', transparent=True, dpi=300,
    #             bbox_extra_artists=(leg,), bbox_inches='tight')

    fig.savefig(out_path + '.pdf',
                bbox_extra_artists=(leg,), bbox_inches='tight')

    plt.close()

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

        cur_ylim = ax_h.get_ylim()
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
        perc_most_freq_misclsfd = 100*len(most_freq_misclfd) / len(perc_misclsfd[dd])
        this_method_label = "{} - {:.1f}% ".format(method_labels[dd], perc_most_freq_misclsfd)
        if dd == 0:
            this_method_label = this_method_label + 'most frequently misclassfied'

        # for plotting
        if num_datasets > 1 and separate_plots:
            ax_h = ax[dd]
            plt.sca(ax_h)
            ax_h.hist(perc_misclsfd[dd].values(), num_bins)
        else:
            ax_h.hist(list(perc_misclsfd[dd].values()), num_bins,
                      histtype = 'stepfilled', alpha = cfg.MISCLF_HIST_ALPHA,
                      label = this_method_label)

        # for annotation
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
    plt.close()

    return


def metric_distribution(metric, labels, output_path, class_sizes,
                        num_classes=2, metric_label='balanced accuracy'):
    """

    Distribution plots of various metrics such as balanced accuracy!

    metric is expected to be ndarray of size [num_repetitions, num_datasets]

    """

    num_repetitions = metric.shape[0]
    num_datasets = metric.shape[1]
    if len(labels) < num_datasets:
        raise ValueError("Insufficient number of labels for {} features!".format(num_datasets))
    method_ticks = 1.0 + np.arange(num_datasets)

    fig, ax = plt.subplots(figsize=cfg.COMMON_FIG_SIZE)
    line_coll = ax.violinplot(metric, widths=cfg.violin_width,
                              bw_method=cfg.violin_bandwidth,
                              showmedians=True, showextrema=False,
                              positions=method_ticks)

    cmap = cm.get_cmap(cfg.CMAP_DATASETS, num_datasets)
    for cc, ln in enumerate(line_coll['bodies']):
        ln.set_facecolor(cmap(cc))
        ln.set_label(labels[cc])

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(axis='y', which='major', linewidth=cfg.LINE_WIDTH, zorder=0)

    lower_lim = np.round(np.min([ np.float64(0.9 / num_classes), metric.min() ]), cfg.PRECISION_METRICS)
    upper_lim = np.round(np.min([ 1.01, metric.max() ]), cfg.PRECISION_METRICS)
    step_tick = 0.05
    ax.set_ylim(lower_lim, upper_lim)

    ax.set_xlim(np.min(method_ticks) - 1, np.max(method_ticks) + 1)
    ax.set_xticks(method_ticks)
    # ax.set_xticklabels(labels, rotation=45)  # 'vertical'

    ytick_loc = np.arange(lower_lim, upper_lim, step_tick)
    # add a tick for chance accuracy and/or % of majority class
    # given the classifier trained on stratified set, we must use the balanced version
    chance_acc = chance_accuracy(class_sizes, 'balanced')

    # rounding to ensure improved labels
    chance_acc = np.round(chance_acc, cfg.PRECISION_METRICS)
    ytick_loc = np.round(np.append(ytick_loc, chance_acc), cfg.PRECISION_METRICS)

    ax.set_yticks(ytick_loc)
    ax.set_yticklabels(ytick_loc)
    plt.text(0.05, chance_acc, 'chance accuracy')
    plt.ylabel(metric_label, fontsize=cfg.FONT_SIZE)

    plt.tick_params(axis='both', which='major', labelsize=cfg.FONT_SIZE)

    # numbered labels
    numbered_labels = ['{} {}'.format(int(ix),lbl) for ix, lbl in zip(method_ticks,labels)]

    # putting legends outside the plot below.
    fig.subplots_adjust(bottom=0.2)
    leg = ax.legend(numbered_labels, ncol=2, loc=9, bbox_to_anchor=(0.5, -0.1))
    # setting colors manually as plot has been through arbitray jumps
    for ix, lh in enumerate(leg.legendHandles):
        lh.set_color(cmap(ix))

    leg.set_frame_on(False) # making leg background transparent

    # fig.savefig(output_path + '.png', transparent=True, dpi=300,
    #             bbox_extra_artists=(leg,), bbox_inches='tight')

    fig.savefig(output_path + '.pdf', bbox_extra_artists=(leg,), bbox_inches='tight')

    plt.close()

    return


if __name__ == '__main__':
    pass
