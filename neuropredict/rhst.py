from __future__ import print_function

__all__ = ['run', 'load_results', 'save_results']

import os
import sys
import pickle
import logging
import warnings
from collections import Counter, namedtuple
from sys import version_info
from os.path import join as pjoin, exists as pexists
from multiprocessing import Pool, Manager
from functools import partial
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit, RepeatedKFold
import traceback
import shutil
from pyradigm import MLDataset

if version_info.major > 2:
    from neuropredict import config_neuropredict as cfg
    from neuropredict.algorithms import get_pipeline, get_feature_importance
    from neuropredict.reports import report_best_params, export_results
    from neuropredict.utils import check_feature_sets_are_comparable, check_params_rhst, balanced_accuracy, \
        load_options, sub_group_identifier, make_numeric_labels
    from neuropredict.io import load_pyradigms
else:
    raise NotImplementedError('neuropredict requires Python 3+.')


def eval_optimized_model_on_testset(train_fs, test_fs,
                                    impute_strategy=cfg.default_imputation_strategy,
                                    label_order_in_conf_matrix=None,
                                    feat_sel_size=cfg.default_num_features_to_select,
                                    train_perc=0.5,
                                    grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT,
                                    classifier_name=cfg.default_classifier,
                                    feat_select_method=cfg.default_feat_select_method):
    """
    Optimize the classifier on the training set and return predictions on test set.

    Parameters
    ----------
    train_fs : MLDataset
        Dataset to optimize a given classifier on.

    test_fs : MLDataset
        Dataset to make predictions on using the classifier optimized on training set.

    impute_strategy : str
        Strategy to handle the missing data: whether to raise an error if data is missing, or
            to impute them using the method chosen here.

    label_order_in_conf_matrix : list
        List of labels to compute the order of confusion matrix.

    feat_sel_size : str or int
        Metho to choose the number of featurese to select.

    train_perc : float
        Training set fraction to run the inner cross-validation.

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be user for optimization.

    classifier_name : str
        String identifying a scikit-learn classifier.

    feat_select_method : str
        String identifying a valid scikit-learn feature selection method.

    Returns
    -------

    """

    if label_order_in_conf_matrix is None:
        raise ValueError('Label order for confusion matrix must be specified for accurate results/visulizations.')

    train_data_mat, train_labels, _ = train_fs.data_and_labels()
    test_data_mat, true_test_labels, test_sample_ids = test_fs.data_and_labels()

    if impute_strategy is not None:
        train_data_mat, test_data_mat = impute_missing_data(train_data_mat, train_labels,
                                                            impute_strategy, test_data_mat)

    train_class_sizes = list(train_fs.class_sizes.values())

    # TODO look for ways to avoid building this every iter and every dataset.
    pipeline, param_grid = get_pipeline(train_class_sizes,
                                        feat_sel_size,
                                        train_fs.num_features,
                                        grid_search_level=grid_search_level,
                                        classifier_name=classifier_name,
                                        feat_selector_name=feat_select_method)

    best_pipeline, best_params = optimize_pipeline_via_grid_search_CV(pipeline, train_data_mat, train_labels,
                                                                      param_grid, train_perc)
    # best_model, best_params = optimize_RF_via_training_oob_score(train_data_mat, train_labels,
    #     param_grid['random_forest_clf__min_samples_leaf'], param_grid['random_forest_clf__max_features'])

    # assuming order in pipeline construction :
    #   - step 0 : preprocessign (robust scaling)
    #   - step 1 : feature selector
    _, best_fsr = best_pipeline.steps[1]
    _, best_clf = best_pipeline.steps[-1]  # the final step in an sklearn pipeline is always an estimator/classifier

    # could be useful to compute frequency of selection
    index_selected_features = best_fsr.get_support(indices=True)

    # making predictions on the test set and assessing their performance
    pred_test_labels = best_pipeline.predict(test_data_mat)

    # only the selected features (via index_selected_features) get non-nan value
    feat_importance = get_feature_importance(classifier_name, best_clf,
                                             train_fs.num_features, index_selected_features)

    # TODO NOW test if the gathering of prob data is consistent across multiple calls to this method
    #   perhaps by controlling the class order in input
    # The order of the classes corresponds to that in the attribute best_model.classes_.
    if hasattr(best_pipeline, 'predict_proba'):
        pred_prob = best_pipeline.predict_proba(test_data_mat)
    else:
        pred_prob = None

    conf_mat = confusion_matrix(true_test_labels, pred_test_labels,
                                labels=label_order_in_conf_matrix)

    misclsfd_samples = test_sample_ids[true_test_labels != pred_test_labels]

    return pred_prob, pred_test_labels, true_test_labels, \
           conf_mat, misclsfd_samples, \
           feat_importance, best_params


def optimize_RF_via_training_oob_score(train_data_mat, train_labels, range_min_leafsize, range_num_predictors):
    "Finds the best parameters just based on out of bag error within the training set (supposed to reflect test error)."

    oob_error_train = np.full([len(range_min_leafsize), len(range_num_predictors)], np.nan)

    for idx_ls, minls in enumerate(range_min_leafsize):
        for idx_np, num_pred in enumerate(range_num_predictors):
            rf = RandomForestClassifier(max_features=num_pred, min_samples_leaf=minls,
                                        n_estimators=cfg.NUM_TREES, max_depth=None,
                                        oob_score=True)  # , random_state=SEED_RANDOM)
            rf.fit(train_data_mat, train_labels)
            oob_error_train[idx_ls, idx_np] = rf.oob_score_

    # identifying the best parameters
    best_idx_ls, best_idx_numpred = np.unravel_index(oob_error_train.argmin(), oob_error_train.shape)
    best_minleafsize = range_min_leafsize[best_idx_ls]
    best_num_predictors = range_num_predictors[best_idx_numpred]
    best_params = {'min_samples_leaf': best_minleafsize,
                   'max_features'    : best_num_predictors}

    # training the RF using the best parameters
    best_rf = RandomForestClassifier(max_features=best_num_predictors, min_samples_leaf=best_minleafsize,
                                     oob_score=True,
                                     n_estimators=cfg.NUM_TREES)  # , random_state=SEED_RANDOM)
    best_rf.fit(train_data_mat, train_labels)

    return best_rf, best_params


def optimize_pipeline_via_grid_search_CV(pipeline, train_data_mat, train_labels, param_grid, train_perc):
    "Performs GridSearchCV and returns the best parameters and refitted Pipeline on full dataset with the best parameters."

    # TODO perhaps k-fold is a better inner CV, which guarantees full use of training set with fewer repeats?
    inner_cv = ShuffleSplit(n_splits=cfg.INNER_CV_NUM_SPLITS, train_size=train_perc, test_size=1.0 - train_perc)
    # inner_cv = RepeatedKFold(n_splits=cfg.INNER_CV_NUM_FOLDS, n_repeats=cfg.INNER_CV_NUM_REPEATS)

    # gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv,
    #                   n_jobs=cfg.GRIDSEARCH_NUM_JOBS, pre_dispatch=cfg.GRIDSEARCH_PRE_DISPATCH)

    # not specifying n_jobs to avoid any kind of parallelism (joblib) from within sklearn
    # to avoid potentially bad interactions with outer parallization with builtin multiprocessing library
    gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv,
                      refit=cfg.refit_best_model_on_ALL_training_set)

    # ignoring some not-so-critical warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(action='once', category=UserWarning, module='joblib',
                                message='Multiprocessing-backed parallel loops cannot be nested, setting n_jobs=1')
        warnings.filterwarnings(action='once', category=UserWarning, message='Some inputs do not have OOB scores')
        np.seterr(divide='ignore', invalid='ignore')
        warnings.filterwarnings(action='once', category=RuntimeWarning,
                                message='invalid value encountered in true_divide')
        warnings.simplefilter(action='once', category=DeprecationWarning)

        gs.fit(train_data_mat, train_labels)

    return gs.best_estimator_, gs.best_params_


def save_results(out_dir, dict_of_objects_to_save):
    "Serializes the results to disk."

    # LATER choose a more universal serialization method (that could be loaded from a web app)
    try:
        out_results_path = pjoin(out_dir, cfg.file_name_results)
        with open(out_results_path, 'wb') as resfid:
            pickle.dump(dict_of_objects_to_save, resfid)
    except:
        raise IOError('Error saving the results to disk!')
    else:
        # deleting temp results only when saving full results is successful
        cleanup(out_dir)

    return out_results_path


def load_results_from_folder(results_folder):
    """

    Given a base output folder, possibly containing results for multiple sub-groups,
        returns a dictionary of results, keyed in by sub group identifier.

    """

    results = dict()
    options = load_options(results_folder)
    for ix, sg in enumerate(options['sub_groups']):
        sg_id = sub_group_identifier(sg, ix)
        results_file_path = pjoin(results_folder, sg_id, cfg.file_name_results)
        if not pexists(results_file_path) or os.path.getsize(results_file_path) <= 0:
            raise IOError('Results file for sub group {} does not exist'
                          ' or is empty!'.format(sg_id))
        results[sg_id] = load_results_dict(results_file_path)

    return results


def load_results_dict(results_file_path):
    "Loads the results serialized by RHsT."
    # TODO need to standardize what needs to saved/read back

    if not pexists(results_file_path) or os.path.getsize(results_file_path) <= 0:
        raise IOError("Results file to be loaded doesn't exist, or empty!")

    try:
        with open(results_file_path, 'rb') as rf:
            results_dict = pickle.load(rf)
    except:
        raise IOError('Error loading the saved results from \n{}'.format(results_file_path))

    return results_dict


def load_results(results_file_path):
    "Loads the results serialized by RHsT."
    # TODO need to standardize what needs to saved/read back

    if not pexists(results_file_path):
        raise IOError("Results file to be loaded doesn't exist!")

    try:
        with open(results_file_path, 'rb') as rf:
            results_dict = pickle.load(rf)
            # # below is possible, but not explicit and a bad practice
            # # importing the keys and their values into the workspace
            # locals().update(results_dict)

            dataset_paths, method_names, train_perc, num_repetitions, num_classes, \
            pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, \
            best_params, feature_importances_rf, \
            feature_names, num_times_misclfd, num_times_tested, \
            confusion_matrix, class_set, class_sizes, accuracy_balanced, \
            auc_weighted, positive_class, classifier_name, feat_select_method = \
                [results_dict.get(var_name) for var_name in cfg.rhst_data_variables_to_persist]

    except:
        raise IOError('Error loading the saved results from \n{}'.format(results_file_path))

    # TODO need a consolidated way to deal with what variable are saved and in what order
    return dataset_paths, method_names, train_perc, num_repetitions, num_classes, \
           pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, \
           best_params, feature_importances_rf, feature_names, \
           num_times_misclfd, num_times_tested, \
           confusion_matrix, class_set, class_sizes, \
           accuracy_balanced, auc_weighted, positive_class, classifier_name, feat_select_method


def run(dataset_path_file, method_names, out_results_dir,
        train_perc=0.8, num_repetitions=200,
        positive_class=None, sub_group=None,
        feat_sel_size=cfg.default_num_features_to_select,
        impute_strategy=cfg.default_imputation_strategy,
        missing_flag=None,
        num_procs=4,
        grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT,
        classifier_name=cfg.default_classifier,
        feat_select_method=cfg.default_feat_select_method,
        options_path=None):
    """

    Parameters
    ----------
    dataset_path_file : str
        path to file containing list of paths (each containing a valid MLDataset).

    method_names : list
        A list of names to denote the different feature extraction methods

    out_results_dir : str
        Path to output directory to save the cross validation results to.

    train_perc : float or numpy.float, optional
        Percetange of subjects to train the classifier on.
        The percentage is applied to the size of the smallest class to estimate
        the number of subjects from each class to be reserved for training.
        The smallest class is chosen to avoid class-imbalance in the training set.

        Default: 0.8 (80%).
    num_repetitions : int or numpy.int, optional
        Number of repetitions of cross-validation estimation. Default: 200.

    positive_class : str
        Name of the class to be treated as positive in calculation of AUC

    feat_sel_size : str or int
        Number of features to retain after feature selection.
        Must be a method (tenth or square root of the size of smallest class in training set,
            or a finite integer smaller than the data dimensionality.

    sub_group : list
        List of classes to focus on for classification. Default: all classes available.

    num_procs : int
        Number of parallel processes to run to parallelize the repetitions of CV

    grid_search_level : str
        If 'none', no grid search will be performed, choosing parameters based on 'folk wisdom'.
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be user for optimization.

    options_path : str
        Path to a pickle file which contains the all user chosen options.

    Returns
    -------
    results_path : str
        Path to pickle file containing full set of CV results.

    """

    dataset_paths, num_repetitions, num_procs, sub_group = check_params_rhst(dataset_path_file, out_results_dir,
                                                                             num_repetitions, train_perc, sub_group,
                                                                             num_procs, grid_search_level,
                                                                             classifier_name, feat_select_method)

    # loading datasets
    datasets = load_pyradigms(dataset_paths, sub_group)

    # making sure different feature sets are comparable
    common_ds, class_set, label_set, class_sizes, \
    num_samples, num_classes, num_datasets, num_features = check_feature_sets_are_comparable(datasets)
    # TODO warning when num_rep are not suficient: need a heuristic to assess it

    # re-map the labels (from 1 to n) to ensure numeric labels do not differ
    datasets, positive_class, pos_class_index = remap_labels(datasets, common_ds, class_set, positive_class)

    # determine the common size for training
    train_size_common, total_test_samples = determine_training_size(train_perc, class_sizes, num_classes)

    # the main parallel loop to crunch optimizations, predictions and evaluations
    # chunk_size = int(np.ceil(num_repetitions/num_procs))
    if num_procs > 1:
        print('Parallelizing the repetitions of CV with {} processes ...'.format(num_procs))
        with Manager() as proxy_manager:
            shared_inputs = proxy_manager.list([datasets, impute_strategy,
                                                train_size_common, feat_sel_size, train_perc,
                                                total_test_samples, num_classes, num_features, label_set,
                                                method_names, pos_class_index, out_results_dir,
                                                grid_search_level, classifier_name, feat_select_method])
            partial_func_holdout = partial(holdout_trial_compare_datasets, *shared_inputs)

            with Pool(processes=num_procs) as pool:
                cv_results = pool.map(partial_func_holdout, range(num_repetitions))
    else:
        # switching to regular sequential for loop
        partial_func_holdout = partial(holdout_trial_compare_datasets, datasets, impute_strategy,
                                       train_size_common, feat_sel_size,
                                       train_perc, total_test_samples, num_classes, num_features, label_set,
                                       method_names, pos_class_index, out_results_dir, grid_search_level,
                                       classifier_name, feat_select_method)
        cv_results = [partial_func_holdout(rep_id=rep) for rep in range(num_repetitions)]

    # re-assemble results into a convenient form
    pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, confusion_matrix, \
    accuracy_balanced, auc_weighted, best_params, feature_names, \
    feature_importances_per_rep, feature_importances_rf, num_times_misclfd, num_times_tested = \
        gather_results_across_trials(cv_results, common_ds, datasets, total_test_samples,
                                     num_repetitions, num_datasets, num_classes, num_features)

    # saving the required variables to disk in a dict
    locals_var_dict = locals()
    dict_to_save = {var: locals_var_dict[var] for var in cfg.rhst_data_variables_to_persist}
    out_results_path = save_results(out_results_dir, dict_to_save)

    report_best_params(best_params, method_names, out_results_dir)

    # exporting the results right away, without waiting for figures
    export_results(dict_to_save, out_results_dir, options_path)

    summarize_perf(accuracy_balanced, auc_weighted, method_names, num_classes, num_datasets)

    return out_results_path


def determine_training_size(train_perc, class_sizes, num_classes):
    """Computes the maximum training size that the smallest class can provide """

    print("Different classes in the training set are stratified to match the smallest class!")
    train_size_per_class = np.int64(np.floor(train_perc * class_sizes).astype(np.float64))
    # per-class
    train_size_common = np.int64(np.minimum(min(train_size_per_class), train_size_per_class))
    # single number
    reduced_sizes = np.unique(train_size_common)
    if len(reduced_sizes) != 1:
        raise ValueError("Error in stratification of training set based on the smallest class!")
    train_size_common = reduced_sizes[0]

    if train_size_common < 1:
        raise ValueError('Invalid state - Zero samples selected for training!'
                         'Check the class size distribution in dataset!')

    total_test_samples = np.int64(np.sum(class_sizes) - num_classes * train_size_common)

    return train_size_common, total_test_samples


def initialize_misclf_counters(sample_ids, num_datasets):
    """Initialize misclassification counters."""

    num_times_tested = list()
    num_times_misclfd = list()
    for dd in range(num_datasets):
        num_times_tested.append(Counter(sample_ids))
        num_times_misclfd.append(Counter(sample_ids))
        for subid in sample_ids:
            num_times_tested[dd][subid] = 0
            num_times_misclfd[dd][subid] = 0

    return num_times_misclfd, num_times_tested


def initialize_result_containers(common_ds, datasets, total_test_samples,
                                 num_repetitions, num_datasets, num_classes, num_features):
    """Prepare containers for various outputs"""

    pred_prob_per_class = np.full([num_repetitions, num_datasets, total_test_samples, num_classes], np.nan)
    pred_labels_per_rep_fs = np.full([num_repetitions, num_datasets, total_test_samples], np.nan)
    test_labels_per_rep = np.full([num_repetitions, total_test_samples], np.nan)

    best_params = [None] * num_repetitions

    num_times_misclfd, num_times_tested = initialize_misclf_counters(common_ds.sample_ids, num_datasets)

    # multi-class metrics
    confusion_matrix = np.full([num_repetitions, num_classes, num_classes, num_datasets], np.nan)
    accuracy_balanced = np.full([num_repetitions, num_datasets], np.nan)
    auc_weighted = np.full([num_repetitions, num_datasets], np.nan)

    feature_names = [None] * num_datasets
    feature_importances_per_rep = [None] * num_repetitions
    feature_importances_rf = [None] * num_datasets
    for idx in range(num_datasets):
        feature_importances_rf[idx] = np.full([num_repetitions, num_features[idx]], np.nan)
        feature_names[idx] = datasets[idx].feature_names

    return pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, \
           confusion_matrix, accuracy_balanced, auc_weighted, best_params, \
           feature_names, feature_importances_per_rep, feature_importances_rf, \
           num_times_misclfd, num_times_tested


def get_pretty_print_options(method_names, num_datasets):
    """Returns field widths for formatting"""

    if len(method_names) < num_datasets:
        raise ValueError('Insufficient number of names (n={}) '
                         'for the given feature sets (n={}).'.format(len(method_names), num_datasets))

    max_width_method_names = max(map(len, method_names))
    ndigits_ndatasets = len(str(num_datasets))
    pretty_print = namedtuple('pprint', ['str_width', 'num_digits'])
    print_options = pretty_print(max_width_method_names, ndigits_ndatasets)

    return print_options


def impute_missing_data(train_data, train_labels, strategy, test_data):
    """
    Imputes missing values in train/test data matrices using the given strategy,
    based on train data alone.

    """

    from sklearn.impute import SimpleImputer
    # TODO integrate and use the missingdata pkg (with more methods) when time permits
    imputer = SimpleImputer(missing_values=cfg.missing_value_identifier,
                            strategy=strategy)
    imputer.fit(train_data, train_labels)

    return imputer.transform(train_data), imputer.transform(test_data)


def remap_labels(datasets, common_ds, class_set, positive_class=None):
    """re-map the labels (from 1 to n) to ensure numeric labels do not differ"""

    numeric_labels = make_numeric_labels(class_set)

    # finding the numeric label for positive class
    # label will also be in the index into the arrays over classes due to construction above
    if positive_class is None:
        positive_class = class_set[-1]
    elif positive_class not in class_set:
        raise ValueError('Chosen positive class does not exist in the dataset')
    pos_class_index = class_set.index(positive_class)  # remapped_class_labels[positive_class]

    labels_with_correspondence = dict()
    for subid in common_ds.sample_ids:
        labels_with_correspondence[subid] = numeric_labels[common_ds.classes[subid]]

    for idx in range(len(datasets)):
        datasets[idx].labels = labels_with_correspondence

    return datasets, positive_class, pos_class_index


def holdout_trial_compare_datasets(datasets, impute_strategy,
                                   train_size_common, feat_sel_size, train_perc,
                                   total_test_samples, num_classes, num_features_per_dataset,
                                   label_set, method_names, pos_class_index,
                                   out_results_dir, grid_search_level,
                                   classifier_name, feat_select_method, rep_id=None):
    """
    Runs a single iteration of optimizing the chosen pipeline on the chosen training set,
    and evaluations on the given test set.

    Parameters
    ----------
    datasets

    impute_strategy : str
        Strategy to handle the missing data: whether to raise an error if data is missing, or
            to impute them using the method chosen here.

    train_size_common
    feat_sel_size
    train_perc
    total_test_samples
    num_classes
    num_features_per_dataset
    label_set
    method_names
    pos_class_index
    out_results_dir
    rep_id

    grid_search_level : str
        If 'light', grid search resolution will be reduced to speed up optimization.
        If 'exhaustive', most values for most parameters will be user for optimization.

    Returns
    -------

    """

    common_ds = datasets[cfg.COMMON_DATASET_INDEX]
    num_datasets = len(datasets)

    pred_prob_per_class = np.full([num_datasets, total_test_samples, num_classes], np.nan)
    pred_labels_per_rep_fs = np.full([num_datasets, total_test_samples], np.nan)
    true_test_labels = np.full(total_test_samples, np.nan)

    # multi-class metrics
    confusion_matrix = np.full([num_classes, num_classes, num_datasets], np.nan)
    accuracy_balanced = np.full(num_datasets, np.nan)
    auc_weighted = np.full(num_datasets, np.nan)
    best_params = [None] * num_datasets
    misclsfd_ids_this_run = [None] * num_datasets

    feature_importances = [None] * num_datasets
    for idx in range(num_datasets):
        feature_importances[idx] = np.full(num_features_per_dataset[idx], np.nan)

    # set of subjects for training and testing, common for all datasets.
    train_set, test_set = common_ds.train_test_split_ids(count_per_class=train_size_common)
    true_test_labels = [common_ds.labels[sid] for sid in test_set if sid in common_ds.labels]

    # to uniquely identify this iteration
    if rep_id is None:
        rep_proc_id = 'process{}'.format(os.getpid())  # str(os.getpid())
    else:
        rep_proc_id = str(rep_id)
    print_options = get_pretty_print_options(method_names, num_datasets)

    # evaluating each feature/dataset
    for dd in range(num_datasets):
        print("CV trial {rep:6} feature {index:{nd}} {name:>{namewidth}} : ".format(rep=rep_proc_id, index=dd,
                                                                                    name=method_names[dd],
                                                                                    nd=print_options.num_digits,
                                                                                    namewidth=print_options.str_width),
              end='')

        # using the same train/test sets for all feature sets.
        train_fs = datasets[dd].get_subset(train_set)
        test_fs = datasets[dd].get_subset(test_set)

        pred_prob_per_class[dd, :, :], pred_labels_per_rep_fs[dd, :], true_test_labels, \
        conf_mat, misclsfd_ids_this_run[dd], feature_importances[dd], best_params[dd] = \
            eval_optimized_model_on_testset(train_fs, test_fs,
                                            impute_strategy=impute_strategy,
                                            train_perc=train_perc,
                                            feat_sel_size=feat_sel_size,
                                            label_order_in_conf_matrix=label_set,
                                            grid_search_level=grid_search_level,
                                            classifier_name=classifier_name,
                                            feat_select_method=feat_select_method)

        # TODO new feature: add additional metrics such as PPV
        accuracy_balanced[dd] = balanced_accuracy(conf_mat)
        confusion_matrix[:, :, dd] = conf_mat
        print('balanced accuracy: {:.4f} '.format(accuracy_balanced[dd]), end='')

        if num_classes == 2:
            auc_weighted[dd] = roc_auc_score(true_test_labels, pred_prob_per_class[dd, :, pos_class_index],
                                             average='weighted')
            print('\t weighted AUC: {:.4f}'.format(auc_weighted[dd]), end='')

        print('', flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

    results_list = [pred_prob_per_class, pred_labels_per_rep_fs, true_test_labels, accuracy_balanced,
                    confusion_matrix, auc_weighted, feature_importances, best_params,
                    misclsfd_ids_this_run, test_set]

    tmp_dir = get_temp_dir(out_results_dir)
    out_path = pjoin(tmp_dir, '{}_{}.pkl'.format(cfg.temp_prefix_rhst, rep_proc_id))
    logging.info('results from rep {} saved to {}'.format(rep_proc_id, out_path))
    with open(out_path, 'bw') as of:
        pickle.dump(results_list, of)

    return results_list


def get_temp_dir(out_results_dir):
    "Scratch directory to save temporary results to"

    tmp_dir = pjoin(out_results_dir, cfg.temp_results_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    return tmp_dir


def cleanup(out_dir):
    "Helper to perform cleanup"

    tmp_dir = get_temp_dir(out_dir)
    try:
        shutil.rmtree(tmp_dir)
    except:
        traceback.print_exc()
        print('Unable to delete temporary folder at:\n\t{}\n'
              'Remove it manually if you would like to save space.'.format(tmp_dir))

    return


def gather_results_across_trials(cv_results, common_ds, datasets, total_test_samples,
                                 num_repetitions, num_datasets, num_classes, num_features):
    "Reorganizes list of indiv CV trial results into rectangular arrays."

    pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, confusion_matrix, \
    accuracy_balanced, auc_weighted, best_params, feature_names, \
    feature_importances_per_rep, feature_importances_rf, \
    num_times_misclfd, num_times_tested = initialize_result_containers(common_ds, datasets,
                                                                       total_test_samples, num_repetitions,
                                                                       num_datasets, num_classes, num_features)

    for rep in range(num_repetitions):
        # unpacking each rep
        _rep_pred_prob_per_class, _rep_pred_labels_per_rep_fs, _rep_true_test_labels, \
        _rep_accuracy_balanced, _rep_confusion_matrix, _rep_auc_weighted, _rep_feature_importances, \
        _rep_best_params, _rep_misclsfd_ids_this_run, _rep_test_set = cv_results[rep]

        pred_prob_per_class[rep, :, :, :] = _rep_pred_prob_per_class
        pred_labels_per_rep_fs[rep, :, :] = _rep_pred_labels_per_rep_fs
        test_labels_per_rep[rep, :] = _rep_true_test_labels
        accuracy_balanced[rep, :] = _rep_accuracy_balanced
        confusion_matrix[rep, :, :, :] = _rep_confusion_matrix
        auc_weighted[rep, :] = _rep_auc_weighted
        best_params[rep] = _rep_best_params

        for dd in range(num_datasets):
            num_times_misclfd[dd].update(_rep_misclsfd_ids_this_run[dd])
            num_times_tested[dd].update(_rep_test_set)
            feature_importances_rf[dd][rep, :] = _rep_feature_importances[dd]

        # this variable is not being saved/used in any other way.
        feature_importances_per_rep[rep] = _rep_feature_importances

    return pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, confusion_matrix, \
           accuracy_balanced, auc_weighted, best_params, feature_names, \
           feature_importances_per_rep, feature_importances_rf, num_times_misclfd, num_times_tested


def summarize_perf(accuracy_balanced, auc_weighted, method_names, num_classes, num_datasets):
    """Prints median performance for each feature set"""

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='All-NaN slice encountered',
                                module='numpy', category=RuntimeWarning)

        # assuming the first column (axis 0) is over num_repititions
        median_bal_acc = np.nanmedian(accuracy_balanced, axis=0)
        if num_classes == 2:
            median_wtd_auc = np.nanmedian(auc_weighted, axis=0)

    print_options = get_pretty_print_options(method_names, num_datasets)

    print('\nMedian performance summary:', end='')
    for dd in range(num_datasets):
        print("\nfeature {index:{nd}} {name:>{namewidth}} : "
              "balanced accuracy {accuracy:2.2f} ".format(index=dd, name=method_names[dd],
                                                          accuracy=median_bal_acc[dd],
                                                          namewidth=print_options.str_width,
                                                          nd=print_options.num_digits), end='')
        if num_classes == 2:
            print("\t AUC {auc:2.2f}".format(auc=median_wtd_auc[dd]), end='')

    print('')
    return


if __name__ == '__main__':
    pass
