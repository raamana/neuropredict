from __future__ import print_function

__all__ = ['run', 'load_results', 'save_results']

import os
import pickle
import logging
import warnings
from collections import Counter, namedtuple
from sys import version_info
from os.path import join as pjoin, exists as pexists, realpath
from multiprocessing import Pool, Manager, Array as multiArray, Value as multiValue
from functools import partial

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.feature_selection import mutual_info_classif, SelectKBest, VarianceThreshold
from sklearn.pipeline import Pipeline

from pyradigm import MLDataset

if version_info.major == 2 and version_info.minor == 7:
    import config_neuropredict as cfg
    import run_workflow
elif version_info.major > 2:
    from neuropredict import config_neuropredict as cfg
    from neuropredict import run_workflow
else:
    raise NotImplementedError('neuropredict supports only 2.7 or Python 3+. Upgrade to Python 3+ is recommended.')


def eval_optimized_model_on_testset(train_fs, test_fs,
                                    label_order_in_conf_matrix=None,
                                    feat_sel_size=cfg.default_num_features_to_select,
                                    train_perc=0.5):
    """
    Optimize the classifier on the training set and return predictions on test set.

    Parameters
    ----------
    train_fs : MLDataset
        Dataset to optimize a given classifier on.

    test_fs : MLDataset
        Dataset to make predictions on using the classifier optimized on training set.

    label_order_in_conf_matrix : list
        List of labels to compute the order of confusion matrix.

    feat_sel_size : str or int
        Metho to choose the number of featurese to select.

    train_perc : float
        Training set fraction to run the inner cross-validation.

    Returns
    -------

    """

    if label_order_in_conf_matrix is None:
        raise ValueError('Label order for confusion matrix must be specified for accurate results/visulizations.')

    train_data_mat, train_labels, _ = train_fs.data_and_labels()
    test_data_mat, true_test_labels, test_sample_ids = test_fs.data_and_labels()

    train_class_sizes = list(train_fs.class_sizes.values())
    pipeline, param_grid = get_pipeline(train_class_sizes, feat_sel_size, train_fs.num_features)

    best_pipeline, best_params = optimize_pipeline_via_grid_search_CV(pipeline, train_data_mat, train_labels,
                                                                      param_grid, train_perc)
    # best_model, best_params = optimize_RF_via_training_oob_score(train_data_mat, train_labels,
    #     param_grid['random_forest_clf__min_samples_leaf'], param_grid['random_forest_clf__max_features'])

    _, best_fsr = best_pipeline.steps[0]  # assuming order in pipeline construction - step 0 : feature selector
    _, best_clf = best_pipeline.steps[-1]  # the final step in an sklearn pipeline is always an estimator/classifier

    # could be useful to compute frequency of selection
    index_selected_features = best_fsr.get_support(indices=True)

    # making predictions on the test set and assessing their performance
    pred_test_labels = best_pipeline.predict(test_data_mat)

    feat_importance = np.full(train_fs.num_features, np.nan)
    if hasattr(best_clf, 'feature_importances_'):
        feat_importance[index_selected_features] = best_clf.feature_importances_

    # TODO NOW test if the gathering of prob data is consistent across multiple calls to this method
    #   perhaps by controlling the class order in input
    # The order of the classes corresponds to that in the attribute best_model.classes_.
    pred_prob = best_pipeline.predict_proba(test_data_mat)

    conf_mat = confusion_matrix(true_test_labels, pred_test_labels, label_order_in_conf_matrix)

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

    inner_cv = ShuffleSplit(n_splits=cfg.INNER_CV_NUM_SPLITS, train_size=train_perc, test_size=1.0 - train_perc)
    gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv,
                      n_jobs=cfg.GRIDSEARCH_NUM_JOBS, pre_dispatch=cfg.GRIDSEARCH_PRE_DISPATCH)

    with warnings.catch_warnings() as warn_ctrl:
        warnings.filterwarnings(action='once', category=UserWarning, module='joblib',
                                message='Multiprocessing-backed parallel loops cannot be nested, setting n_jobs=1')
        gs.fit(train_data_mat, train_labels)

    return gs.best_estimator_, gs.best_params_


def optimize_RF_via_grid_search_CV(train_data_mat, train_labels, param_grid, train_perc):
    "Performs GridSearchCV and returns the best parameters and refitted RandomForest on full dataset with the best parameters."

    # sample classifier
    rf = RandomForestClassifier(max_features=10, n_estimators=10, oob_score=True)

    inner_cv = ShuffleSplit(n_splits=cfg.INNER_CV_NUM_SPLITS, train_size=train_perc, test_size=1.0 - train_perc)
    gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=inner_cv)
    gs.fit(train_data_mat, train_labels)

    return gs.best_estimator_, gs.best_params_


def __max_dimensionality_to_avoid_curseofdimensionality(num_samples, num_features,
                                                        perc_prob_error_allowed=cfg.PERC_PROB_ERROR_ALLOWED):
    """
    Computes the largest dimensionality that can be used to train a predictive model
        avoiding curse of dimensionality for a 5% probability of error.
    Optional argument can specify the amount of error that the user wants to allow (deafult : 5% prob of error ).

    Citation: Michael Fitzpatrick and Milan Sonka, Handbook of Medical Imaging, 2000
    
    Parameters
    ----------
    num_samples : int
    num_features : int
    perc_prob_error_allowed : float

    Returns
    -------
    max_red_dim : int
    

    """

    if num_samples < 1.0 / (2.0 * perc_prob_error_allowed):
        max_red_dim = 1
    else:
        max_red_dim = np.floor(num_samples * (2.0 * perc_prob_error_allowed))

    # to ensure we don't request more features than available
    max_red_dim = np.int64(min(max_red_dim, num_features))

    return max_red_dim


def compute_reduced_dimensionality(select_method, train_class_sizes, train_data_dim):
    """
    Estimates the number of features to retain for feature selection based on method chosen.

    Parameters
    ----------
    select_method : str
        Type of feature selection.
    train_class_sizes : iterable
        Sizes of all classes in training set.
    train_data_dim : int
        Data dimensionality for the feature set.

    Returns
    -------
    reduced_dim : int
        Reduced dimensionality n, such that 1 <= n <= train_data_dim.

    Raises
    ------
    ValueError
        If method choices were invalid.
    """

    # default to use them all.
    if select_method in [None, 'all']:
        return train_data_dim


    def do_sqrt(size):
        return np.ceil(np.sqrt(size))


    def do_tenth(size):
        return np.ceil(size / 10)


    def do_log2(size):
        return np.ceil(np.log2(size))


    get_reduced_dim = {'tenth': do_tenth,
                       'sqrt' : do_sqrt,
                       'log2' : do_log2}

    if isinstance(select_method, str):
        if select_method in get_reduced_dim:
            smallest_class_size = np.sum(train_class_sizes)
            calc_size = get_reduced_dim[select_method](smallest_class_size)
        else:
            # arg could be string coming from command line
            calc_size = np.int64(select_method)
        reduced_dim = min(calc_size, train_data_dim)
    elif isinstance(select_method, int):
        if select_method > train_data_dim:
            reduced_dim = train_data_dim
            print('Reducing the feature selection size to {}, '
                  'to accommondate the current feature set.'.format(train_data_dim))
        else:
            reduced_dim = select_method
    else:
        raise ValueError('method to choose feature selection size can only be string or integer!')

    # ensuring it is an integer >= 1
    reduced_dim = np.int64(np.max([reduced_dim, 1]))

    return reduced_dim


def chance_accuracy(class_sizes, method='imbalanced'):
    """
    Computes the chance accuracy for a given set of classes with varying sizes.

    Parameters
    ----------
    class_sizes : list
        List of sizes of the classes.

    method : str
        Type of method to use to compute the chance accuracy. Two options:

            - `imbalanced` : uses the proportions of all classes [Default]
            - `zero_rule`  : uses the so called Zero Rule (fraction of majority class)

        Both methods return similar results, with Zero Rule erring on the side higher chance accuracy.

    Useful discussion at `stackexchange.com <https://stats.stackexchange.com/questions/148149/what-is-the-chance-level-accuracy-in-unbalanced-classification-problems>`_

    Returns
    -------
    chance_acc : float
        Accuracy of purely random/chance classifier on this particular dataset.

    """

    num_classes = len(class_sizes)
    num_samples = sum(class_sizes)
    # # the following is wrong if imbalance is present
    # chance_acc = 1.0 / num_classes

    method = method.lower()
    if method in ['imbalanced', ]:
        chance_acc = np.sum(np.square(class_sizes / num_samples))
    elif method in ['zero_rule', 'zeror' ]:
        # zero rule: fraction of largest class
        chance_acc = np.max(class_sizes) / num_samples
    else:
        raise ValueError('Invalid choice of method to compute choice accuracy!')

    return chance_acc


def balanced_accuracy(confmat):
    "Computes the balanced accuracy in a given confusion matrix!"

    num_classes = confmat.shape[0]
    if num_classes != confmat.shape[1]:
        raise ValueError("given confusion matrix is not square!")

    confmat = confmat.astype(np.float64)

    indiv_class_acc = np.full([num_classes, 1], np.nan)
    for cc in range(num_classes):
        indiv_class_acc[cc] = confmat[cc, cc] / np.sum(confmat[cc, :])

    bal_acc = np.mean(indiv_class_acc)

    return bal_acc


def save_results(out_dir, dict_of_objects_to_save):
    "Serializes the results to disk."

    # LATER choose a more universal serialization method (that could be loaded from a web app)
    try:
        out_results_path = pjoin(out_dir, cfg.file_name_results)
        with open(out_results_path, 'wb') as resfid:
            pickle.dump(dict_of_objects_to_save, resfid)
    except:
        raise IOError('Error saving the results to disk!')

    return out_results_path


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
            auc_weighted, positive_class = [results_dict.get(var_name) for var_name in
                                            cfg.rhst_data_variables_to_persist]

    except:
        raise IOError('Error loading the saved results from \n{}'.format(results_file_path))

    # TODO need a consolidated way to deal with what variable are saved and in what order
    return dataset_paths, method_names, train_perc, num_repetitions, num_classes, \
           pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, \
           best_params, feature_importances_rf, feature_names, \
           num_times_misclfd, num_times_tested, \
           confusion_matrix, class_set, class_sizes, \
           accuracy_balanced, auc_weighted, positive_class


def get_RandomForestClassifier(reduced_dim='all'):
    """ Returns the Random Forest classifier and its parameter grid. """

    range_num_trees = range(cfg.NUM_TREES_RANGE[0], cfg.NUM_TREES_RANGE[1], cfg.NUM_TREES_STEP)
    range_min_leafsize = range(cfg.MAX_MIN_LEAFSIZE, 1, -cfg.LEAF_SIZE_STEP)
    range_num_predictors = range(reduced_dim, 1, -cfg.NUM_PREDICTORS_STEP)

    # capturing the edge cases
    if len(range_min_leafsize) < 1:
        range_min_leafsize = [1, ]
    if len(range_num_predictors) <= 1:
        range_num_predictors = [reduced_dim, ]
    if len(range_num_trees) < 1:
        range_num_trees = [cfg.NUM_TREES, ]

    # name clf_model chosen to enable generic selection classifier later on
    # not optimizing over number of features to save time
    clf_name = 'random_forest_clf'
    param_name = lambda string: '{}__{}'.format(clf_name, string)
    param_grid = {param_name('min_samples_leaf'): range_min_leafsize,
                  param_name('max_features')    : range_num_predictors,
                  param_name('n_estimators')    : range_num_trees}

    rfc = RandomForestClassifier(max_features=10, n_estimators=10, oob_score=True)

    return rfc, clf_name, param_grid


def get_classifier(classifier_name='RandomForestClassifier',
                   reduced_dim='all'):
    """
    Returns the named classifier and its parameter grid.

    Parameters
    ----------
    classifier_name : str
        String referring to a valid scikit-learn classifier.

    reduced_dim : int or str
        Reduced dimensionality, either an integer or "all", which defaults to using everything.

    Returns
    -------
    clf : sklearn.estimator
        Valid scikit-learn estimator.
    clf_name : str
        String identifying the classifier to construct the parameter grid.
    param_grid : dict
        Dict of named ranges to construct the parameter grid.

    """

    if classifier_name.lower() in ['randomforestclassifier', ]:
        clf, clf_name, param_grid = get_RandomForestClassifier(reduced_dim)
    else:
        raise NotImplementedError('Invalid name or classifier not implemented.')

    return clf, clf_name, param_grid


def get_feature_selector(feat_selector_name='variancethreshold',
                         reduced_dim='all'):
    """
    Returns the named classifier and its parameter grid.

    Parameters
    ----------
    feat_selector_name : str
        String referring to a valid scikit-learn feature selector.
    reduced_dim : str or int
        Reduced dimensionality, either an integer or "all", which defaults to using everything.

    Returns
    -------
    feat_selector : sklearn.feature_selection method
        Valid scikit-learn feature selector.
    clf_name : str
        String identifying the feature selector to construct the parameter grid.
    param_grid : dict
        Dict of named ranges to construct the parameter grid.

    """

    fs_name = feat_selector_name.lower()
    if fs_name in ['selectkbest_mutual_info_classif', ]:
        # no param optimization for feat selector for now.
        feat_selector = SelectKBest(score_func=mutual_info_classif, k=reduced_dim)
        fs_param_grid = None
    elif fs_name in ['variancethreshold', ]:
        feat_selector = VarianceThreshold(threshold=cfg.variance_threshold)
        fs_param_grid = None
    else:
        raise NotImplementedError('Invalid name or feature selector not implemented.')

    return feat_selector, fs_name, fs_param_grid


def get_pipeline(train_class_sizes, feat_sel_size, num_features,
                 classifier_name='RandomForestClassifier',
                 feat_selector_name='variancethreshold'):
    """
    Constructor for pipeline (feature selection followed by a classifier).

    Parameters
    ----------
    train_class_sizes : list
        sizes of different classes in the training set, to estimate the reduced dimensionality.

    feat_sel_size : str or int
        Method specify the method to estimate the reduced dimensionality.
        Choices : integer, or one of the methods in ['sqrt', 'tenth', 'log2', 'all']

    num_features : int
        Number of features in the training set.

    classifier_name : str
        String referring to a valid scikit-learn classifier.

    feat_selector_name : str
        String referring to a valid scikit-learn feature selector.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Valid scikit learn pipeline.
    param_grid : dict
        Dict of named ranges to construct the parameter grid.
        Grid contains ranges for parameters of both feature selector and classifier.

    """

    reduced_dim = compute_reduced_dimensionality(feat_sel_size, train_class_sizes, num_features)

    estimator, est_name, clf_param_grid = get_classifier(classifier_name, reduced_dim)
    feat_selector, fs_name, fs_param_grid = get_feature_selector(feat_selector_name, reduced_dim)

    param_grid = clf_param_grid
    if fs_param_grid:  # None or empty dict both evaluate to False
        fs_params = set(fs_param_grid.keys())
        clf_params = set(param_grid.keys())
        if len(fs_params.intersection(clf_params)):
            # raising error to avoid silent overwrite
            raise ValueError('Overlap in parameters betwen feature selector and the classifier. Remove it.')

        # dict gets extended due to lack of overlap in keys
        param_grid.update(fs_param_grid)

    steps = [(fs_name, feat_selector),
             (est_name, estimator)]
    pipeline = Pipeline(steps)

    return pipeline, param_grid


def run(dataset_path_file, method_names, out_results_dir,
        train_perc=0.8, num_repetitions=200,
        positive_class=None, sub_group=None,
        feat_sel_size=cfg.default_num_features_to_select,
        num_procs=4):
    """

    Parameters
    ----------
    dataset_path_file : str
        path to file containing list of paths (each containing a valid MLDataset).

    method_names : list
        A list of names to denote the different feature extraction methods

    out_results_dir : str
        Path to output directory to save the cross validation results to.

    train_perc : float, optional
        Percetange of subjects to train the classifier on.
        The percentage is applied to the size of the smallest class to estimate
        the number of subjects from each class to be reserved for training.
        The smallest class is chosen to avoid class-imbalance in the training set.

        Default: 0.8 (80%).
    num_repetitions : int, optional
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

    Returns
    -------
    results_path : str
        Path to pickle file containing full set of CV results.

    """

    # structure of this function
    # load datasets
    # validate each dataset
    # ensure same number of subjects across all datasets
    #        same number of features/subject in each dataset
    #        same class set across all datasets
    # re-map the labels (from 1 to n) to ensure numeric labels do not differ
    # sort them if need be (not needed if MLDatasets)
    # for rep 1 to N, for feat 1 to M,
    #   run train/test/evaluate.
    #   keep tab on misclassifications
    # save results (comprehensive and reloadable manner)


    dataset_paths, num_repetitions, num_procs, sub_group = check_params_rhst(dataset_path_file, out_results_dir,
                                                                             num_repetitions, train_perc, sub_group, num_procs)

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
    chunk_size = int(np.ceil(num_repetitions/num_procs))
    with Manager() as proxy_manager:
        shared_inputs = proxy_manager.list([datasets, train_size_common, feat_sel_size, train_perc, total_test_samples,
                                         num_classes, num_features, label_set,
                                         method_names, pos_class_index, out_results_dir])
        partial_func_holdout = partial(holdout_trial_compare_datasets, *shared_inputs)

        with Pool(processes=num_procs) as pool:
            cv_results = pool.map(partial_func_holdout, range(num_repetitions), chunk_size)

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

    # exporting the results right away, without waiting for figures
    run_workflow.export_results(dict_to_save, out_results_dir)

    summarize_perf(accuracy_balanced, auc_weighted, method_names, num_classes, num_datasets)

    return out_results_path


def check_params_rhst(dataset_path_file, out_results_dir, num_repetitions, train_perc, sub_groups, num_procs):
    """Validates inputs and returns paths to feature sets to load"""

    if not pexists(dataset_path_file):
        raise IOError("File containing dataset paths does not exist.")

    with open(dataset_path_file, 'r') as dpf:
        dataset_paths = dpf.read().splitlines()
        # removing duplicates
        dataset_paths = set(dataset_paths)

    try:
        out_results_dir = realpath(out_results_dir)
        if not pexists(out_results_dir):
            os.mkdir(out_results_dir)
    except:
        raise IOError('Error in checking or creating output directiory. Ensure write permissions!')

    num_repetitions = int(num_repetitions)
    if not np.isfinite(num_repetitions):
        raise ValueError("Infinite number of repetitions is not recommened!")

    if num_repetitions <= 1:
        raise ValueError("More than 1 repetition is necessary!")

    if not 0.01 <= train_perc <= 0.99:
        raise ValueError("Training percentage {} out of bounds "
                         "- must be > 0.01 and < 0.99".format(train_perc))

    num_procs = run_workflow.check_num_procs(num_procs)

    # removing empty elements
    if sub_groups is not None:
        sub_groups = [ group for group in sub_groups if group]
    # NOTE: here, we are not ensuring classes in all the subgroups actually exist in all datasets
    # that happens when loading data.

    return dataset_paths, num_repetitions, num_procs, sub_groups


def load_pyradigms(dataset_paths, sub_group=None):
    """Reads in a list of datasets in pyradigm format.

    Parameters
    ----------
    dataset_paths : iterable
        List of paths to pyradigm dataset

    sub_group : iterable
        subset of classes to return. Default: return all classes.
        If sub_group is specified, returns only that subset of classes for all datasets.

    Raises
    ------
        ValueError
            If all the datasets do not contain the request subset of classes.

    """

    if sub_group is not None:
        sub_group = set(sub_group)

    # loading datasets
    datasets = list()
    for fp in dataset_paths:
        if not pexists(fp):
            raise IOError("Dataset @ {} does not exist.".format(fp))

        try:
            # there is an internal validation of dataset
            ds_in = MLDataset(fp)
        except:
            print("Dataset @ {} is not a valid MLDataset!".format(fp))
            raise

        class_set = set(ds_in.class_set)
        if sub_group is None or sub_group == class_set:
            ds_out = ds_in
        elif sub_group < class_set: # < on sets is an issubset operation
            ds_out = ds_in.get_class(sub_group)
        else:
            raise ValueError('One or more classes in {} does not exist in\n{}'.format(sub_group, fp))

        # add the valid dataset to list
        datasets.append(ds_out)

    return datasets


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


def check_feature_sets_are_comparable(datasets, common_ds_index=cfg.COMMON_DATASET_INDEX):
    """Validating all the datasets are comparable e.g. with same samples and classes."""

    # looking into the first dataset
    common_ds = datasets[common_ds_index]
    class_set, label_set_in_ds, class_sizes = common_ds.summarize_classes()
    # below code turns the labels numeric regardless of dataset
    label_set = list(range(len(label_set_in_ds)))

    common_samples = set(common_ds.sample_ids)

    num_samples = common_ds.num_samples
    num_classes = len(class_set)

    num_datasets = len(datasets)
    if num_datasets > 1:
        for idx in range(1, num_datasets):
            this_ds = datasets[idx]
            if num_samples != this_ds.num_samples:
                raise ValueError("Number of samples in different datasets differ!")
            if common_samples != set(this_ds.sample_ids):
                raise ValueError("Sample IDs differ across atleast two datasets!\n"
                                 "All datasets must have the same set of samples, "
                                 "even if the dimensionality of individual feature set changes.")
            if set(class_set) != set(this_ds.classes.values()):
                raise ValueError("Classes differ among datasets! \n One dataset: {} \n Another: {}".format(
                        set(class_set), set(this_ds.classes.values())))

    num_features = np.zeros(num_datasets).astype(np.int64)
    for idx in range(num_datasets):
        num_features[idx] = datasets[idx].num_features

    return common_ds, class_set, label_set, class_sizes, num_samples, num_classes, num_datasets, num_features


def remap_labels(datasets, common_ds, class_set, positive_class=None):
    """re-map the labels (from 1 to n) to ensure numeric labels do not differ"""

    remapped_class_labels = dict()
    for idx, cls in enumerate(class_set):
        remapped_class_labels[cls] = idx

    # finding the numeric label for positive class
    # label will also be in the index into the arrays over classes due to construction above
    if positive_class is None:
        positive_class = class_set[-1]
    elif positive_class not in class_set:
        raise ValueError('Chosen positive class does not exist in the dataset')
    pos_class_index = class_set.index(positive_class)  # remapped_class_labels[positive_class]

    labels_with_correspondence = dict()
    for subid in common_ds.sample_ids:
        labels_with_correspondence[subid] = remapped_class_labels[common_ds.classes[subid]]

    for idx in range(len(datasets)):
        datasets[idx].labels = labels_with_correspondence

    return datasets, positive_class, pos_class_index


def holdout_trial_compare_datasets(datasets, train_size_common, feat_sel_size, train_perc,
                                   total_test_samples, num_classes, num_features_per_dataset,
                                   label_set, method_names, pos_class_index,
                                   out_results_dir, rep_id=None):
    """Runs a single iteration of optimizing the chosen pipeline on the chosen training set,
    and evaluations on the given test set. """

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
        rep_proc_id = 'process{}'.format(os.getpid()) # str(os.getpid())
    else:
        rep_proc_id = str(rep_id)
    print_options = get_pretty_print_options(method_names, num_datasets)

    # evaluating each feature/dataset
    for dd in range(num_datasets):
        print("CV trial {rep:6} feature {index:{nd}} {name:>{namewidth}} : ".format(rep=rep_proc_id, index=dd,
            name=method_names[dd], nd=print_options.num_digits, namewidth=print_options.str_width), end='')

        # using the same train/test sets for all feature sets.
        train_fs = datasets[dd].get_subset(train_set)
        test_fs = datasets[dd].get_subset(test_set)

        pred_prob_per_class[dd, :, :], pred_labels_per_rep_fs[dd, :], true_test_labels, \
        confmat, misclsfd_ids_this_run[dd], feature_importances[dd], best_params[dd] = \
            eval_optimized_model_on_testset(train_fs, test_fs, train_perc=train_perc, feat_sel_size=feat_sel_size,
                                            label_order_in_conf_matrix=label_set)

        accuracy_balanced[dd] = balanced_accuracy(confmat)
        confusion_matrix[:, :, dd] = confmat
        print('balanced accuracy: {:.4f} '.format(accuracy_balanced[dd]), end='')

        if num_classes == 2:
            # TODO FIX auc calculation flipped
            # TODO store fpr and tpr per trial, and provide the user to option to vizualize the average if they wish
            auc_weighted[dd] = roc_auc_score(true_test_labels, pred_prob_per_class[dd, :, pos_class_index],
                                             average='weighted')
            print('\t weighted AUC: {:.4f}'.format(auc_weighted[dd]), end='')

        print('')

    results_list = [pred_prob_per_class, pred_labels_per_rep_fs, true_test_labels, accuracy_balanced,
                    confusion_matrix, auc_weighted, feature_importances, best_params,
                    misclsfd_ids_this_run, test_set]

    out_path = pjoin(out_results_dir, 'trial_{}.pkl'.format(rep_proc_id))
    logging.info('results from rep {} saved to {}'.format(rep_proc_id, out_path))
    with open(out_path, 'bw') as of:
        pickle.dump(results_list, of)

    return results_list


def gather_results_across_trials(cv_results, common_ds, datasets, total_test_samples,
                                 num_repetitions, num_datasets, num_classes, num_features):
    "Reorganizes list of indiv CV trial results into rectangular arrays."

    pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, confusion_matrix, \
        accuracy_balanced, auc_weighted, best_params, feature_names, \
        feature_importances_per_rep, feature_importances_rf, \
        num_times_misclfd, num_times_tested = initialize_result_containers(common_ds, datasets,
                total_test_samples, num_repetitions, num_datasets, num_classes, num_features)

    for rep in range(num_repetitions):
        # unpacking each rep
        _rep_pred_prob_per_class, _rep_pred_labels_per_rep_fs, _rep_true_test_labels, \
            _rep_accuracy_balanced, _rep_confusion_matrix, _rep_auc_weighted, _rep_feature_importances, \
            _rep_best_params, _rep_misclsfd_ids_this_run, _rep_test_set = cv_results[rep]

        pred_prob_per_class[rep, :, :, :]   = _rep_pred_prob_per_class
        pred_labels_per_rep_fs[rep, :, :]   = _rep_pred_labels_per_rep_fs
        test_labels_per_rep[rep, :]         = _rep_true_test_labels
        accuracy_balanced[rep, :]           = _rep_accuracy_balanced
        confusion_matrix[rep, :, :, :]      = _rep_confusion_matrix
        auc_weighted[rep, :]                = _rep_auc_weighted
        best_params[rep]                    = _rep_best_params

        for dd in range(num_datasets):
            num_times_misclfd[dd].update(_rep_misclsfd_ids_this_run[dd])
            num_times_tested[dd].update(_rep_test_set)
            feature_importances_rf[dd][rep,:] = _rep_feature_importances[dd]

        # this variable is not being saved/used in any other way.
        feature_importances_per_rep[rep]    = _rep_feature_importances

    return pred_prob_per_class, pred_labels_per_rep_fs, test_labels_per_rep, confusion_matrix, \
        accuracy_balanced, auc_weighted, best_params, feature_names, \
        feature_importances_per_rep, feature_importances_rf, num_times_misclfd, num_times_tested


def summarize_perf(accuracy_balanced, auc_weighted, method_names, num_classes, num_datasets):
    """Prints median performance for each feature set"""

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='All-NaN slice encountered',
                                module='numpy', category=RuntimeWarning)

        # assuming the first column (axis=0) is over num_repititions
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

    return


if __name__ == '__main__':
    pass
