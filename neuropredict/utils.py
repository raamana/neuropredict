
import os
import sys
import re
import pickle
from neuropredict import config_neuropredict as cfg
import numpy as np
import os.path
from os.path import join as pjoin, exists as pexists, realpath
from multiprocessing import cpu_count
from time import localtime, strftime

__re_delimiters_word = '_|:|; |, |\*|\n'

def check_params_rhst(dataset_path_file, out_results_dir, num_repetitions, train_perc,
                      sub_groups, num_procs, grid_search_level,
                      classifier_name, feat_select_method):
    """Validates inputs and returns paths to feature sets to load"""

    if not pexists(dataset_path_file):
        raise IOError("File containing dataset paths does not exist.")

    with open(dataset_path_file, 'r') as dpf:
        dataset_paths = dpf.read().splitlines()
        # alert for duplicates
        if len(set(dataset_paths)) < len(dataset_paths):
            raise RuntimeError('Duplicate paths for input datasets found!\n'
                               'Try distinguish inputs further. '
                               'Otherwise report this bug at:'
                               'github.com/raamana/neuropredict/issues/new')
        # do not apply set(dataset_paths) to remove duplicates,
        # as set destroys current order, that is necessary to correspond to method_names

    try:
        out_results_dir = realpath(out_results_dir)
        os.makedirs(out_results_dir, exist_ok=True)
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

    num_procs = check_num_procs(num_procs)

    # removing empty elements
    if sub_groups is not None:
        sub_groups = [ group for group in sub_groups if group]
    # NOTE: here, we are not ensuring classes in all the subgroups actually exist in all datasets
    # that happens when loading data.

    if grid_search_level.lower() not in cfg.GRIDSEARCH_LEVELS:
        raise ValueError('Unrecognized level of grid search. Valid choices: {}'.format(cfg.GRIDSEARCH_LEVELS))

    classifier_name = check_classifier(classifier_name)

    if feat_select_method.lower() not in cfg.feature_selection_choices:
        raise ValueError('Feature selection method not recognized: {}\n '
                         'Implemented choices: {}'
                         ''.format(feat_select_method,
                                   cfg.feature_selection_choices))


    # printing the chosen options
    print('Training percentage      : {:.2}'.format(train_perc))
    print('Number of CV repetitions : {}'.format(num_repetitions))
    print('Classifier chosen        : {}'.format(classifier_name))
    print('Feature selection chosen : {}'.format(feat_select_method))
    print('Level of grid search     : {}'.format(grid_search_level))
    print('Number of processors     : {}'.format(num_procs))
    print('Saving the results to \n  {}'.format(out_results_dir))

    return dataset_paths, num_repetitions, num_procs, sub_groups


def check_classifier(clf_name=cfg.default_classifier):
    """Validates the classifier choice, and ensures necessary modules are installed."""

    clf_name = clf_name.lower()
    if clf_name not in cfg.classifier_choices:
        raise ValueError('Classifier not recognized : {}\n'
                         'Choose one of: {}'.format(clf_name, cfg.classifier_choices))

    if clf_name in cfg.additional_modules_reqd:
        try:
            from importlib import import_module
            import_module(cfg.additional_modules_reqd[clf_name])
        except ImportError:
            raise ImportError('choosing classifier {} requires installation of '
                              'another package. Try running\n'
                              'pip install -U {} '
                              ''.format(clf_name, cfg.additional_modules_reqd[clf_name]))

    return clf_name


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
    remaining = set(range(num_datasets))
    remaining.remove(common_ds_index)
    if num_datasets > 1:
        for idx in remaining:
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

    # displaying info on what is common across datasets
    common_ds.description = ' ' # this description is not reflective of all datasets
    dash_line = '-'*25
    print('\n{line}\nAll datasets contain:\n{ds:full}\n{line}\n'.format(line=dash_line, ds=common_ds))

    # choosing 'balanced' or 1/n_c for chance accuracy as training set is stratified
    print('Estimated chance accuracy : {:.3f}\n'.format(chance_accuracy(class_sizes, 'balanced')))

    num_features = np.zeros(num_datasets).astype(np.int64)
    for idx in range(num_datasets):
        num_features[idx] = datasets[idx].num_features

    return common_ds, class_set, label_set, class_sizes, num_samples, num_classes, num_datasets, num_features


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

    if not isinstance(class_sizes, np.ndarray):
        class_sizes = np.array(class_sizes)

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
    elif method in ['balanced', 'traditional']:
        chance_acc = 1 / num_classes
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

    return np.mean(indiv_class_acc)


def check_num_procs(requested_num_procs=cfg.DEFAULT_NUM_PROCS):
    "Ensures num_procs is finite and <= available cpu count."

    num_procs  = int(requested_num_procs)
    avail_cpu_count = int(cpu_count())

    def get_avail_slot_count(avail_cpu_count=avail_cpu_count):
        "Method to query HPC-specific number of slots available."

        from os import getenv

        hpc_num_procs_spec = [('SGE',   'JOB_ID',      'NSLOTS',        'slots'),
                              ('SLURM', 'SLURM_JOBID', 'SLURM_CPUS_PER_TASK',  'processors'),
                              ('PBS',   'PBS_JOBID',   'PBS_NUM_PPN',   'processors per node')]

        for hpc_env, id_jobid, var_slot_count, var_descr in hpc_num_procs_spec:
            if getenv(id_jobid):
                avail_cpu_count = int(getenv(var_slot_count, cpu_count()))
                print('{} recognized, job set up with {} {}.'.format(hpc_env, avail_cpu_count, var_descr))

        return avail_cpu_count


    avail_cpu_count = get_avail_slot_count()

    if num_procs < 1 or not np.isfinite(num_procs) or num_procs is None:
        num_procs = avail_cpu_count
        print('Invalid value for num_procs.')

    if num_procs > avail_cpu_count:
        print('# CPUs requested higher than available {}'.format(avail_cpu_count))
        num_procs = avail_cpu_count

    sys.stdout.flush()

    return num_procs


def save_options(options_to_save, out_dir_in):
    "Helper to save chosen options"

    sample_ids, classes, out_dir, user_feature_paths, user_feature_type, fs_subject_dir, \
        train_perc, num_rep_cv, positive_class, subgroups, feature_selection_size, num_procs, \
        grid_search_level, classifier_name, feat_select_method = options_to_save

    user_options = {
        'sample_ids'            : sample_ids,
        'classes'               : classes,
        'classifier_name'       : classifier_name,
        'feat_select_method'    : feat_select_method,
        'gs_level'              : grid_search_level,
        'feature_selection_size': feature_selection_size,
        'num_procs'             : num_procs,
        'num_rep_cv'            : num_rep_cv,
        'positive_class'        : positive_class,
        'sub_groups'            : subgroups,
        'train_perc'            : train_perc,
        'fs_subject_dir'        : fs_subject_dir,
        'user_feature_type'     : user_feature_type,
        'user_feature_paths'    : user_feature_paths,
        'out_dir'               : out_dir,}

    try:
        options_path = pjoin(out_dir_in, cfg.file_name_options)
        with open(options_path, 'wb') as opt_file:
            pickle.dump(user_options, opt_file)
    except:
        raise IOError('Unable to save the options to\n {}'.format(out_dir_in))

    return options_path


def load_options(out_dir, options_path=None):
    "Helper to load the saved options"


    if options_path is None:
        options_path = pjoin(out_dir, cfg.file_name_options)

    if not pexists(options_path):
        raise IOError('Filepath for options file does not exist:\n\t{}'
                         ''.format(options_path))

    try:
        with open(options_path, 'rb') as opt_file:
            user_options = pickle.load(opt_file)
    except:
        raise IOError('Unable to load the options from\n {}'.format(out_dir))

    return user_options


def check_paths(paths, path_type=''):
    "Converts path to absolute paths and ensures they all exist!"

    abs_paths = list(map(realpath, paths))
    for pp in abs_paths:
        if not pexists(pp):
            raise IOError("One of {} paths specified does not exist:\n {}".format(path_type, pp))

    return abs_paths


def validate_feature_selection_size(feature_select_method, dim_in_data=None):
    """
    Ensures method chosen for the type of computation for the size of reduced dimensionality.

    Parameters
    ----------
    feature_select_method
    dim_in_data

    Returns
    -------

    """


    try:
        # checking if its convertible to number
        num_select = np.float(feature_select_method)
    except:
        # if it is a str, it must be recognized
        if isinstance(feature_select_method, str):
            if feature_select_method.lower() in cfg.feature_selection_size_methods:
                num_select = feature_select_method.lower()
            else:
                raise ValueError('Invalid str choice {} - choose one of these:\n{}'
                                 ''.format(feature_select_method,
                                           cfg.feature_selection_size_methods))
        else:
            raise ValueError('Invalid choice {} for reduced dim size!\n'
                             'Choose an int between 1 and N-1, '
                             'or float between 0 and 1 (exclusive)\n'
                             'or one of these methods:\n{}'
                             ''.format(feature_select_method,
                                       cfg.feature_selection_size_methods))
    else:
        if dim_in_data is not None:
            upper_limit = np.Inf
        else:
            upper_limit = dim_in_data
        if not 0 < num_select < np.Inf:
            raise UnboundLocalError(
                'feature selection size out of bounds.\n'
                'Must be > 0 and < {}'.format(upper_limit))

    return num_select


def uniquify_in_order(seq):
    """Produces a list with unique elements in the same order as original sequence.

    https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def uniq_combined_name(method_names, max_len=50, num_char_each_word=1):
    "Function to produce a uniq, and not a long combined name. Recursive"

    combined_name = '_'.join(method_names)
    # depending on number and lengths of method_names, this can get very long
    if len(combined_name) > max_len:
        first_letters = list()
        for mname in method_names:
            this_FLs = ''.join([word[:num_char_each_word]
                                for word in re.split(__re_delimiters_word, mname)])
            first_letters.append(this_FLs)
        combined_name = '_'.join(uniquify_in_order(first_letters))

        if len(combined_name) > max_len:
            combined_name = uniq_combined_name(first_letters)

    return combined_name


def sub_group_identifier(group_names, sg_index=None, num_letters=3):
    """
    Constructs clean identifier to refer to a group of classes.
        Names will be sorted to allow for reproducibility.
        If there are too many classes or if the names are too long,
        only the first few letters of each word are used to generate the identifier.
    """

    group_names.sort()
    words = [ word for group in group_names
              for word in re.split(__re_delimiters_word, group) if word ]
    identifier = '_'.join(words)

    if len(identifier) > cfg.max_len_identifiers:
        # if there are too many groups, choosing the first few
        if len(group_names) >= cfg.max_len_identifiers:
            num_letters = 1
            words = words[:cfg.max_len_identifiers]

        # choosing first few letters from each word
        shorter_words = [word[:num_letters] for word in words]
        if sg_index is None:
            sg_index = 1
        identifier = 'subgroup{}_{}'.format(sg_index, '_'.join(shorter_words))

    return identifier


def make_numeric_labels(class_set):
    """Generates numeric labels (to feed external tools) for a given set of strings"""

    return { cls : idx for idx, cls in enumerate(class_set) }


def make_dataset_filename(method_name):
    "File name constructor."

    file_name = 'consolidated_{}_{}.MLDataset.pkl'.format(method_name, make_time_stamp())

    return file_name


def make_time_stamp():
    "Returns a timestamp string."

    # just by the hour
    return strftime('%Y%m%d-T%H', localtime())


def not_unspecified(var):
    """ Checks for null values of a give variable! """

    return var not in ['None', 'none', None, '']