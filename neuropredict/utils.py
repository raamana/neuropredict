import pickle
import re
from collections.abc import Iterable
from multiprocessing import cpu_count
from os.path import exists as pexists, join as pjoin, realpath
from time import localtime, strftime

import numpy as np
from neuropredict import config as cfg

__re_delimiters_word = '_|:|; |, |\*|\n'

def round_(array):
    """Shorthand for a rounding function with a controlled precision"""
    return np.round(array, cfg.PRECISION_METRICS)

def check_covariate_options(covar_list, covar_method):
    """Basic validation of covariate-related user options"""

    if covar_list is not None and not isinstance(covar_list, Iterable):
        raise ValueError('covariates can only be None or str or list of strings')
    # actual check for whether they exist in datasets under study will be done
    # after loading the datasets

    if isinstance(covar_list, str):
        covar_list = (covar_list, )

    covar_method = covar_method.lower()
    if covar_method not in cfg.avail_deconfounding_methods:
        raise ValueError('Unrecognized method to handle covarites/confounds.'
                         ' Must be one of {}'.format(cfg.avail_deconfounding_methods))

    return covar_list, covar_method


def check_covariates(multi_ds, covar_list, deconfounder):
    """Checks the existence of covariates in the given set of datasets"""

    if covar_list is not None:

        for covar in covar_list:
            if covar not in multi_ds.common_attr:
                raise AttributeError('Covariate {} does not exist in input datasets'
                                     ''.format(covar))
            num_set = len(multi_ds.common_attr[covar])
            if num_set < multi_ds.num_samplets:
                raise AttributeError('Covariate {} is only set for only {} of {} '
                                     'samplets! Double check and fix input datasets.'
                                     ''.format(covar, num_set, multi_ds.num_samplets))
    else:
        covar_list = ()

    # not doing anything with deconfounder for now
    # validity of covar data types for certain deconf methods can be checked here

    return covar_list, deconfounder


def check_classifier(clf_name=cfg.default_classifier):
    """Validates classifier choice, and ensures necessary modules are installed."""

    clf_name = clf_name.lower()
    if clf_name not in cfg.classifier_choices:
        raise ValueError('Predictive model not recognized : {}\n'
                         'Choose one of: {}'
                         ''.format(clf_name, cfg.classifier_choices))

    if clf_name in cfg.additional_modules_reqd:
        try:
            from importlib import import_module
            import_module(cfg.additional_modules_reqd[clf_name])
        except ImportError:
            raise ImportError('choosing classifier {} requires installation of '
                              'another package. Try running\n'
                              'pip install -U {} '
                              ''.format(clf_name,
                                        cfg.additional_modules_reqd[clf_name]))

    return clf_name


def check_regressor(est_name=cfg.default_regressor):
    """Validates classifier choice, and ensures necessary modules are installed."""

    est_name = est_name.lower()
    if est_name not in cfg.regressor_choices:
        raise ValueError('Predictive model not recognized : {}\n'
                         'Choose one of: {}'
                         ''.format(est_name, cfg.regressor_choices))

    if est_name in cfg.additional_modules_reqd:
        try:
            from importlib import import_module
            import_module(cfg.additional_modules_reqd[est_name])
        except ImportError:
            raise ImportError('choosing model {} requires installation of '
                              'another package. Try running\n'
                              'pip install -U {} '
                              ''.format(est_name,
                                        cfg.additional_modules_reqd[est_name]))

    return est_name


def get_cmap(name, length=None):
    """
    Helper to fetch a cmap of given length from the matplotlib ColorMapRegistry

    Parameters
    ----------
    name : `matplotlib.colors.Colormap` or str or None, default: None
        If a `.Colormap` instance, it will be returned. Otherwise, the name of
        a colormap known to Matplotlib, which will be resampled by *lut*. The
        default, None, means :rc:`image.cmap`.
    length : int or None, default: None
        If *name* is not already a Colormap instance and *lut* is not None, the
        colormap will be resampled to have *lut* entries in the lookup table.

    Returns
    -------
    Colormap
    """

    from matplotlib import colormaps

    if length is None:
        return colormaps[name]
    else:
        return colormaps[name].resampled(length)


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

        Both methods return similar results,
        with Zero Rule erring on the side higher chance accuracy.

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
                print('{} recognized, job set up with {} {}.'
                      ''.format(hpc_env, avail_cpu_count, var_descr))

        return avail_cpu_count


    avail_cpu_count = get_avail_slot_count()

    if num_procs < 1 or not np.isfinite(num_procs) or num_procs is None:
        num_procs = avail_cpu_count
        print('Invalid value for num_procs.')

    if num_procs > avail_cpu_count:
        print('# CPUs requested higher than available {}'.format(avail_cpu_count))
        num_procs = avail_cpu_count

    # sys.stdout.flush()

    return num_procs


def save_options(options_to_save, out_dir_in):
    "Helper to save chosen options"

    sample_ids, targets, out_dir, user_feature_paths, user_feature_type, \
    fs_subject_dir, train_perc, num_rep_cv, positive_class, subgroups, \
    reduced_dim_size, num_procs, grid_search_level, pred_model_name, \
    dim_red_method = options_to_save

    user_options = {
        'sample_ids'        : sample_ids,
        'targets'           : targets,
        'pred_model_name'   : pred_model_name,
        'dim_red_method'    : dim_red_method,
        'gs_level'          : grid_search_level,
        'reduced_dim_size'  : reduced_dim_size,
        'num_procs'         : num_procs,
        'num_rep_cv'        : num_rep_cv,
        'positive_class'    : positive_class,
        'sub_groups'        : subgroups,
        'train_perc'        : train_perc,
        'fs_subject_dir'    : fs_subject_dir,
        'user_feature_type' : user_feature_type,
        'user_feature_paths': user_feature_paths,
        'out_dir'           : out_dir, }

    try:
        options_path = pjoin(out_dir_in, cfg.file_name_options)
        with open(options_path, 'wb') as opt_file:
            pickle.dump(user_options, opt_file)
    except:
        raise IOError('Unable to save the options to\n {}'.format(out_dir_in))

    return user_options, options_path


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
            raise IOError("One of {} paths specified does not exist:"
                          "\n {}".format(path_type, pp))

    return abs_paths


def validate_feature_selection_size(feature_select_method, dim_in_data=None):
    """
    Validates the magnitude of reduced dimensionality or the method name.

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
        if dim_in_data is None:
            upper_limit = np.Inf
        else:
            upper_limit = dim_in_data
        if not 0 < num_select < upper_limit:
            raise UnboundLocalError(
                'feature selection size out of bounds.\n'
                'Must be > 0 and < {}'.format(upper_limit))

    return num_select


def validate_impute_strategy(user_choice):
    """Checks that user made a valid choice."""

    user_choice = user_choice.lower()
    if user_choice != cfg.default_imputation_strategy and \
            user_choice not in cfg.avail_imputation_strategies:
        raise ValueError('Unrecognized imputation strategy!\n\tchoose one of {}'
                         ''.format(cfg.avail_imputation_strategies))

    return user_choice


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

    file_name = 'consolidated_{}_{}.MLDataset.pkl' \
                ''.format(method_name, make_time_stamp())

    return file_name


def make_time_stamp():
    "Returns a timestamp string."

    # just by the hour
    return strftime('%Y%m%d-T%H', localtime())


def not_unspecified(var):
    """ Checks for null values of a give variable! """

    return var not in ['None', 'none', None, '']


def print_options(run_dir):
    """
    Prints options used in a previous run.

    Parameters
    ----------
    run_dir : str
        Path to a folder to with options from a previous run stored.

    """

    user_options = load_options(run_dir)

    # print(user_options)
    print('\n\nOptions used in the run\n{}\n'.format(run_dir))
    for key, val in user_options.items():
        if key.lower() not in ('sample_ids', 'targets'):
            print('{:>25} : {}'.format(key, val))

    return


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


def is_iterable_but_not_str(input_obj, min_length=1):
    """Boolean check for iterables that are not strings and of a minimum length"""

    if not (not isinstance(input_obj, str) and isinstance(input_obj, Iterable)):
        return False

    if len(input_obj) < min_length:
        return False
    else:
        return True


def median_of_medians(metric_array, axis=0):
    """Compute median of medians for each row/columsn"""

    if len(metric_array.shape) > 2:
        raise ValueError('Input array can only be 2D!')

    medians_along_axis = np.nanmedian(metric_array, axis=axis)

    return np.median(medians_along_axis)