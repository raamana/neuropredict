"""
neuropredict : easy and comprehensive predictive analysis.

"""
from __future__ import print_function

__all__ = ['run', 'cli', 'get_parser']

import argparse
import os
import sys
import textwrap
import traceback
import warnings
import matplotlib
import matplotlib.pyplot as plt
from sys import version_info
from os.path import join as pjoin, exists as pexists, abspath, realpath, basename
import numpy as np
from pyradigm import MLDataset

if version_info.major > 2:
    # the order of import is very important to avoid circular imports
    from neuropredict import __version__
    from neuropredict import config_neuropredict as cfg
    from neuropredict import rhst, visualize
    from neuropredict.freesurfer import aseg_stats_subcortical, aseg_stats_whole_brain
    from neuropredict.io import get_metadata, get_features, get_metadata_in_pyradigm, \
        get_data_matrix, get_dir_of_dirs, get_pyradigm, get_arff, saved_dataset_matches
    from neuropredict.utils import check_paths, uniq_combined_name, check_num_procs, \
        sub_group_identifier, save_options, load_options, validate_feature_selection_size, \
        make_dataset_filename, not_unspecified, check_classifier
else:
    raise NotImplementedError('neuropredict requires Python 3+.')


class NeuroPredictException(Exception):
    """Custom exception to distinguish neuropredict related errors (usage etc) from the usual."""
    pass

class MissingDataException(NeuroPredictException):
    """Custom exception to uniquely identify this error. Helpful for testing etc"""
    pass


def get_parser():
    "Parser to specify arguments and their defaults."

    parser = argparse.ArgumentParser(prog="neuropredict", formatter_class=argparse.RawTextHelpFormatter,
                                     description='Easy, standardized and comprehensive predictive analysis.')

    help_text_fs_dir = textwrap.dedent("""
    Absolute path to ``SUBJECTS_DIR`` containing the finished runs of Freesurfer parcellation
    Each subject will be queried after its ID in the metadata file.

    E.g. ``--fs_subject_dir /project/freesurfer_v5.3``
    \n \n """)

    help_text_user_defined_folder = textwrap.dedent("""
    List of absolute paths to user's own features.

    Format: Each of these folders contains a separate folder for each subject (named after its ID in the metadata file)
    containing a file called features.txt with one number per line.
    All the subjects (in a given folder) must have the number of features (#lines in file).
    Different parent folders (describing one feature set) can have different number of features for each subject,
    but they must all have the same number of subjects (folders) within them.

    Names of each folder is used to annotate the results in visualizations.
    Hence name them uniquely and meaningfully, keeping in mind these figures will be included in your papers.
    For example,

    .. parsed-literal::

        --user_feature_paths /project/fmri/ /project/dti/ /project/t1_volumes/

    Only one of ``--pyradigm_paths``, ``user_feature_paths``, ``data_matrix_path`` or ``arff_paths`` options can be specified.
    \n \n """)

    help_text_pyradigm_paths = textwrap.dedent("""
    Path(s) to pyradigm datasets.

    Each path is self-contained dataset identifying each sample, its class and features.
    \n \n """)

    help_text_data_matrix = textwrap.dedent("""
    List of absolute paths to text files containing one matrix of size N x p  (num_samples x num_features).

    Each row in the data matrix file must represent data corresponding to sample in the same row
    of the meta data file (meta data file and data matrix must be in row-wise correspondence).

    Name of this file will be used to annotate the results and visualizations.

    E.g. ``--data_matrix_paths /project/fmri.csv /project/dti.csv /project/t1_volumes.csv ``

    Only one of ``--pyradigm_paths``, ``user_feature_paths``, ``data_matrix_path`` or ``arff_paths`` options can be specified.
    
    File format could be
     - a simple comma-separated text file (with extension .csv or .txt): which can easily be read back with
        numpy.loadtxt(filepath, delimiter=',')
        or
     - a numpy array saved to disk (with extension .npy or .numpy) that can read in with numpy.load(filepath).

     One could use ``numpy.savetxt(data_array, delimiter=',')`` or ``numpy.save(data_array)`` to save features.

     File format is inferred from its extension.
     \n \n """)

    help_text_arff_paths = textwrap.dedent("""
    List of paths to files saved in Weka's ARFF dataset format.
    
    Note: 
     - this format does NOT allow IDs for each subject.
     - given feature values are saved in text format, this can lead to large files with high-dimensional data, 
        compared to numpy arrays saved to disk in binary format.
    
    More info: https://www.cs.waikato.ac.nz/ml/weka/arff.html
    \n \n """)

    help_text_positive_class = textwrap.dedent("""
    Name of the positive class (e.g. Alzheimers, MCI etc) to be used in calculation of area under the ROC curve.
    Applicable only for binary classification experiments.

    Default: class appearing last in order specified in metadata file.
    \n \n """)

    help_text_train_perc = textwrap.dedent("""
    Percentage of the smallest class to be reserved for training.

    Must be in the interval [0.01 0.99].

    If sample size is sufficiently big, we recommend 0.5.
    If sample size is small, or class imbalance is high, choose 0.8.
    \n \n """)

    help_text_num_rep_cv = textwrap.dedent("""
    Number of repetitions of the repeated-holdout cross-validation.

    The larger the number, more stable the estimates will be.
    \n \n """)

    help_text_sub_groups = textwrap.dedent("""
    This option allows the user to study different combinations of classes in a multi-class (N>2) dataset.

    For example, in a dataset with 3 classes CN, FTD and AD,
    two studies of pair-wise combinations can be studied separately
    with the following flag ``--sub_groups CN,FTD CN,AD``.
    This allows the user to focus on few interesting subgroups depending on their dataset/goal.

    Format: Different subgroups must be separated by space,
    and each sub-group must be a comma-separated list of class names defined in the meta data file.
    Hence it is strongly recommended to use class names without any spaces, commas, hyphens and special characters,
    and ideally just alphanumeric characters separated by underscores.

    Any number of subgroups can be specified, but each subgroup must have atleast two distinct classes.

    Default: ``'all'``, leading to inclusion of all available classes in a all-vs-all multi-class setting.
    \n \n """)

    help_text_metadata_file = textwrap.dedent("""
    Abs path to file containing metadata for subjects to be included for analysis.

    At the minimum, each subject should have an id per row followed by the class it belongs to.

    E.g.
    .. parsed-literal::

        sub001,control
        sub002,control
        sub003,disease
        sub004,disease

    \n \n """)

    help_text_feature_selection = textwrap.dedent("""Number of features to select as part of feature selection.
    Options:

         - 'tenth'
         - 'sqrt'
         - 'log2'
         - 'all'

    Default: \'tenth\' of the number of samples in the training set.

    For example, if your dataset has 90 samples, you chose 50 percent for training (default),
    then Y will have 90*.5=45 samples in training set, leading to 5 features to be selected for taining.
    If you choose a fixed integer, ensure all the feature sets under evaluation have atleast that many features.
    \n \n """)

    help_text_gs_level = textwrap.dedent("""
    Flag to specify the level of grid search during hyper-parameter optimization on the training set.
    Allowed options are : 'none', 'light' and 'exhaustive', in the order of how many values/values will be optimized. 
    
    More parameters and more values demand more resources and much longer time for optimization.
    
    The 'light' option tries to "folk wisdom" to try least number of values (no more than one or two),
     for the parameters for the given classifier. (e.g. a lage number say 500 trees for a random forest optimization). 
     The 'light' will be the fastest and should give a "rough idea" of predictive performance. 
     The 'exhaustive' option will try to most parameter values for the most parameters that can be optimized.
    """)

    help_text_make_vis = textwrap.dedent("""
    Option to make visualizations from existing results in the given path. 
    This is helpful when neuropredict failed to generate result figures automatically 
    e.g. on a HPC cluster, or another environment when DISPLAY is either not available.
    
    """)

    help_text_atlas = textwrap.dedent("""
    Name of the atlas to use for visualization. Default: fsaverage, if available.
    \n \n """)
    help_text_num_cpus = textwrap.dedent("""
    Number of CPUs to use to parallelize CV repetitions.

    Default : 4.

    Number of CPUs will be capped at the number available on the machine if higher is requested.
    \n \n """)

    help_text_out_dir = textwrap.dedent("""
    Output folder to store gathered features & results.
    \n \n """)

    help_classifier = textwrap.dedent("""
    
    String specifying one of the implemented classifiers. 
    (Classifiers are carefully chosen to allow for the comprehensive report provided by neuropredict).
    
    Default: 'RandomForestClassifier'
    
    """)

    help_feat_select_method = textwrap.dedent("""
    Feature selection method to apply prior to training the classifier.
    
    Default: 'VarianceThreshold', removing features with 0.001 percent of lowest variance (zeros etc).
    
    """)

    help_imputation_strategy = textwrap.dedent("""
    Strategy to impute any missing data (as encoded by NaNs).
    
    Default: 'raise', which raises an error if there is any missing data anywhere.
    Currently available imputation strategies are: {}
    
    """.format(cfg.avail_imputation_strategies))

    help_text_print_options = textwrap.dedent("""
    Prints the options used in the run in an output folder.
    
    """)

    parser.add_argument("-m", "--meta_file", action="store", dest="meta_file",
                        default=None, required=False, help=help_text_metadata_file)

    parser.add_argument("-o", "--out_dir", action="store", dest="out_dir", required=False, help=help_text_out_dir,
                        default=None )

    parser.add_argument("-f", "--fs_subject_dir", action="store", dest="fs_subject_dir",
                        default=None, help=help_text_fs_dir)

    user_defined = parser.add_argument_group(title='Input data and formats',
                                             description='Only one of the following types can be specified.')

    user_defined.add_argument("-y", "--pyradigm_paths", action="store", dest="pyradigm_paths",
                              nargs='+',  # to allow for multiple features
                              default=None,
                              help=help_text_pyradigm_paths)

    user_defined.add_argument("-u", "--user_feature_paths", action="store", dest="user_feature_paths",
                              nargs='+',  # to allow for multiple features
                              default=None,
                              help=help_text_user_defined_folder)

    user_defined.add_argument("-d", "--data_matrix_paths", action="store", dest="data_matrix_paths",
                              nargs='+',
                              default=None,
                              help=help_text_data_matrix)

    user_defined.add_argument("-a", "--arff_paths", action="store", dest="arff_paths",
                              nargs='+',
                              default=None,
                              help=help_text_arff_paths)

    cv_args_group = parser.add_argument_group(title='Cross-validation',
                                              description='Parameters related to training and optimization during cross-validation')
    cv_args_group.add_argument("-p", "--positive_class", action="store", dest="positive_class",
                        default=None,
                        help=help_text_positive_class)

    cv_args_group.add_argument("-t", "--train_perc", action="store", dest="train_perc",
                        default=cfg.default_train_perc,
                        help=help_text_train_perc)

    cv_args_group.add_argument("-n", "--num_rep_cv", action="store", dest="num_rep_cv",
                        default=cfg.default_num_repetitions,
                        help=help_text_num_rep_cv)

    cv_args_group.add_argument("-k", "--num_features_to_select", dest="num_features_to_select",
                        action="store", default=cfg.default_num_features_to_select,
                        help=help_text_feature_selection)

    cv_args_group.add_argument("-sg", "--sub_groups", action="store", dest="sub_groups",
                        nargs="*",
                        default="all",
                        help=help_text_sub_groups)

    cv_args_group.add_argument("-g", "--gs_level", action="store", dest="gs_level",
                        default="light", help=help_text_gs_level, choices=cfg.GRIDSEARCH_LEVELS)

    pipeline_group = parser.add_argument_group(title='Predictive Model',
                                              description='Parameters related to pipeline comprising the predictive model')

    pipeline_group.add_argument("-is", "--impute_strategy", action="store",
                                dest="impute_strategy",
                                default=cfg.default_imputation_strategy,
                                help=help_imputation_strategy,
                                choices=cfg.avail_imputation_strategies_with_raise)

    pipeline_group.add_argument("-fs", "--feat_select_method", action="store", dest="feat_select_method",
                                default=cfg.default_feat_select_method, help=help_feat_select_method,
                                choices=cfg.feature_selection_choices)

    pipeline_group.add_argument("-e", "--classifier", action="store", dest="classifier",
                                default=cfg.default_classifier, help=help_classifier,
                                choices=cfg.classifier_choices)

    vis_args = parser.add_argument_group(title='Visualization',
                                         description='Parameters related to generating visualizations')

    vis_args.add_argument("-z", "--make_vis", action="store", dest="make_vis",
                        default=None, help=help_text_make_vis)

    comp_args = parser.add_argument_group(title='Computing',
                                         description='Parameters related to computations/debugging')

    comp_args.add_argument("-c", "--num_procs", action="store", dest="num_procs",
                        default=cfg.DEFAULT_NUM_PROCS, help=help_text_num_cpus)

    comp_args.add_argument("--po", "--print_options", action="store",
                           dest="print_opt_dir",
                        default=False, help=help_text_print_options)

    comp_args.add_argument('-v', '--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    return parser


def print_options(run_dir):
    """
    Prints options used in a previous run.

    Parameters
    ----------
    run_dir : str
        Path to a folder to with options from a previous run stored.

    """

    from neuropredict.utils import load_options
    user_options = load_options(run_dir)

    # print(user_options)
    print('\n\nOptions used in the run\n{}\n'.format(run_dir))
    for key, val in user_options.items():
        if key.lower() not in ('sample_ids', 'classes'):
            print('{:>25} : {}'.format(key, val))

    return


def organize_inputs(user_args):
    """
    Validates the input features specified and returns organized list of paths and readers.

    Parameters
    ----------
    user_args : ArgParse object
        Various options specified by the user.

    Returns
    -------
    user_feature_paths : list
        List of paths to specified input features
    user_feature_type : str
        String identifying the type of user-defined input
    fs_subject_dir : str
        Path to freesurfer subject directory, if supplied.

    """

    atleast_one_feature_specified = False
    # specifying pyradigm avoids the need for separate meta data file
    meta_data_supplied = False
    meta_data_format = None

    if not_unspecified(user_args.fs_subject_dir):
        fs_subject_dir = abspath(user_args.fs_subject_dir)
        if not pexists(fs_subject_dir):
            raise IOError("Given Freesurfer directory doesn't exist.")
        atleast_one_feature_specified = True
    else:
        fs_subject_dir = None

    # ensuring only one type is specified
    mutually_excl_formats = ['user_feature_paths', 'data_matrix_paths', 'pyradigm_paths', 'arff_paths']
    not_none_count = 0
    for format in mutually_excl_formats:
        if not_unspecified(getattr(user_args, format)):
            not_none_count = not_none_count + 1
    if not_none_count > 1:
        raise ValueError('Only one of the following formats can be specified:\n{}'.format(mutually_excl_formats))

    if not_unspecified(user_args.user_feature_paths):
        user_feature_paths = check_paths(user_args.user_feature_paths, path_type='user defined (dir_of_dirs)')
        atleast_one_feature_specified = True
        user_feature_type = 'dir_of_dirs'

    elif not_unspecified(user_args.data_matrix_paths):
        user_feature_paths = check_paths(user_args.data_matrix_paths, path_type='data matrix')
        atleast_one_feature_specified = True
        user_feature_type = 'data_matrix'

    elif not_unspecified(user_args.pyradigm_paths):
        user_feature_paths = check_paths(user_args.pyradigm_paths, path_type='pyradigm')
        atleast_one_feature_specified = True
        meta_data_supplied = user_feature_paths[0]
        meta_data_format = 'pyradigm'
        user_feature_type = 'pyradigm'

    elif not_unspecified(user_args.arff_paths):
        user_feature_paths = check_paths(user_args.arff_paths, path_type='ARFF')
        atleast_one_feature_specified = True
        user_feature_type = 'arff'
        meta_data_supplied = user_feature_paths[0]
        meta_data_format = 'arff'
    else:
        user_feature_paths = None
        user_feature_type = None

    # map in python 3 returns a generator, not a list, so len() wouldnt work
    if not isinstance(user_feature_paths, list):
        user_feature_paths = list(user_feature_paths)

    if not atleast_one_feature_specified:
        raise ValueError('Atleast one method specifying features must be specified. '
                         'It can be a path(s) to pyradigm dataset, matrix file, user-defined folder or a Freesurfer subject directory.')

    return user_feature_paths, user_feature_type, fs_subject_dir, meta_data_supplied, meta_data_format


def parse_args():
    """Parser/validator for the cmd line args."""

    parser = get_parser()

    if len(sys.argv) < 2:
        print('Too few arguments!')
        parser.print_help()
        parser.exit(1)

    # parsing
    try:
        user_args = parser.parse_args()
    except:
        parser.exit(1)

    if len(sys.argv) == 3:
        # only if no features were specified to be assessed
        if not any(not_unspecified(getattr(user_args, attr))
               for attr in ('user_feature_paths', 'data_matrix_paths',
                            'pyradigm_paths', 'arff_paths')):

            if not_unspecified(user_args.print_opt_dir) and user_args.print_opt_dir:
                run_dir = realpath(user_args.print_opt_dir)
                print_options(run_dir)

            if not_unspecified(user_args.make_vis):
                out_dir = realpath(user_args.make_vis)
                res_path = pjoin(out_dir, cfg.file_name_results)
                if pexists(out_dir) and pexists(res_path):
                    if not_unspecified(user_args.make_vis):
                        print('\n\nSaving the visualizations to \n{}'.format(out_dir))
                        make_visualizations(res_path, out_dir)
                else:
                    raise ValueError('Given folder does not exist, or has no results!')

            sys.exit(0)

    user_feature_paths, user_feature_type, fs_subject_dir, meta_data_path, meta_data_format = organize_inputs(user_args)

    if not meta_data_path:
        if user_args.meta_file is not None:
            meta_file = abspath(user_args.meta_file)
            if not pexists(meta_file):
                raise IOError("Meta data file doesn't exist.")
        else:
            raise ValueError('Metadata file must be provided when not using pyradigm/ARFF inputs.')

        sample_ids, classes = get_metadata(meta_file)
    else:
        print('Using meta data from:\n{}'.format(meta_data_path))
        sample_ids, classes = get_metadata_in_pyradigm(meta_data_path, meta_data_format)

    if user_args.out_dir is not None:
        out_dir = realpath(user_args.out_dir)
    else:
        out_dir = pjoin(realpath(os.getcwd()), cfg.output_dir_default)

    try:
        os.makedirs(out_dir, exist_ok=True)
    except:
        raise IOError('Output folder could not be created.')

    train_perc = np.float32(user_args.train_perc)
    if not ( 0.01 <= train_perc <= 0.99):
        raise ValueError("Training percentage {} out of bounds - must be >= 0.01 and <= 0.99".format(train_perc))

    num_rep_cv = np.int64(user_args.num_rep_cv)
    if num_rep_cv < 10:
        raise ValueError("Atleast 10 repetitions of CV is recommened.")

    num_procs = check_num_procs(user_args.num_procs)

    class_set, subgroups, positive_class = validate_class_set(classes, user_args.sub_groups, user_args.positive_class)

    feature_selection_size = validate_feature_selection_size(user_args.num_features_to_select)

    impute_strategy = validate_impute_strategy(user_args.impute_strategy)

    grid_search_level = user_args.gs_level.lower()
    if grid_search_level not in cfg.GRIDSEARCH_LEVELS:
        raise ValueError('Unrecognized level of grid search. Valid choices: {}'.format(cfg.GRIDSEARCH_LEVELS))

    classifier = check_classifier(user_args.classifier)
    feat_select_method = user_args.feat_select_method.lower()

    # saving the validated and expanded values to disk for later use.
    options_to_save = [sample_ids, classes, out_dir, user_feature_paths,
                       user_feature_type, fs_subject_dir, train_perc, num_rep_cv,
                       positive_class, subgroups, feature_selection_size, num_procs,
                       grid_search_level, classifier, feat_select_method]
    options_path = save_options(options_to_save, out_dir)

    return sample_ids, classes, out_dir, options_path, \
           user_feature_paths, user_feature_type, fs_subject_dir, \
           train_perc, num_rep_cv, \
           positive_class, subgroups, \
           feature_selection_size, impute_strategy, num_procs, \
           grid_search_level, classifier, feat_select_method


def make_visualizations(results_file_path, out_dir, options_path=None):
    """
    Produces the performance visualizations/comparisons from the cross-validation results.

    Parameters
    ----------
    results_file_path : str
        Path to file containing results produced by `rhst`

    out_dir : str
        Path to a folder to store results.

    """

    results_dict = rhst.load_results_dict(results_file_path)

    # using shorter names for readability
    accuracy_balanced       = results_dict['accuracy_balanced']
    method_names            = results_dict['method_names']
    num_classes             = results_dict['num_classes']
    class_sizes             = results_dict['class_sizes']
    confusion_matrix        = results_dict['confusion_matrix']
    class_order             = results_dict['class_set']
    feature_importances_rf  = results_dict['feature_importances_rf']
    feature_names           = results_dict['feature_names']
    num_times_misclfd       = results_dict['num_times_misclfd']
    num_times_tested        = results_dict['num_times_tested']

    num_methods = len(method_names)
    if len(set(method_names)) < num_methods:
        method_names = ['m{}_{}'.format(ix,mn)
                        for ix, mn in enumerate(method_names)]

    feature_importances_available = True
    if options_path is not None:
        user_options = load_options(out_dir, options_path)
        if user_options['classifier_name'].lower() not in cfg.clfs_with_feature_importance:
            feature_importances_available = False
    else:
        # check if the all values are NaN
        unusable = [ np.all(np.isnan(method_fi.flatten())) for method_fi in feature_importances_rf ]
        feature_importances_available = not np.all(unusable)

    try:

        balacc_fig_path = pjoin(out_dir, 'balanced_accuracy')
        visualize.metric_distribution(accuracy_balanced, method_names, balacc_fig_path,
                                      class_sizes, num_classes, "Balanced Accuracy")

        confmat_fig_path = pjoin(out_dir, 'confusion_matrix')
        visualize.confusion_matrices(confusion_matrix, class_order, method_names, confmat_fig_path)

        cmp_misclf_fig_path = pjoin(out_dir, 'compare_misclf_rates')
        if num_classes > 2:
            visualize.compare_misclf_pairwise(confusion_matrix, class_order, method_names, cmp_misclf_fig_path)
        elif num_classes == 2:
            visualize.compare_misclf_pairwise_parallel_coord_plot(confusion_matrix, class_order, method_names,
                                                                  cmp_misclf_fig_path)

        if feature_importances_available:
            featimp_fig_path = pjoin(out_dir, 'feature_importance')
            visualize.feature_importance_map(feature_importances_rf, method_names, featimp_fig_path, feature_names)
        else:
            print('\nCurrent predictive model does not provide feature importance values. Skipping them.')

        misclf_out_path = pjoin(out_dir, 'misclassified_subjects')
        visualize.freq_hist_misclassifications(num_times_misclfd, num_times_tested, method_names, misclf_out_path)
    except:
        traceback.print_exc()
        warnings.warn('Error generating the visualizations! Skipping ..')

    # cleaning up
    plt.close('all')

    return


def validate_impute_strategy(user_choice):
    """Checks that user made a valid choice."""

    user_choice = user_choice.lower()
    if user_choice != cfg.default_imputation_strategy and \
            user_choice not in cfg.avail_imputation_strategies:
        raise ValueError('Unrecognized imputation strategy!\n\tchoose one of {}'
                         ''.format(cfg.avail_imputation_strategies))

    return user_choice


def validate_class_set(classes, subgroups, positive_class=None):
    "Ensures class names are valid and sub-groups exist."

    class_set = list(set(classes.values()))

    sub_group_list = list()
    if subgroups != 'all':
        if isinstance(subgroups, str):
            subgroups = [ subgroups, ]

        for comb in subgroups:
            cls_list = comb.split(',')
            # ensuring each subgroup has atleast two classes
            if len(set(cls_list)) < 2:
                raise ValueError('Subgroup {} does not contain 2 unique classes! '
                                 'Each subgroup must contain atleast two classes '
                                 'for classification experiments.'
                                 ''.format(comb))

            # verify each of them were defined in meta
            for cls in cls_list:
                if cls not in class_set:
                    raise ValueError("Class {} in combination {} "
                                     "does not exist in meta data.".format(cls, comb))

            sub_group_list.append(cls_list)
    else:
        # using all classes
        sub_group_list.append(class_set)

    # the following loop is required to preserve original order
    # this does not: class_order_in_meta = list(set(classes.values()))
    class_order_in_meta = list()
    for x in class_set:
        if x not in class_order_in_meta:
            class_order_in_meta.append(x)

    num_classes = len(class_order_in_meta)
    if num_classes < 2:
        raise ValueError("Atleast two classes are required for predictive analysis! "
                         "Only one given ({})".format(set(classes.values())))

    if num_classes == 2:
        if not_unspecified(positive_class):
            if positive_class not in class_order_in_meta:
                raise ValueError('Positive class specified does not exist in meta data.\n'
                                 'Choose one of {}'.format(class_order_in_meta))
            print('Positive class specified for AUC calculation: {}'.format(positive_class))
        else:
            positive_class = class_order_in_meta[-1]
            print('Positive class inferred for AUC calculation: {}'.format(positive_class))

    return class_set, sub_group_list, positive_class


def import_datasets(method_list, out_dir, subjects, classes,
                    feature_path, feature_type='dir_of_dirs',
                    user_impute_strategy=cfg.default_imputation_strategy):
    """
    Imports all the specified feature sets and organizes them into datasets.

    Parameters
    ----------
    method_list : list of callables
        Set of predefined methods returning a vector of features for a given sample id and location

    out_dir : str
        Path to the output folder

    subjects : list of str
        List of sample ids

    classes : dict
        Dict identifying the class for each sample id in the dataset.

    feature_path : list of str
        List of paths to the root directory containing the features (pre- or user-defined).
        Must be of same length as method_list

    feature_type : str
        a string identifying the structure of feature set.
        Choices = ('dir_of_dirs', 'data_matrix')

    user_impute_strategy : str
        Strategy to handle the missing data: whether to raise an error if data is missing, or
            to impute them using the method chosen here.

    Returns
    -------
    method_names : list of str
        List of method names used for annotation

    dataset_paths_file : str
        Path to the file containing paths to imported feature sets

    missing_data_flag : list
        List of boolean flags indicating whether data is missing in each of the input datasets.

    """

    def clean_str(string): return ' '.join(string.strip().split(' _-:\n\r\t'))

    from neuropredict.io import process_pyradigm, process_arff

    method_names = list()
    outpath_list = list()
    missing_data_flag = list() # boolean flag for each dataset

    for mm, cur_method in enumerate(method_list):
        if cur_method in [get_pyradigm]:

            method_name, out_path_cur_dataset = process_pyradigm(feature_path[mm], subjects, classes)

            # if feature_type in ['pyradigm']:
            #     loaded_dataset = MLDataset(filepath=feature_path[mm])
            # else:
            #     raise ValueError('Invalid state of the program!')
            #
            # if len(loaded_dataset.description) > 1:
            #     method_name = loaded_dataset.description
            # else:
            #     method_name = basename(feature_path[mm])
            #
            # method_names.append(clean_str(method_name))
            # if not saved_dataset_matches(loaded_dataset, subjects, classes):
            #     raise ValueError(
            #         'supplied pyradigm dataset does not match samples in the meta data.')
            # else:
            #     out_path_cur_dataset = feature_path[mm]

        elif cur_method in [get_arff]:

            method_name, out_path_cur_dataset = process_arff(feature_path[mm], subjects, classes,
                                                             out_dir)

            # loaded_dataset = MLDataset(arff_path=feature_path[mm])
            # if len(loaded_dataset.description) > 1:
            #     method_name = loaded_dataset.description
            # else:
            #     method_name = basename(feature_path[mm])
            #
            # method_names.append(clean_str(method_name))
            # out_name = make_dataset_filename(method_name)
            # out_path_cur_dataset = pjoin(out_dir, out_name)
            # loaded_dataset.save(out_path_cur_dataset)
        else:

            if cur_method in [get_dir_of_dirs]:
                method_name = basename(feature_path[mm])

            elif cur_method in [get_data_matrix]:
                method_name = os.path.splitext(basename(feature_path[mm]))[0]

            else:
                method_name = cur_method.__name__

            out_name = make_dataset_filename(method_name)

            out_path_cur_dataset = pjoin(out_dir, out_name)
            if not saved_dataset_matches(out_path_cur_dataset, subjects, classes):
                # noinspection PyTypeChecker
                out_path_cur_dataset = get_features(subjects, classes,
                                                    feature_path[mm],
                                                    out_dir, out_name,
                                                    cur_method, feature_type)

        # checking for presence of any missing data
        data_mat, targets, ids = MLDataset(filepath=out_path_cur_dataset).data_and_labels()
        is_nan = np.isnan(data_mat)
        if is_nan.any():
            data_missing_here = True
            num_sub_with_md = np.sum(is_nan.sum(axis=1) > 0)
            num_var_with_md = np.sum(is_nan.sum(axis=0) > 0)
            if user_impute_strategy == 'raise':
                raise MissingDataException(
                    '{}/{} subjects with missing data found in {}/{} features\n'
                    '\tin {} dataset at {}\n'
                    '\tFill them and rerun, '
                    'or choose one of the available imputation strategies: {}'
                    ''.format(num_sub_with_md, data_mat.shape[0],
                              num_var_with_md, data_mat.shape[1],
                              method_name, out_path_cur_dataset,
                              cfg.avail_imputation_strategies))
        else:
            data_missing_here = False

        method_names.append(clean_str(method_name))
        outpath_list.append(out_path_cur_dataset)
        missing_data_flag.append(data_missing_here)

    # finalizing the imputation strategy
    if any(missing_data_flag):
        print('\nOne or more of the input datasets have missing data!')
        if user_impute_strategy == 'raise':
            raise MissingDataException('Fill them and rerun, or choose one of the available '
                                       'imputation strategies: {}'
                                       ''.format(cfg.avail_imputation_strategies))
        else:
            impute_strategy = user_impute_strategy
            print('The imputation strategy chosen is: {}'.format(impute_strategy))
    else:
        # disabling the imputation altogether if there is no missing data
        impute_strategy = None
        if user_impute_strategy in ('raise', None):
            print('Ignoring imputation strategy chosen, as no missing data were found!')

    combined_name = uniq_combined_name(method_names)

    # checking if there are any duplicates
    if len(set(outpath_list)) < len(outpath_list):
        raise RuntimeError('Duplicate paths to input dataset found!\n'
                           'Try distinguish inputs further. Otherwise report this bug '
                           '@ github.com/raamana/neuropredict/issues/new')

    dataset_paths_file = pjoin(out_dir, 'datasetlist.' + combined_name + '.txt')
    with open(dataset_paths_file, 'w') as dpf:
        dpf.writelines('\n'.join(outpath_list))

    print('\nData import is done.\n\n')

    return method_names, dataset_paths_file, missing_data_flag, impute_strategy



def make_method_list(fs_subject_dir, user_feature_paths, user_feature_type='dir_of_dirs'):
    """
    Returns an organized list of feature paths and methods to read in features.

    Parameters
    ----------
    fs_subject_dir : str
    user_feature_paths : list of str
    user_feature_type : str

    Returns
    -------
    feature_dir : list
    method_list : list


    """

    freesurfer_readers = [aseg_stats_subcortical, aseg_stats_whole_brain]
    userdefined_readers = {'dir_of_dirs': get_dir_of_dirs,
                           'data_matrix': get_data_matrix,
                           'pyradigm': get_pyradigm,
                           'arff': get_arff}

    feature_dir = list()
    method_list = list()
    if not_unspecified(user_feature_paths):
        if user_feature_type not in userdefined_readers:
            raise NotImplementedError("Invalid feature type or its reader is not implemented yet!")

        for upath in user_feature_paths:
            feature_dir.append(upath)
            method_list.append(userdefined_readers[user_feature_type])

    if not_unspecified(fs_subject_dir):
        for fsrdr in freesurfer_readers:
            feature_dir.append(fs_subject_dir)
            method_list.append(fsrdr)

    if len(method_list) != len(feature_dir):
        raise ValueError('Invalid specification for features!')

    if len(method_list) < 1:
        raise ValueError('Atleast one feature set must be specified.')

    print("\nRequested features for analysis:")
    for mm, method in enumerate(method_list):
        print("{} from {}".format(method.__name__, feature_dir[mm]))

    return feature_dir, method_list


def prepare_and_run(subjects, classes, out_dir, options_path,
                    user_feature_paths, user_feature_type, fs_subject_dir,
                    train_perc, num_rep_cv, positive_class,
                    sub_group_list, feature_selection_size, user_impute_strategy, num_procs,
                    grid_search_level, classifier, feat_select_method):
    "Organizes the inputs and prepares them for CV"

    feature_dir, method_list = make_method_list(fs_subject_dir, user_feature_paths,
                                                user_feature_type)

    method_names, dataset_paths_file, missing_flag, impute_strategy \
        = import_datasets(method_list, out_dir, subjects, classes,
                          feature_dir, user_feature_type, user_impute_strategy)

    print('Requested processing for the following subgroups:'
          '\n{}\n'.format('\n'.join([','.join(sg) for sg in sub_group_list])))

    # iterating through the given set of subgroups
    num_sg = len(sub_group_list)
    for sgi, sub_group in enumerate(sub_group_list):
        print('{}\nProcessing subgroup : {} ({}/{})'
              '\n{}'.format('-'*80, ','.join(sub_group), sgi+1, num_sg, '-'*80))
        out_dir_sg = pjoin(out_dir, sub_group_identifier(sub_group, sg_index=sgi+1))
        results_file_path = rhst.run(dataset_paths_file, method_names, out_dir_sg,
                                     train_perc=train_perc, num_repetitions=num_rep_cv,
                                     positive_class=positive_class, sub_group=sub_group,
                                     feat_sel_size=feature_selection_size,
                                     impute_strategy=impute_strategy,
                                     missing_flag=missing_flag,
                                     num_procs=num_procs,
                                     grid_search_level=grid_search_level,
                                     classifier_name=classifier,
                                     feat_select_method=feat_select_method,
                                     options_path=options_path)

        print('\n\nSaving the visualizations to \n{}'.format(out_dir))
        make_visualizations(results_file_path, out_dir_sg, options_path)
        print('\n')

    return


def cli():
    """
    Main entry point.

    """

    subjects, classes, out_dir, options_path, user_feature_paths, user_feature_type, \
        fs_subject_dir, train_perc, num_rep_cv, positive_class, sub_group_list, \
        feature_selection_size, impute_strategy, num_procs, \
        grid_search_level, classifier, feat_select_method = parse_args()

    print('Running neuropredict {}'.format(__version__))
    prepare_and_run(subjects, classes, out_dir, options_path,
                    user_feature_paths, user_feature_type, fs_subject_dir,
                    train_perc, num_rep_cv, positive_class,
                    sub_group_list, feature_selection_size, impute_strategy, num_procs,
                    grid_search_level, classifier, feat_select_method)

    return


def run(feature_sets,
        feature_type=cfg.default_feature_type,
        meta_data=None,
        output_dir=None,
        pipeline=None,
        train_perc=0.5,
        num_repetitions=200,
        positive_class=None,
        feat_sel_size=cfg.default_num_features_to_select,
        sub_groups='all',
        grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT,
        num_procs=2):
    """
    Generate comprehensive report on the predictive performance for different feature sets and statistically compare them.

    Main entry point for API access.

    Parameters
    ----------
    feature_sets : list
        The input can be specified in either of the following ways:
            - list of paths to pyradigm datasets saved on disk
            - path to a file containing list of paths (each line containing path to a valid MLDataset)
            - list of MLDatasets that are already loaded
            - list of tuples (to specify multiple features), each element containing (X, y) i.e. data and target labels
            - a single tuple containing (X, y) i.e. data and target labels
            - list of paths to CSV files, each containing one type of features.

            When specifying multiple sets of input features, ensure:
            - all of them contain the same number of samples
            - each sample belongs to same class across all feature sets.

    feature_type : str
        String identifying the type of features as described above. It could be:
        'list_of_pyradigm_paths', 'pyradigm_list',
        'list_of_tuples', 'tuple', 'list_of_csv_paths'

    meta_data : multiple
        The meta data can be specified in either of the following ways:

            - a path to a meta data file (see :doc:`features` page)
            - a dict keyed in by sample IDs with values representing their classes.
            - None, if meta data is already specified in ``feature_sets`` input (e.g. with pyradigms).

    pipeline : str or object
        If a string, it identifying one of the implemented classifiers e.g. 'RandomForestClassifier' or 'ExtraTreesClassifier'
        If an object, it must be a sciki-learn pipeline describing the sequence of steps.
        This is typically a set of feature selections or dimensionality reduction steps followed by an estimator (classifier).

        See http://scikit-learn.org/stable/modules/pipeline.html#pipeline for more details.

        Default: None, which leads to the selection of a Random Forest classifier,
         with robust scaling, followed by removal of low variance features.

    method_names : list
        A list of names to denote the different feature sets

    out_results_dir : str
        Path to output directory to save the cross validation results to.
        If not specified, a new directory named 'neuropredict' will be created in the current directory.

    train_perc : float, optional
        Percetange of subjects to train the classifier on.
        The percentage is applied to the size of the smallest class to estimate
        the number of subjects from each class to be reserved for training.
        The smallest class is chosen to avoid class-imbalance in the training set.
        Default: 0.8 (80%).

    positive_class : str
        Name of the class to be treated as positive in calculation of AUC

    feat_sel_size : str or int
        Number of features to select as part of feature selection. Options:

             - 'tenth'
             - 'sqrt'
             - 'log2'
             - 'all'

            Default: \'tenth\' of the number of samples in the training set. For example, if your dataset has 90 samples, you chose 50 percent for training (default),  then Y will have 90*.5=45 samples in training set, leading to 5 features to be selected for taining. If you choose a fixed integer, ensure all the feature sets under evaluation have atleast that many features.

    num_repetitions : int, optional
        Number of repetitions of cross-validation estimation. Default: 200.

    num_procs : int, optional
        Number of CPUs to use to parallelize CV repetitions.

        Default : 4. Number of CPUs will be capped at the number available on the machine if higher is requested.

    sub_groups : list
        This option allows the user to study different combinations of classes in a multi-class (N>2) dataset. For example, in a dataset with 3 classes CN, FTD and AD, two studies of pair-wise combinations can be studied separately with the following flag ``--sub_groups CN,FTD CN,AD``. This allows the user to focus on few interesting subgroups depending on their dataset/goal.

        Format: Different subgroups must be separated by space, and each sub-group must be a comma-separated list of class names defined in the meta data file. Hence it is strongly recommended to use class names without any spaces, commas, hyphens and special characters, and ideally just alphanumeric characters separated by underscores. Any number of subgroups can be specified, but each subgroup must have atleast two distinct classes.

        Default: ``'all'``, leading to inclusion of all available classes in a all-vs-all multi-class setting.

    grid_search_level : str
        Flag to specify the level of grid search during hyper-parameter optimization on the training set.
        Allowed options are : 'none', 'light' and 'exhaustive', in the order of how many values/values will be optimized.

        More parameters and more values demand more resources and much longer time for optimization.

        The 'light' option tries to "folk wisdom" to try least number of values (no more than one or two),
         for the parameters for the given classifier. (e.g. a lage number say 500 trees for a random forest optimization).
         The 'light' will be the fastest and should give a "rough idea" of predictive performance.
         The 'exhaustive' option will try to most parameter values for the most parameters that can be optimized.

    Returns
    -------
    results_path : str
        Path to pickle file containing full set of CV results.

    """

    raise NotImplementedError

    return

if __name__ == '__main__':
    cli()
