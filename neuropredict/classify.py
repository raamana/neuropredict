"""Module to define classification-oriented workflows and methods"""

import os
import sys
import textwrap
import traceback
import warnings
from os.path import abspath, basename, exists as pexists, join as pjoin, realpath
from os import makedirs
import matplotlib.pyplot as plt
import numpy as np
# the order of import is very important to avoid circular imports
from neuropredict import __version__, config_neuropredict as cfg, rhst, visualize
from neuropredict.base import BaseWorkflow, organize_inputs
from neuropredict.freesurfer import (aseg_stats_subcortical,
                                     aseg_stats_whole_brain)
from neuropredict.datasets import load_datasets, detect_missing_data
from neuropredict.io import (get_arff, get_data_matrix, get_dir_of_dirs,
                             get_features, get_metadata, get_metadata_in_pyradigm,
                             get_pyradigm, saved_dataset_matches,
                             process_pyradigm, process_arff)
from neuropredict.utils import (check_classifier, check_num_procs, load_options,
                                make_dataset_filename, not_unspecified,
                                print_options, save_options,
                                sub_group_identifier, uniq_combined_name,
                                uniquify_in_order,
                                validate_feature_selection_size,
                                validate_impute_strategy)
from pyradigm.multiple import MultiDatasetClassify
from pyradigm.utils import load_dataset
from sklearn.metrics import confusion_matrix, roc_auc_score

def auc_weighted(true_labels, predicted_proba_per_class):
    """Wrapper around sklearn roc_auc_score to ensure it is weighted."""

    return roc_auc_score(true_labels, predicted_proba_per_class, average='weighted')

auc_metric_name = auc_weighted.__name__
predict_proba_name = 'predict_proba'

class ClassificationWorkflow(BaseWorkflow):
    """
    Class defining an neuropredict experiment to be run.

    Encapsulates all the details necessary for execution,
        hence easing the save/load/decide workflow.

    """


    def __init__(self,
                 datasets,
                 pred_model=cfg.default_classifier,
                 impute_strategy=cfg.default_imputation_strategy,
                 dim_red_method=cfg.default_dim_red_method,
                 reduced_dim=cfg.default_reduced_dim_size,
                 train_perc=cfg.default_train_perc,
                 num_rep_cv=cfg.default_num_repetitions,
                 scoring=cfg.default_scoring_metric,
                 positive_class=None,
                 grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT,
                 out_dir=None,
                 num_procs=cfg.DEFAULT_NUM_PROCS,
                 user_options=None,
                 checkpointing=cfg.default_checkpointing):
        super().__init__(datasets,
                         pred_model=pred_model,
                         impute_strategy=impute_strategy,
                         dim_red_method=dim_red_method,
                         reduced_dim=reduced_dim,
                         train_perc=train_perc,
                         num_rep_cv=num_rep_cv,
                         scoring=scoring,
                         grid_search_level=grid_search_level,
                         num_procs=num_procs,
                         user_options=user_options,
                         checkpointing=checkpointing)

        self.out_dir = out_dir
        makedirs(self.out_dir, exist_ok=True)

        # order of target_set is crucial, for AUC computation as well as confusion
        # matrix row/column, hence making it a tuple to prevent accidental mutation
        #  ordering them in order of their appearance in also important: uniquify!
        self._target_set = tuple(uniquify_in_order(self.datasets.targets.values()))
        self._positive_class, self._positive_class_index = \
            check_positive_class(self._target_set, positive_class)


    def _eval_predictions(self, pipeline, test_data, true_targets, run_id, ds_id):
        """
        Evaluate predictions and perf estimates to results class.

        Prints a quick summary too, as an indication of progress.
        """

        predicted_targets = pipeline.predict(test_data)
        self.results.add(run_id, ds_id, predicted_targets, true_targets)

        if hasattr(pipeline, predict_proba_name):
            predicted_prob = pipeline.predict_proba(test_data)
            self.results.add_attr(run_id, ds_id, predict_proba_name, predicted_prob)
            # TODO it is possible the column order in predicted_prob may not match
            #   the order in self._target_set
            auc = auc_weighted(true_targets,
                               predicted_prob[:,self._positive_class_index])
            self.results.add_metric(run_id, ds_id, auc_metric_name, auc)

        conf_mat = confusion_matrix(true_targets, predicted_targets,
                                    labels=self._target_set)  # to control row order
        self.results.add_diagnostics(conf_mat,
                                     true_targets[predicted_targets != true_targets])


    def save(self):
        """Method to save the workflow and results."""

        print('\n\n---\nNOT SAVING RESULTS DURING DEV VERSION\n----\n\n')


    def load(self):
        """Mechanism to reload results.

        Useful for check-pointing, and restore upon crash etc
        """

        raise NotImplementedError()


    def summarize(self):
        """Simple summary of the results produced, for logging and user info"""

        print('\n\n---\nNOT SUMMARIZING RESULTS DURING DEV VERSION\n----\n\n')


    def visualize(self):
        """Method to produce all the relevant visualizations based on the results
        from this workflow."""

        raise NotImplementedError()


def make_visualizations(results_file_path, out_dir, options_path=None):
    """
    Produces the performance visualizations/comparison plots from the
    cross-validation results.

    Parameters
    ----------
    results_file_path : str
        Path to file containing results produced by `rhst`

    out_dir : str
        Path to a folder to store results.

    """

    results_dict = rhst.load_results_dict(results_file_path)

    # using shorter names for readability
    accuracy_balanced = results_dict['accuracy_balanced']
    method_names = results_dict['method_names']
    num_classes = results_dict['num_classes']
    class_sizes = results_dict['target_sizes']
    confusion_matrix = results_dict['confusion_matrix']
    class_order = results_dict['class_set']
    feature_importances_rf = results_dict['feature_importances_rf']
    feature_names = results_dict['feature_names']
    num_times_misclfd = results_dict['num_times_misclfd']
    num_times_tested = results_dict['num_times_tested']

    num_methods = len(method_names)
    if len(set(method_names)) < num_methods:
        method_names = ['m{}_{}'.format(ix, mn)
                        for ix, mn in enumerate(method_names)]

    feature_importances_available = True
    if options_path is not None:
        user_options = load_options(out_dir, options_path)
        if user_options['classifier_name'].lower() not in \
                cfg.clfs_with_feature_importance:
            feature_importances_available = False
    else:
        # check if the all values are NaN
        unusable = [np.all(np.isnan(method_fi.flatten()))
                    for method_fi in feature_importances_rf]
        feature_importances_available = not np.all(unusable)

    try:

        balacc_fig_path = pjoin(out_dir, 'balanced_accuracy')
        visualize.metric_distribution(accuracy_balanced, method_names,
                                      balacc_fig_path, class_sizes, num_classes,
                                      "Balanced Accuracy")

        confmat_fig_path = pjoin(out_dir, 'confusion_matrix')
        visualize.confusion_matrices(confusion_matrix, class_order, method_names,
                                     confmat_fig_path)

        cmp_misclf_fig_path = pjoin(out_dir, 'compare_misclf_rates')
        if num_classes > 2:
            visualize.compare_misclf_pairwise(confusion_matrix, class_order,
                                              method_names, cmp_misclf_fig_path)
        elif num_classes == 2:
            visualize.compare_misclf_pairwise_parallel_coord_plot(confusion_matrix,
                                                                  class_order,
                                                                  method_names,
                                                                  cmp_misclf_fig_path)

        if feature_importances_available:
            featimp_fig_path = pjoin(out_dir, 'feature_importance')
            visualize.feature_importance_map(feature_importances_rf, method_names,
                                             featimp_fig_path, feature_names)
        else:
            print('\nCurrent predictive model, and/or dimensionality reduction'
                  ' method, does not provide (or allow for computing) feature'
                  ' importance values. Skipping them.')

        misclf_out_path = pjoin(out_dir, 'misclassified_subjects')
        visualize.freq_hist_misclassifications(num_times_misclfd, num_times_tested,
                                               method_names, misclf_out_path)
    except:
        traceback.print_exc()
        warnings.warn('Error generating the visualizations! Skipping ..')

    # cleaning up
    plt.close('all')

    return


def get_parser_classify():
    """"""

    from neuropredict.base import get_parser_base

    parser, user_defined, cv_args_group, pipeline_group, vis_args, comp_args \
        = get_parser_base()

    help_text_fs_dir = textwrap.dedent("""
    Absolute path to ``SUBJECTS_DIR`` containing the finished runs of 
    Freesurfer parcellation. Each subject will be queried after its ID in the 
    metadata file. E.g. ``--fs_subject_dir /project/freesurfer_v5.3``
    \n \n """)

    help_text_arff_paths = textwrap.dedent("""
    List of paths to files saved in Weka's ARFF dataset format.

    Note: 
     - this format does NOT allow IDs for each subject.
     - given feature values are saved in text format, this can lead to large files 
     with high-dimensional data, 
        compared to numpy arrays saved to disk in binary format.

    More info: https://www.cs.waikato.ac.nz/ml/weka/arff.html
    \n \n """)

    help_text_positive_class = textwrap.dedent("""
    Name of the positive class (e.g. Alzheimers, MCI etc) to be used in 
    calculation of area under the ROC curve. This is applicable only for binary 
    classification experiments.

    Default: class appearing last in order specified in metadata file.
    \n \n """)

    help_text_sub_groups = textwrap.dedent("""
    This option allows the user to study different combinations of classes in a 
    multi-class (N>2) dataset.

    For example, in a dataset with 3 classes CN, FTD and AD, two studies of 
    pair-wise combinations can be studied separately with the following flag 
    ``--sub_groups CN,FTD CN,AD``. This allows the user to focus on few 
    interesting subgroups depending on their dataset/goal.

    Format: Different subgroups must be separated by space, and each sub-group 
    must be a comma-separated list of class names defined in the meta data file. 
    Hence it is strongly recommended to use class names without any spaces, 
    commas, hyphens and special characters, and ideally just alphanumeric 
    characters separated by underscores.

    Any number of subgroups can be specified, but each subgroup must have atleast 
    two distinct classes.

    Default: ``'all'``, leading to inclusion of all available classes in a 
    all-vs-all multi-class setting.
    \n \n """)

    help_classifier = textwrap.dedent("""

    String specifying one of the implemented classifiers. 
    (Classifiers are carefully chosen to allow for the comprehensive report 
    provided by neuropredict).

    Default: 'RandomForestClassifier'

    """)

    parser.add_argument("-f", "--fs_subject_dir", action="store",
                        dest="fs_subject_dir",
                        default=None, help=help_text_fs_dir)

    user_defined.add_argument("-a", "--arff_paths", action="store",
                              dest="arff_paths",
                              nargs='+',
                              default=None,
                              help=help_text_arff_paths)

    cv_args_group.add_argument("-p", "--positive_class", action="store",
                               dest="positive_class",
                               default=None,
                               help=help_text_positive_class)

    cv_args_group.add_argument("-sg", "--sub_groups", action="store",
                               dest="sub_groups",
                               nargs="*",
                               default="all",
                               help=help_text_sub_groups)

    pipeline_group.add_argument("-e", "--classifier", action="store",
                                dest="classifier",
                                default=cfg.default_classifier, help=help_classifier,
                                choices=cfg.classifier_choices, type=str.lower)

    return parser


def parse_args():
    """Parser/validator for the cmd line args."""

    parser = get_parser_classify()

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
                        print(
                            '\n\nSaving the visualizations to \n{}'.format(out_dir))
                        make_visualizations(res_path, out_dir)
                else:
                    raise ValueError('Given folder does not exist, '
                                     'or has no results file!')

            sys.exit(0)

    user_feature_paths, user_feature_type, fs_subject_dir, meta_data_path, \
    meta_data_format = organize_inputs(user_args)

    if not meta_data_path:
        if user_args.meta_file is not None:
            meta_file = abspath(user_args.meta_file)
            if not pexists(meta_file):
                raise IOError("Meta data file doesn't exist.")
        else:
            raise ValueError('Metadata file must be provided '
                             'when not using pyradigm/ARFF inputs.')

        sample_ids, classes = get_metadata(meta_file)
    else:
        print('Using meta data from:\n{}'.format(meta_data_path))
        sample_ids, classes = get_metadata_in_pyradigm(meta_data_path,
                                                       meta_data_format)

    if user_args.out_dir is not None:
        out_dir = realpath(user_args.out_dir)
    else:
        out_dir = pjoin(realpath(os.getcwd()), cfg.output_dir_default)

    try:
        os.makedirs(out_dir, exist_ok=True)
    except:
        raise IOError('Output folder could not be created.')

    train_perc = np.float32(user_args.train_perc)
    if not (0.01 <= train_perc <= 0.99):
        raise ValueError("Training percentage {} out of bounds "
                         "- must be >= 0.01 and <= 0.99".format(train_perc))

    num_rep_cv = np.int64(user_args.num_rep_cv)
    if num_rep_cv < 10:
        raise ValueError("Atleast 10 repetitions of CV is recommened.")

    num_procs = check_num_procs(user_args.num_procs)

    class_set, subgroups, positive_class = validate_class_set(classes,
                                                              user_args.sub_groups,
                                                              user_args.positive_class)

    feature_selection_size = validate_feature_selection_size(
            user_args.reduced_dim_size)

    impute_strategy = validate_impute_strategy(user_args.impute_strategy)

    grid_search_level = user_args.gs_level.lower()
    if grid_search_level not in cfg.GRIDSEARCH_LEVELS:
        raise ValueError('Unrecognized level of grid search. Valid choices: {}'
                         ''.format(cfg.GRIDSEARCH_LEVELS))

    classifier = check_classifier(user_args.classifier)
    dim_red_method = user_args.dim_red_method.lower()

    # saving the validated and expanded values to disk for later use.
    options_to_save = [sample_ids, classes, out_dir, user_feature_paths,
                       user_feature_type, fs_subject_dir, train_perc, num_rep_cv,
                       positive_class, subgroups, feature_selection_size, num_procs,
                       grid_search_level, classifier, dim_red_method]
    options_path = save_options(options_to_save, out_dir)

    return sample_ids, classes, out_dir, options_path, \
           user_feature_paths, user_feature_type, fs_subject_dir, \
           train_perc, num_rep_cv, \
           positive_class, subgroups, \
           feature_selection_size, impute_strategy, num_procs, \
           grid_search_level, classifier, dim_red_method


def validate_class_set(classes, subgroups, positive_class=None):
    "Ensures class names are valid and sub-groups exist."

    class_set = list(set(classes.values()))

    sub_group_list = list()
    if subgroups != 'all':
        if isinstance(subgroups, str):
            subgroups = [subgroups, ]

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
                                     "does not exist in meta data.".format(cls,
                                                                           comb))

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
                raise ValueError('Positive class specified does not exist in meta '
                                 'data.\n Choose one of {}'
                                 ''.format(class_order_in_meta))
            print('Positive class specified for AUC calculation: {}'
                  ''.format(positive_class))
        else:
            positive_class = class_order_in_meta[-1]
            print('Positive class inferred for AUC calculation: {}'
                  ''.format(positive_class))

    return class_set, sub_group_list, positive_class


def check_positive_class(class_set, positive_class=None):
    """Checks the provided positive class, and returns its index"""

    if positive_class is None:
        positive_class = class_set[-1]
    elif positive_class not in class_set:
        raise ValueError('Chosen positive class {} does not exist in the dataset,'
                         ' with classes {}'.format(positive_class, class_set))
    pos_class_index = class_set.index(positive_class)

    return positive_class, pos_class_index


def make_method_list(fs_subject_dir, user_feature_paths,
                     user_feature_type='dir_of_dirs'):
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
                           'pyradigm'   : get_pyradigm,
                           'arff'       : get_arff}

    feature_dir = list()
    method_list = list()
    if not_unspecified(user_feature_paths):
        if user_feature_type not in userdefined_readers:
            raise NotImplementedError("Invalid feature type or "
                                      "its reader is not implemented yet!")

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


def import_datasets(method_list, out_dir, subjects, classes,
                    feature_path, feature_type='dir_of_dirs'):
    """
    Imports all the specified feature sets and organizes them into datasets.

    Parameters
    ----------
    method_list : list of callables
        Set of predefined methods returning a vector of features f
        or a given sample id and location

    out_dir : str
        Path to the output folder

    subjects : list of str
        List of sample ids

    classes : dict
        Dict identifying the class for each sample id in the dataset.

    feature_path : list of str
        List of paths to the root folder containing features (pre- or user-defined).
        Must be of same length as method_list

    feature_type : str
        a string identifying the structure of feature set.
        Choices = ('dir_of_dirs', 'data_matrix')

    user_impute_strategy : str
        Strategy to handle the missing data:
        whether to raise an error if data is missing,
        or to impute them using the method chosen here.

    Returns
    -------
    method_names : list of str
        List of method names used for annotation

    dataset_paths_file : str
        Path to the file containing paths to imported feature sets

    missing_data_flag : list
        List of boolean flags
        indicating whether data is missing in each of the input datasets.

    """

    def clean_str(string):
        return ' '.join(string.strip().split(' _-:\n\r\t'))

    method_names = list()
    outpath_list = list()

    for mm, cur_method in enumerate(method_list):
        if cur_method in [get_pyradigm]:

            method_name, out_path_cur_dataset = process_pyradigm(feature_path[mm],
                                                                 subjects, classes)
        elif cur_method in [get_arff]:
            method_name, out_path_cur_dataset = process_arff(
                    feature_path[mm], subjects, classes, out_dir)
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

        method_names.append(clean_str(method_name))
        outpath_list.append(out_path_cur_dataset)

    # checking if there are any duplicates
    if len(set(outpath_list)) < len(outpath_list):
        raise RuntimeError('Duplicate paths to input dataset found!\n'
                           'Try distinguish inputs further. Otherwise report this '
                           'bug @ github.com/raamana/neuropredict/issues/new')

    print('\nData import is done.\n\n')

    return method_names, outpath_list


def prepare_and_run(subjects, classes, out_dir, options_path,
                    user_feature_paths, user_feature_type, fs_subject_dir,
                    train_perc, num_rep_cv, positive_class,
                    sub_group_list, feature_selection_size,
                    impute_strategy, num_procs,
                    grid_search_level, classifier, feat_select_method):
    "Organizes the inputs and prepares them for CV"

    feature_dir, method_list = make_method_list(fs_subject_dir, user_feature_paths,
                                                user_feature_type)
    method_names, outpath_list = import_datasets(method_list, out_dir, subjects,
                                                 classes, feature_dir,
                                                 user_feature_type)

    print('Requested processing for the following subgroups:'
          '\n{}\n'.format('\n'.join([','.join(sg) for sg in sub_group_list])))

    # iterating through the given set of subgroups
    num_sg = len(sub_group_list)
    for sgi, sub_group in enumerate(sub_group_list):
        print('{}\nProcessing subgroup : {} ({}/{})\n{}'
              ''.format('-' * 80, ','.join(sub_group), sgi + 1, num_sg, '-' * 80))
        sub_group_id = sub_group_identifier(sub_group, sg_index=sgi + 1)
        out_dir_sg = pjoin(out_dir, sub_group_id)

        multi_ds = load_datasets(outpath_list, task_type='classify',
                                 subgroup=sub_group, name=sub_group_id)
        print(multi_ds)
        impute_strategy = detect_missing_data(multi_ds, impute_strategy)

        clf_expt = ClassificationWorkflow(datasets=multi_ds,
                                          pred_model=classifier,
                                          impute_strategy=impute_strategy,
                                          dim_red_method=feat_select_method,
                                          reduced_dim=feature_selection_size,
                                          train_perc=train_perc,
                                          num_rep_cv=num_rep_cv,
                                          scoring=cfg.default_metric_set_classification,
                                          positive_class=positive_class,
                                          grid_search_level=grid_search_level,
                                          out_dir=out_dir_sg,
                                          num_procs=num_procs,
                                          user_options=options_path,
                                          checkpointing=cfg.default_checkpointing)

        clf_expt.run()

        # print('\n\nSaving the visualizations to \n{}'.format(out_dir))
        # make_visualizations(results_file_path, out_dir_sg, options_path)
        # print('\n')

    return


def cli():
    """
    Main entry point.

    """

    subjects, classes, out_dir, options_path, user_feature_paths, \
    user_feature_type, \
    fs_subject_dir, train_perc, num_rep_cv, positive_class, sub_group_list, \
    feature_selection_size, impute_strategy, num_procs, \
    grid_search_level, classifier, feat_select_method = parse_args()

    print('Running neuropredict version {} for Classification'.format(__version__))
    prepare_and_run(subjects, classes, out_dir, options_path,
                    user_feature_paths, user_feature_type, fs_subject_dir,
                    train_perc, num_rep_cv, positive_class,
                    sub_group_list, feature_selection_size, impute_strategy,
                    num_procs, grid_search_level, classifier, feat_select_method)

    return


if __name__ == '__main__':
    cli()
