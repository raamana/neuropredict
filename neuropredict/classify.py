"""Module to define classification-oriented workflows and methods"""

import os
import textwrap
from os.path import basename, join as pjoin

import numpy as np
# the order of import is very important to avoid circular imports
from neuropredict import __version__, config as cfg
from neuropredict.base import BaseWorkflow, parse_common_args
from neuropredict.datasets import detect_missing_data, load_datasets
from neuropredict.freesurfer import (aseg_stats_subcortical,
                                     aseg_stats_whole_brain)
from neuropredict.io import (get_arff, get_data_matrix, get_dir_of_dirs,
                             get_features, get_pyradigm, process_arff,
                             process_pyradigm, saved_dataset_matches)
from neuropredict.utils import (check_classifier, check_covariates,
                                make_dataset_filename, not_unspecified, save_options,
                                sub_group_identifier, uniquify_in_order)
from neuropredict.visualize import (compare_distributions, compare_misclf_pairwise,
                                    compare_misclf_pairwise_parallel_coord_plot,
                                    confusion_matrices)
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import auc as auc_sklearn


def area_under_roc(true_labels, predicted_prob_pos_class, pos_class):
    """Wrapper to avoid relying on sklearn automatic inference of positive class!"""

    false_pr, true_pr, _thresholds = roc_curve(true_labels, predicted_prob_pos_class,
                                               pos_label=pos_class)
    return auc_sklearn(false_pr, true_pr)


auc_metric_name = area_under_roc.__name__
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
                 covariates=None,
                 deconfounder=cfg.default_deconfounding_method,
                 dim_red_method=cfg.default_dim_red_method,
                 reduced_dim=cfg.default_reduced_dim_size,
                 train_perc=cfg.default_train_perc,
                 num_rep_cv=cfg.default_num_repetitions,
                 scoring=cfg.default_metric_set_classification,
                 positive_class=None,
                 grid_search_level=cfg.GRIDSEARCH_LEVEL_DEFAULT,
                 out_dir=None,
                 num_procs=cfg.DEFAULT_NUM_PROCS,
                 user_options=None,
                 checkpointing=cfg.default_checkpointing):
        super().__init__(datasets,
                         pred_model=pred_model,
                         impute_strategy=impute_strategy,
                         covariates=covariates,
                         deconfounder=deconfounder,
                         dim_red_method=dim_red_method,
                         reduced_dim=reduced_dim,
                         train_perc=train_perc,
                         num_rep_cv=num_rep_cv,
                         scoring=scoring,
                         grid_search_level=grid_search_level,
                         out_dir=out_dir,
                         num_procs=num_procs,
                         user_options=user_options,
                         checkpointing=checkpointing,
                         workflow_type='classify')

        # order of target_set is crucial, for AUC computation as well as confusion
        # matrix row/column, hence making it a tuple to prevent accidental mutation
        #  ordering them in order of their appearance in also important: uniquify!
        self._target_set = tuple(uniquify_in_order(self.datasets.targets.values()))
        self._positive_class, _ = \
            check_positive_class(self._target_set, positive_class)

        self.results.meta['target_set'] = self._target_set
        self.results.meta['positive_class'] = self._positive_class


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
            # self._positive_class_index is not static as pipeline.classes_ seem
            # to differ in order from self._target_set
            pos_cls_idx = np.nonzero(pipeline.classes_==self._positive_class)[0][0]
            if len(self._target_set) == 2:
                auc = area_under_roc(true_targets,
                                     predicted_prob[:, pos_cls_idx],
                                     self._positive_class)
                self.results.add_metric(run_id, ds_id, auc_metric_name, auc)

        conf_mat = confusion_matrix(true_targets, predicted_targets,
                                    labels=self._target_set)  # to control row order
        self.results.add_diagnostics(run_id, ds_id, conf_mat,
                                     true_targets[predicted_targets != true_targets])


    def summarize(self):
        """Simple summary of the results produced, for logging and user info"""

        print(self.results)


    def visualize(self):
        """Method to produce all the relevant visualizations based on the results
        from this workflow."""

        self._compare_metric_distr()
        self._viz_confusion_matrices()
        self._plot_feature_importance()
        self._identify_freq_misclassified()


    def _compare_metric_distr(self):
        """Main perf comparion plot"""

        for metric, m_data in self.results.metric_val.items():
            metric = metric.lower()
            consolidated = np.empty((self.num_rep_cv, len(m_data)))
            for index, ds_id in enumerate(m_data.keys()):
                consolidated[:, index] = m_data[ds_id]

            fig_out_path = pjoin(self._fig_out_dir, 'compare_{}'.format(metric))
            if 'accuracy' in metric:
                horiz_line_loc = self._chance_accuracy
                horiz_line_label = 'chance'
            elif 'roc' in metric or 'auc' in metric:
                horiz_line_loc = 0.5
                horiz_line_label = 'chance'
            else:
                horiz_line_loc = None
                horiz_line_label = None
            compare_distributions(consolidated, self.datasets.modality_ids,
                                  fig_out_path, y_label=metric,
                                  horiz_line_loc=horiz_line_loc,
                                  horiz_line_label=horiz_line_label,
                                  upper_lim_y=1.01, ytick_step=0.05)


    def _viz_confusion_matrices(self):
        """Confusion matrices for each feature set, as plots of misclf rate"""

        # forcing a tuple to ensure the order, in compound array and in viz's
        ds_id_order = tuple(self.datasets.modality_ids)
        num_datasets = len(ds_id_order)
        num_classes = len(self._target_set)
        conf_mat_all = np.empty((self.num_rep_cv, num_classes, num_classes,
                                 num_datasets))
        for idx, ds in enumerate(ds_id_order):
            for run in range(self.num_rep_cv):
                conf_mat_all[run, :, :, idx] = self.results.confusion_mat[(ds, run)]

        cm_out_fig_path = pjoin(self._fig_out_dir, 'confusion_matrix')
        confusion_matrices(conf_mat_all, self._target_set, ds_id_order,
                           cm_out_fig_path)

        self._compare_misclf_rate(conf_mat_all, ds_id_order, num_classes)


    def _compare_misclf_rate(self, conf_mat_all, method_names, num_classes):
        """Misclassification rate plot"""

        fig_path = pjoin(self._fig_out_dir, 'compare_misclf_rates')
        if num_classes > 2:
            compare_misclf_pairwise(conf_mat_all, self._target_set, method_names,
                                    fig_path)
        elif num_classes == 2:
            compare_misclf_pairwise_parallel_coord_plot(
                    conf_mat_all, self._target_set, method_names, fig_path)


    def _identify_freq_misclassified(self):
        """Diagnostic utility to list frequently misclassified subjects"""

        # TODO pass CVResults data to visualize.freq_hist_misclassifications


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

    user_args, user_feature_paths, user_feature_type, fs_subject_dir, \
    _, meta_data_format, sample_ids, classes, out_dir, train_perc, \
    num_rep_cv, num_procs, reduced_dim_size, impute_strategy, covar_list, \
    covar_method, grid_search_level, dim_red_method = parse_common_args(parser)

    class_set, subgroups, positive_class = validate_class_set(
            classes, user_args.sub_groups, user_args.positive_class)
    classifier = check_classifier(user_args.classifier)

    # saving the validated and expanded values to disk for later use.
    options_to_save = [sample_ids, classes, out_dir, user_feature_paths,
                       user_feature_type, fs_subject_dir, train_perc, num_rep_cv,
                       positive_class, subgroups, reduced_dim_size, num_procs,
                       grid_search_level, classifier, dim_red_method]
    user_options, options_path = save_options(options_to_save, out_dir)

    return sample_ids, classes, out_dir, user_options, \
           user_feature_paths, user_feature_type, fs_subject_dir, \
           train_perc, num_rep_cv, \
           positive_class, subgroups, \
           reduced_dim_size, impute_strategy, num_procs, \
           grid_search_level, classifier, dim_red_method, \
           covar_list, covar_method


def validate_class_set(classes, subgroups, positive_class=None):
    """Ensures class names are valid and sub-groups exist."""

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
    # not retaining it in self._positive_class_index as self._target_set is not
    # guaranteed to be the same as in pipeline.classes_

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


def cli():
    """ Main entry point. """

    print('\nneuropredict version {} for Classification'.format(__version__))
    from datetime import datetime
    init_time = datetime.now()
    print('\tTime stamp : {}\n'.format(init_time.strftime('%Y-%m-%d %H:%M:%S')))

    subjects, classes, out_dir, options_path, user_feature_paths, \
    user_feature_type, fs_subject_dir, train_perc, num_rep_cv, positive_class, \
    sub_group_list, feature_selection_size, impute_strategy, num_procs, \
    grid_search_level, classifier, feat_select_method, covar_list, covar_method = \
        parse_args()

    feature_dir, method_list = make_method_list(fs_subject_dir, user_feature_paths,
                                                user_feature_type)
    # noinspection PyTupleAssignmentBalance
    _, outpath_list = import_datasets(method_list, out_dir, subjects,
                                                 classes, feature_dir,
                                                 user_feature_type)

    print('Requested processing for the following subgroups:'
          '\n{}\n'.format('\n'.join([','.join(sg) for sg in sub_group_list])))

    # iterating through the given set of subgroups
    num_sg = len(sub_group_list)
    result_paths = dict()
    for sgi, sub_group in enumerate(sub_group_list):
        print('{line}\nProcessing subgroup : {id_} ({idx}/{cnt})\n{line}'
              ''.format(line='-' * 80, id_=','.join(sub_group),
                        idx=sgi + 1, cnt=num_sg))
        sub_group_id = sub_group_identifier(sub_group, sg_index=sgi + 1)
        out_dir_sg = pjoin(out_dir, sub_group_id)

        multi_ds = load_datasets(outpath_list, task_type='classify',
                                 subgroup=sub_group, name=sub_group_id)

        covariates, deconfounder = check_covariates(multi_ds, covar_list,
                                                    covar_method)

        print(multi_ds)
        impute_strategy = detect_missing_data(multi_ds, impute_strategy)

        clf_expt = ClassificationWorkflow(datasets=multi_ds,
                                          pred_model=classifier,
                                          impute_strategy=impute_strategy,
                                          dim_red_method=feat_select_method,
                                          covariates=covariates,
                                          deconfounder=deconfounder,
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

        result_paths[sub_group_id] = clf_expt.run()

    timedelta = datetime.now() - init_time
    print('All done. Elapsed time: {} HH:MM:SS\n'.format(timedelta))

    return result_paths


if __name__ == '__main__':
    cli()
